"""ED28 — the distributed FIFO queue: delivery semantics follow the consistency model.

The DS0-increment-21 experiment for SPEC-7 §3.2 (the client-op family) — `enqueue`/`dequeue`, a
second data type beside the KV store. The headline is that a queue's **delivery semantics are not a
property of the queue but of the consistency model it runs under**: the *same* `enqueue` then
dual-sided `dequeue` under a partition delivers the item a different number of times depending on
the model. Two properties, measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — delivery under partition (one item, both sides dequeue).** An item is enqueued under
    full connectivity (so every replica holds it), the cluster partitions, and each side dequeues:
      - `eventual` delivers it **twice** (at-least-once / **duplicate delivery**) — the head-removal
        on one side never reaches the other, so the peer re-delivers the same item. Available, not
        exactly-once.
      - `quorum` delivers it **once** — the majority side serves the dequeue (exactly-once) while
        the minority is `unavailable`. The realistic CP middle.
      - `linearizable` delivers it **zero** times — both sides lack all-replica reachability, so
        both dequeues are `unavailable`. Never duplicates, but fully unavailable under this split.
    Delivery count `2 → 1 → 0` as the model strengthens: the queue's CAP tradeoff, exactly the KV
    fork-vs-availability story (ED14) in delivery-semantics form.

  - **Panel B — FIFO order + exactly-once on the connected path.** With full connectivity, `enqueue`
    of `a, b, c` then three `dequeue`s returns `a, b, c` **in order** (FIFO, rate **1.0**), each
    **exactly once**, and a fourth `dequeue` is `empty` (rate **1.0**) — a correct FIFO queue when
    the network is whole, under every model.

Queues are fully replicated and the reachable set / availability are read from the medium (a
coordinator-level decision, like the KV write), so the autonomous-actor system oracle (Tier-B)
reproduces every transition — including the duplicate delivery — bit-for-bit (the W1 retirement,
§5.2). Queue replicas are omitted from the canonical form until the first `enqueue`, so the op
family is purely additive — no prior golden/hash/tokenization changes. (Honest scope: queue
replication is
synchronous to the reachable set — there is no async in-flight medium or anti-entropy for queue ops
yet, so a divergent replica is reconciled only by a fresh op that reaches it; the consistency model
gates availability, which is the lever the delivery semantics turn on.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


@dataclass(frozen=True)
class ED28Config:
    name: str = "ed28-queue"
    nodes: tuple[str, ...] = ("n0", "n1", "n2")
    queue: str = "q"
    item: str = "a"  # the single item whose delivery count Panel A measures
    fifo_items: tuple[str, ...] = ("a", "b", "c")  # Panel B's FIFO sequence

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED28Config:
        b = ED28Config()
        return ED28Config(
            name=d.get("name", b.name),
            nodes=tuple(d.get("nodes", b.nodes)),
            queue=d.get("queue", b.queue),
            item=d.get("item", b.item),
            fifo_items=tuple(d.get("fifo_items", b.fifo_items)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED28Config:
        return ED28Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED28Result:
    #: Panel A: deliveries of one item when both partition sides dequeue, per consistency model.
    eventual_deliveries: int = 0
    quorum_deliveries: int = 0
    linearizable_deliveries: int = 0
    #: Panel B: dequeue order equals enqueue order (FIFO), on the connected path.
    fifo_preserved: bool = False
    #: Panel B: each enqueued item delivered exactly once, then empty.
    exactly_once_connected: bool = False
    #: Tier-B reproduces every enqueue/dequeue transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_model: list[tuple[str, int]] = field(default_factory=list)  # (model, deliveries)


def _config(nodes: tuple[str, ...], model: str) -> DistConfig:
    return DistConfig(
        name=f"ed28-{model}", nodes=nodes, objects=("x",),
        values=("a", "b", "c", "d"), replication_factor=len(nodes),
        consistency_model=model,
    )


def run_ed28(cfg: ED28Config | None = None) -> ED28Result:
    cfg = cfg or ED28Config()
    result = ED28Result()
    tier_b_agree = True
    tier_b_steps = 0

    def step_both(
        ref: ReferenceDistOracle, sysb: SystemDistOracle,
        sa: DistributedState, sb: DistributedState, cmd: str,
    ) -> tuple[DistributedState, DistributedState, str, str]:
        nonlocal tier_b_agree, tier_b_steps
        action = parse_dist_action(cmd)
        ra, rb = ref.step(sa, action), sysb.step(sb, action)
        if cluster_view(ra.state) != cluster_view(rb.state) or (ra.status, ra.value) != (
            rb.status, rb.value
        ):
            tier_b_agree = False
        tier_b_steps += 1
        return ra.state, rb.state, ra.status, ra.value

    # --- Panel A: delivery under partition, per consistency model ---------------------------------
    nodes = cfg.nodes
    minority, majority = nodes[:1], nodes[1:]  # n0 | n1 n2 ...: a 1 | (n-1) split
    deliveries_by_model: dict[str, int] = {}
    for model in ("eventual", "quorum", "linearizable"):
        config = _config(nodes, model)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"enqueue {nodes[0]} {cfg.queue} {cfg.item}")
        part = f"partition {' '.join(minority)} | {' '.join(majority)}"
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, part)
        delivered = 0
        # each side attempts to dequeue the item; count how many times it is actually delivered
        sa, sb, s_min, v_min = step_both(ref, sysb, sa, sb, f"dequeue {minority[0]} {cfg.queue}")
        if s_min == "dequeued" and v_min == cfg.item:
            delivered += 1
        sa, sb, s_maj, v_maj = step_both(ref, sysb, sa, sb, f"dequeue {majority[0]} {cfg.queue}")
        if s_maj == "dequeued" and v_maj == cfg.item:
            delivered += 1
        deliveries_by_model[model] = delivered
        result.per_model.append((model, delivered))
    result.eventual_deliveries = deliveries_by_model["eventual"]
    result.quorum_deliveries = deliveries_by_model["quorum"]
    result.linearizable_deliveries = deliveries_by_model["linearizable"]

    # --- Panel B: FIFO order + exactly-once on the connected path ---------------------------------
    config = _config(nodes, "eventual")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    for it in cfg.fifo_items:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"enqueue {nodes[0]} {cfg.queue} {it}")
    got: list[str] = []
    for _ in range(len(cfg.fifo_items)):
        sa, sb, status, value = step_both(ref, sysb, sa, sb, f"dequeue {nodes[0]} {cfg.queue}")
        if status == "dequeued":
            got.append(value)
    _, _, s_empty, _ = step_both(ref, sysb, sa, sb, f"dequeue {nodes[0]} {cfg.queue}")
    result.fifo_preserved = got == list(cfg.fifo_items)
    result.exactly_once_connected = (
        sorted(got) == sorted(cfg.fifo_items) and len(got) == len(set(got)) and s_empty == "empty"
    )

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED28Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"delivery,eventual_deliveries,{result.eventual_deliveries},at_least_once_duplicate")
    lines.append(f"delivery,quorum_deliveries,{result.quorum_deliveries},exactly_once_on_majority")
    lines.append(f"delivery,linearizable_deliveries,{result.linearizable_deliveries},"
                 f"cp_unavailable_no_duplicate")
    lines.append(f"fifo,fifo_preserved,{1.0 if result.fifo_preserved else 0.0:.4f},"
                 f"dequeue_order_equals_enqueue_order")
    lines.append(f"fifo,exactly_once_connected,{1.0 if result.exactly_once_connected else 0.0:.4f},"
                 f"each_item_once_then_empty")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED28 (the distributed FIFO queue: delivery semantics by model)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed28.csv")
    parser.add_argument("--plot", type=str, default="figures/ed28.png")
    args = parser.parse_args()
    cfg = ED28Config.from_json_file(args.config) if args.config else ED28Config()
    result = run_ed28(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — deliveries of one item when both partition sides dequeue:")
    print(f"    eventual: {result.eventual_deliveries} (at-least-once / duplicate) "
          f"| quorum: {result.quorum_deliveries} (exactly-once on majority) "
          f"| linearizable: {result.linearizable_deliveries} (CP unavailable)")
    print("  Panel B — FIFO + exactly-once on the connected path:")
    print(f"    FIFO order preserved: {result.fifo_preserved} "
          f"| exactly-once then empty: {result.exactly_once_connected}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed28 import plot_ed28

        plot_ed28(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
