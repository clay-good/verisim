"""ED38 — the CRDT MV-register: concurrent writes surface as siblings, not silent loss.

The DS0-increment-31 experiment for SPEC-7 §3.2 — `mvput`/`mvget`, a *multi-value register*, the
Dynamo/Riak data type that **surfaces** a write conflict instead of silently dropping one. The KV
`put` and the counters resolve concurrent writes by last-writer-wins: one survives, one is lost
(ED14). The MV-register makes a different choice — it **keeps both** as *siblings* and lets a later
write resolve them. It reuses the OR-Set's dot/union machinery: `mvput n key val` tags `val` with a
fresh dot, **tombstones every dot it observes** (a write supersedes the values it saw), and
adds its own; `mvget n key` reads the set of surviving (non-tombstoned) values. Both panels are
measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — conflict surfaced, not lost.** Across a cluster-size sweep, `mvput` then `mvget`
    reads back the single value (**1.0**), and a *sequential* overwrite (the writer observes prior)
    **resolves** to one value (**1.0**). But two *concurrent* `mvput`s on opposite sides of a
    partition — neither observing the other — **both survive** as siblings after `heal`+`gossip`
    (**1.0**), where a LWW `put` would keep only one; the conflict is *visible*. And `mvput` is
    **always available** — a partitioned-alone node still writes (**1.0**), the AP property.

  - **Panel B — convergence and resolution.** From the diverged (sibling) state the union join
    converges **every** node to the same sibling set — a `gossip` chain epidemically (**1.0**),
    `anti_entropy` on each node (**1.0**), idempotently (a second `gossip` is a no-op, **1.0**). A
    later context-aware `mvput` (observing both siblings) **resolves** them: after convergence every
    node reads the single new value (**1.0**) — the Dynamo read-and-resolve.

`mvput` is purely node-local (no replication, no in-flight message) and the merge is a
coordinator-level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces each
transition bit-for-bit (§5.2). Two omitted-when-empty `mvreg_vals`/`mvreg_tombs` maps + the
`MVRegWrite`/`MVRegTomb` edits; the ops are purely additive over increment 30 — no prior
golden/hash/tokenization changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


@dataclass(frozen=True)
class ED38Config:
    name: str = "ed38-mvregister"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "r"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED38Config:
        b = ED38Config()
        return ED38Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED38Config:
        return ED38Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED38Result:
    #: Panel A: mvput then mvget reads back the single value (cluster-size sweep).
    basic_read_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a sequential overwrite resolves to one value.
    sequential_resolves: bool = False
    #: Panel A: two concurrent writes both survive as siblings (vs LWW's silent loss).
    siblings_preserved: bool = False
    #: Panel A: a mvput on a partitioned-alone node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel B: a gossip chain after heal converges every node to the same sibling set.
    gossip_converges: bool = False
    #: Panel B: anti_entropy on each node after heal converges every node to the same sibling set.
    anti_entropy_converges: bool = False
    #: Panel B: the union join is idempotent — a second gossip leaves the set unchanged.
    idempotent: bool = False
    #: Panel B: a later context-aware mvput resolves the siblings to one value (read-resolve).
    resolves_conflict: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, basic read ok)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed38-{n}n", nodes=nodes, objects=(key,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _siblings(value: str) -> set[str]:
    """Parse a ``mvget`` value (``{a,b}`` / ``{}``) back into a set of sibling values."""
    inner = value.strip("{}")
    return set(inner.split(",")) if inner else set()


def run_ed38(cfg: ED38Config | None = None) -> ED38Result:
    cfg = cfg or ED38Config()
    result = ED38Result()
    tier_b_agree = True
    tier_b_steps = 0
    k = cfg.key

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

    # --- Panel A: basic read, sequential resolve, siblings, always-available ----------------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, k)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mvput n0 {k} a")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"mvget n0 {k}")
        ok = _siblings(v) == {"a"}
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.basic_read_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # sequential overwrite resolves to one value (the writer observes the prior, supersedes it).
    config = _config(3, k)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mvput n0 {k} a")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mvput n0 {k} b")
    sa, sb, _, v_seq = step_both(ref, sysb, sa, sb, f"mvget n0 {k}")
    result.sequential_resolves = _siblings(v_seq) == {"b"}

    # concurrent writes on opposite sides of a partition both survive as siblings (vs LWW).
    config = _config(5, k)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"mvput n3 {k} b")  # minority side
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mvput n0 {k} a")  # majority side, concurrent
    result.always_available = s_min == "ok"  # the partitioned-minority write is acknowledged (AP)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    sa, sb, _, v0 = step_both(ref, sysb, sa, sb, f"mvget n0 {k}")
    result.siblings_preserved = _siblings(v0) == {"a", "b"}  # both survived — conflict made visible

    # --- Panel B: convergence and resolution ------------------------------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, k)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for cmd in ["partition n0 n1 n2 | n3 n4", f"mvput n0 {k} a", f"mvput n3 {k} b", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        return ref, sysb, sa, sb

    nodes = _config(5, k).nodes
    target = {"a", "b"}  # the two concurrent writes are siblings until resolved
    # gossip chain: a forward sweep then a backward pass converges every node on the line.
    ref, sysb, sa, sb = setup_diverged()
    for p, q in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    for p, q in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    sets = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"mvget {nd} {k}")
        sets.append(_siblings(v))
    result.gossip_converges = all(m == target for m in sets)
    # a second gossip pass is idempotent — the sibling set does not change.
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    sa, sb, _, v_again = step_both(ref, sysb, sa, sb, f"mvget n0 {k}")
    result.idempotent = _siblings(v_again) == target
    # a context-aware mvput (n0 now sees a and b) RESOLVES the conflict to one value cluster-wide.
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mvput n0 {k} c")
    for p, q in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    for p, q in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    resolved = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"mvget {nd} {k}")
        resolved.append(_siblings(v))
    result.resolves_conflict = all(m == {"c"} for m in resolved)

    # anti_entropy on each node also converges every node to the same sibling set.
    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    ae_sets = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"mvget {nd} {k}")
        ae_sets.append(_siblings(v))
    result.anti_entropy_converges = all(m == target for m in ae_sets)

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED38Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"mvreg,basic_read,{result.basic_read_rate:.4f},"
                 f"over_{result.n_sizes}_clusters")
    lines.append(f"mvreg,sequential_resolves,"
                 f"{1.0 if result.sequential_resolves else 0.0:.4f},overwrite_collapses_to_one")
    lines.append(f"mvreg,siblings_preserved,"
                 f"{1.0 if result.siblings_preserved else 0.0:.4f},concurrent_writes_both_survive")
    lines.append(f"mvreg,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_mvput_acked_AP")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},epidemic_all_nodes_same_siblings")
    lines.append(f"converge,anti_entropy,"
                 f"{1.0 if result.anti_entropy_converges else 0.0:.4f},all_nodes_same_siblings")
    lines.append(f"converge,idempotent,"
                 f"{1.0 if result.idempotent else 0.0:.4f},second_gossip_unchanged")
    lines.append(f"converge,resolves,"
                 f"{1.0 if result.resolves_conflict else 0.0:.4f},context_write_collapses_to_one")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED38 (the CRDT MV-register: conflict-surfacing, convergent, resolvable)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed38.csv")
    parser.add_argument("--plot", type=str, default="figures/ed38.png")
    args = parser.parse_args()
    cfg = ED38Config.from_json_file(args.config) if args.config else ED38Config()
    result = run_ed38(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — conflict surfaced, not lost:")
    print(f"    basic read: {result.basic_read_rate:.2f} "
          f"| sequential resolves: {result.sequential_resolves} "
          f"| siblings preserved: {result.siblings_preserved} "
          f"| always available (AP): {result.always_available}")
    print("  Panel B — convergence and resolution:")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent} "
          f"| resolves conflict: {result.resolves_conflict}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed38 import plot_ed38

        plot_ed38(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
