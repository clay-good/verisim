"""ED34 — the atomic counter: read-modify-write and the lost-update problem under eventual.

The DS0-increment-27 experiment for SPEC-7 §3.2 — `incr`, the first *read-modify-write* client op
(`put`/`cas`/`delete` are blind or compare writes), and the canonical case where eventual
last-writer-wins **silently loses updates**. An `incr node key` reads the coordinator's local count
(a non-numeric/absent value is `0`) and writes `count + 1` at a bumped version, reusing the `put`
replication path. Sequentially it is correct, but under a partition the consistency model decides
whether concurrent increments survive — and this is *harder* than the blind-write CAP tradeoff
(ED14), because LWW **loses** a read-modify-write update where it merely makes a blind write
**stale**. Both panels are measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — sequential correctness.** Across a cluster-size sweep (eventual, full connectivity),
    `incr` applied `k` times leaves the counter at exactly `k` (rate **1.0**); and on one cluster
    the same `k`-increment sequence is correct under **all three** consistency models (eventual /
    quorum / linearizable) — when there is no concurrency, every model counts right.

  - **Panel B — the read-modify-write CAP tradeoff under partition.** Two `incr`s on opposite sides
    of a partition: under **`eventual`** both are **acknowledged** yet the count ends up **short by
    one** — a *lost update* (the danger: the client believes both succeeded, **1.0**); under
    **`quorum`** the minority side is **`unavailable`** so only the accepted increment is counted —
    **no silent loss** (**1.0**); under **`linearizable`** an `incr` under *any* partition is
    **`unavailable`** (CP, **1.0**). The counter is the textbook reason "you can't build a correct
    counter on last-writer-wins" — recovered here as a first-class negative.

`incr` reuses the `put` write path (same consistency-model replication, same in-flight medium), so
the autonomous-actor system oracle (Tier-B) reproduces every transition — including the lost update
— bit-for-bit (§5.2). It adds no state field and no edit type (the counter is just a digit-valued
replica), so the op is purely additive — no prior golden/hash/tokenization changes. (A loss-free
eventual counter needs a CRDT/PN-counter — a deferred later increment.)
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
class ED34Config:
    name: str = "ed34-incr"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "c"
    k: int = 3  # number of sequential increments for Panel A

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED34Config:
        b = ED34Config()
        return ED34Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            k=int(d.get("k", b.k)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED34Config:
        return ED34Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED34Result:
    #: Panel A: incr applied k times leaves the counter at exactly k (cluster-size sweep, eventual).
    seq_correct_rate: float = 0.0
    n_sizes: int = 0
    k: int = 0
    #: Panel A: the k-increment sequence is correct under all three consistency models.
    seq_correct_all_models: bool = False
    #: Panel B: under eventual, two concurrent incrs are both acked yet the count is short by one.
    eventual_lost_update: bool = False
    #: Panel B: under quorum, the minority incr is unavailable — only accepted increments count.
    quorum_no_silent_loss: bool = False
    #: Panel B: under linearizable, an incr under any partition is unavailable (CP).
    linearizable_unavailable: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, counted to k)


def _config(n: int, key: str, model: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed34-{n}n", nodes=nodes, objects=(key,), values=("a",),
                      replication_factor=n, consistency_model=model)


def run_ed34(cfg: ED34Config | None = None) -> ED34Result:
    cfg = cfg or ED34Config()
    result = ED34Result(k=cfg.k)
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

    # --- Panel A: sequential correctness ----------------------------------------------------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, "eventual")
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for _ in range(cfg.k):
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"get n0 {cfg.key}")
        ok = v == str(cfg.k)
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.seq_correct_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # the k-increment sequence is correct under all three models (full connectivity).
    all_models = True
    for model in ("eventual", "quorum", "linearizable"):
        config = _config(3, cfg.key, model)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for _ in range(cfg.k):
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"get n0 {cfg.key}")
        if v != str(cfg.k):
            all_models = False
    result.seq_correct_all_models = all_models

    # --- Panel B: the read-modify-write CAP tradeoff under partition ------------------------------
    # eventual: two concurrent incrs both acked, count short by one (the lost update).
    config = _config(5, cfg.key, "eventual")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")  # count 1
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "advance 5")            # replicate the 1 everywhere
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, sa_st, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")  # majority side -> 2
    sa, sb, sb_st, _ = step_both(ref, sysb, sa, sb, f"incr n3 {cfg.key}")  # minority -> 2 (lost)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "advance 5")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "anti_entropy n0")  # converge the merge
    sa, sb, _, final = step_both(ref, sysb, sa, sb, f"get n0 {cfg.key}")
    # three incrs were acknowledged (1 + 2 concurrent), but the count converged to 2: a lost update.
    result.eventual_lost_update = sa_st == "ok" and sb_st == "ok" and final == "2"

    # quorum: the minority incr is unavailable, so only accepted increments count (no silent loss).
    config = _config(5, cfg.key, "quorum")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")  # count 1
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "advance 5")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, q_maj, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")  # majority -> 2
    sa, sb, q_min, _ = step_both(ref, sysb, sa, sb, f"incr n3 {cfg.key}")  # minority -> unavailable
    sa, sb, _, q_final = step_both(ref, sysb, sa, sb, f"get n0 {cfg.key}")
    result.quorum_no_silent_loss = q_maj == "ok" and q_min == "unavailable" and q_final == "2"

    # linearizable: an incr under any partition is unavailable (CP).
    config = _config(5, cfg.key, "linearizable")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")  # count 1 (full connectivity)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 | n1 n2 n3 n4")
    sa, sb, lin_st, _ = step_both(ref, sysb, sa, sb, f"incr n0 {cfg.key}")
    result.linearizable_unavailable = lin_st == "unavailable"

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED34Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"sequential,counts_to_k,{result.seq_correct_rate:.4f},"
                 f"k={result.k}_over_{result.n_sizes}_clusters")
    lines.append(f"sequential,all_models,"
                 f"{1.0 if result.seq_correct_all_models else 0.0:.4f},eventual_quorum_lin")
    lines.append(f"cap,eventual_lost_update,"
                 f"{1.0 if result.eventual_lost_update else 0.0:.4f},two_acked_count_short_one")
    lines.append(f"cap,quorum_no_silent_loss,"
                 f"{1.0 if result.quorum_no_silent_loss else 0.0:.4f},minority_unavailable_no_loss")
    lines.append(f"cap,linearizable_unavailable,"
                 f"{1.0 if result.linearizable_unavailable else 0.0:.4f},cp_rejects_partition")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED34 (the atomic counter: incr + the lost-update problem under eventual)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed34.csv")
    parser.add_argument("--plot", type=str, default="figures/ed34.png")
    args = parser.parse_args()
    cfg = ED34Config.from_json_file(args.config) if args.config else ED34Config()
    result = run_ed34(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — sequential correctness:")
    print(f"    counts to k={result.k}: {result.seq_correct_rate:.2f} "
          f"| correct under all 3 models: {result.seq_correct_all_models}")
    print("  Panel B — the read-modify-write CAP tradeoff:")
    print(f"    eventual lost update: {result.eventual_lost_update} "
          f"| quorum no silent loss: {result.quorum_no_silent_loss} "
          f"| linearizable unavailable: {result.linearizable_unavailable}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed34 import plot_ed34

        plot_ed34(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
