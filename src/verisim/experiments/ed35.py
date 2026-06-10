"""ED35 — the CRDT G-counter: the loss-free, always-available resolution to ED34's lost update.

The DS0-increment-28 experiment for SPEC-7 §3.2 — `cincr`/`cget`, a *state-based* CRDT counter, the
**positive that resolves ED34's negative**. ED34 showed the LWW `incr` silently loses concurrent
increments under eventual consistency. A G-counter fixes this by construction: each node keeps a
per-owner vector of monotone sub-counts, `cincr n key` bumps **only `n`'s own sub-count** (so
concurrent increments at different nodes touch disjoint entries — no conflict, no lost update), and
the CRDT **join is the per-(key, owner) max**, applied by `anti_entropy`/`gossip` — commutative,
associative, idempotent, so it converges regardless of partition or order. And because `cincr` is
purely node-local it is **always available** (a partitioned-alone node still counts — the AP
property the LWW `incr` lacks under `quorum`/`linearizable`). Both panels are measured
dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — loss-free and always available (the resolution to ED34).** Across a cluster-size
    sweep, `cincr` applied `k` times reads back `k` (**1.0**). The direct contrast with ED34: under
    a partition, **three** `cincr`s (two on the majority side, one on the partitioned minority) are
    **all acknowledged** — including the minority one, which a LWW `quorum`/`linearizable` `incr`
    would reject (**always available, 1.0**) — and after `heal`+`gossip` the counter reads exactly
    **3** (**no lost update, 1.0**), where ED34's LWW counter read 2.

  - **Panel B — convergence (the CRDT join).** After the concurrent increments and `heal`, the join
    converges **every** node to the full total: a `gossip` chain spreads it epidemically (**1.0**),
    and `anti_entropy` on each node reaches the same total (**1.0**). The join is **idempotent** — a
    second `gossip` leaves the count unchanged (**1.0**) — the defining CRDT property.

`cincr` is purely node-local (no replication, no in-flight message) and the merge is a coordinator-
level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces every
transition bit-for-bit (§5.2). One omitted-when-empty `gcounters` map + one `GCounterSet` edit; the
op is purely additive — no prior golden/hash/tokenization changes.
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
class ED35Config:
    name: str = "ed35-gcounter"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "c"
    k: int = 3  # number of sequential increments for Panel A

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED35Config:
        b = ED35Config()
        return ED35Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            k=int(d.get("k", b.k)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED35Config:
        return ED35Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED35Result:
    #: Panel A: cincr applied k times reads back k (cluster-size sweep).
    seq_correct_rate: float = 0.0
    n_sizes: int = 0
    k: int = 0
    #: Panel A: a cincr on a partitioned-alone/minority node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel A: three cincrs across a partition all count — after heal+gossip the total is 3.
    no_lost_update: bool = False
    #: Panel B: a gossip chain after heal converges every node to the full total.
    gossip_converges: bool = False
    #: Panel B: anti_entropy on each node after heal converges every node to the full total.
    anti_entropy_converges: bool = False
    #: Panel B: the join is idempotent — a second gossip leaves the count unchanged.
    idempotent: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, counted to k)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed35-{n}n", nodes=nodes, objects=(key,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def run_ed35(cfg: ED35Config | None = None) -> ED35Result:
    cfg = cfg or ED35Config()
    result = ED35Result(k=cfg.k)
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

    # --- Panel A: sequential correctness, always-available, no lost update ------------------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for _ in range(cfg.k):
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cincr n0 {cfg.key}")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"cget n0 {cfg.key}")
        ok = v == str(cfg.k)
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.seq_correct_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # the direct contrast with ED34, on a 5-node cluster: three cincrs across a partition all count.
    config = _config(5, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cincr n0 {cfg.key}")  # majority side -> n0=1
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cincr n0 {cfg.key}")  # majority side -> n0=2
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"cincr n3 {cfg.key}")  # minority side -> n3=1
    result.always_available = s_min == "ok"  # the partitioned-minority cincr is acknowledged (AP)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    sa, sb, _, total = step_both(ref, sysb, sa, sb, f"cget n0 {cfg.key}")
    result.no_lost_update = total == "3"  # 2 (n0) + 1 (n3) — ED34's LWW counter read 2 here

    # --- Panel B: convergence (the CRDT join) ----------------------------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for cmd in ["partition n0 n1 n2 | n3 n4", f"cincr n0 {cfg.key}", f"cincr n0 {cfg.key}",
                    f"cincr n3 {cfg.key}", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        return ref, sysb, sa, sb

    nodes = _config(5, cfg.key).nodes
    # gossip chain: a forward sweep (n0->n4) accumulates the full vector at n4, then a backward pass
    # (n4->n0) distributes it to everyone — the epidemic converges every node on the line.
    ref, sysb, sa, sb = setup_diverged()
    for a, b in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {a} {b}")
    for a, b in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {a} {b}")
    totals = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"cget {nd} {cfg.key}")
        totals.append(v)
    result.gossip_converges = all(t == "3" for t in totals)
    # a second gossip pass is idempotent — the count does not change.
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    sa, sb, _, v_again = step_both(ref, sysb, sa, sb, f"cget n0 {cfg.key}")
    result.idempotent = v_again == "3"

    # anti_entropy on each node also converges every node to the full total.
    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    ae_totals = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"cget {nd} {cfg.key}")
        ae_totals.append(v)
    result.anti_entropy_converges = all(t == "3" for t in ae_totals)

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED35Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"crdt,counts_to_k,{result.seq_correct_rate:.4f},"
                 f"k={result.k}_over_{result.n_sizes}_clusters")
    lines.append(f"crdt,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_cincr_acked_AP")
    lines.append(f"crdt,no_lost_update,"
                 f"{1.0 if result.no_lost_update else 0.0:.4f},three_concurrent_incrs_total_3")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},epidemic_all_nodes_total_3")
    lines.append(f"converge,anti_entropy,"
                 f"{1.0 if result.anti_entropy_converges else 0.0:.4f},all_nodes_total_3")
    lines.append(f"converge,idempotent,"
                 f"{1.0 if result.idempotent else 0.0:.4f},second_gossip_unchanged")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED35 (the CRDT G-counter: loss-free, always-available, convergent)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed35.csv")
    parser.add_argument("--plot", type=str, default="figures/ed35.png")
    args = parser.parse_args()
    cfg = ED35Config.from_json_file(args.config) if args.config else ED35Config()
    result = run_ed35(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — loss-free and always available:")
    print(f"    counts to k={result.k}: {result.seq_correct_rate:.2f} "
          f"| always available (AP): {result.always_available} "
          f"| no lost update (total 3): {result.no_lost_update}")
    print("  Panel B — convergence (the CRDT join):")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed35 import plot_ed35

        plot_ed35(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
