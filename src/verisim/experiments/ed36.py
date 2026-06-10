"""ED36 — the CRDT PN-counter: a decrementable counter that still converges loss-free.

The DS0-increment-29 experiment for SPEC-7 §3.2 — `cdecr`, the decrement that turns ED35's grow-only
G-counter into a full **PN-counter**. A G-counter can only go up; a PN-counter pairs *two*
G-counters, P (the `cincr` half) and N (the `cdecr` half), and reads **P − N**. `cdecr n key` bumps
**only `n`'s own** N sub-count — so, exactly like `cincr`, it is purely node-local (always
available, the AP property), concurrent decrements never conflict, and the per-(key, owner) **max**
join merges *both* halves. So the PN-counter keeps every property of the G-counter (loss-free,
always-available, convergent) while gaining the one it lacked: its value may go **negative**. Both
panels
are measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — decrement works, loss-free, and may go negative (the PN extension).** Across a
    cluster-size sweep, `k` `cincr`s then `m` `cdecr`s read back **k − m** (**1.0**). A `cdecr` on a
    fresh counter reads **−1** — the value goes **below zero**, where a grow-only G-counter cannot
    (**1.0**). A `cdecr` on a partitioned-alone/minority node is acknowledged (**always available**,
    AP, **1.0**). And the direct concurrency contrast: on a partition, **two** `cincr`s (majority)
    and **one** `cdecr` (minority) are **all acknowledged**, and after `heal`+`gossip` the counter
    reads exactly **+2 − 1 = 1** — **no lost update** across *both* halves (**1.0**).

  - **Panel B — convergence (the CRDT join over both halves).** After the concurrent inc/dec and
    `heal`, the join converges **every** node to the net total: a `gossip` chain spreads it
    epidemically (**1.0**), and `anti_entropy` on each node reaches the same net (**1.0**). The join
    is **idempotent** — a second `gossip` leaves the value unchanged (**1.0**).

`cdecr` is purely node-local (no replication, no in-flight message) and the merge is a coordinator-
level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces every transition
bit-for-bit (§5.2). One omitted-when-empty `ncounters` map + one `NCounterSet` edit; the op is
purely additive over increment 28 — no prior golden/hash/tokenization changes.
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
class ED36Config:
    name: str = "ed36-pncounter"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "c"
    k: int = 3  # number of sequential increments for Panel A
    m: int = 1  # number of sequential decrements for Panel A (reads back k - m)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED36Config:
        b = ED36Config()
        return ED36Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            k=int(d.get("k", b.k)),
            m=int(d.get("m", b.m)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED36Config:
        return ED36Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED36Result:
    #: Panel A: k cincrs then m cdecrs reads back k - m (cluster-size sweep).
    net_correct_rate: float = 0.0
    n_sizes: int = 0
    k: int = 0
    m: int = 0
    #: Panel A: a cdecr on a fresh counter reads -1 — the value goes below zero (G-counter cannot).
    goes_negative: bool = False
    #: Panel A: a cdecr on a partitioned-alone/minority node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel A: +2 (majority) and -1 (minority) across a partition all count — net 1 after heal.
    no_lost_update: bool = False
    #: Panel B: a gossip chain after heal converges every node to the net total.
    gossip_converges: bool = False
    #: Panel B: anti_entropy on each node after heal converges every node to the net total.
    anti_entropy_converges: bool = False
    #: Panel B: the join is idempotent — a second gossip leaves the value unchanged.
    idempotent: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, netted to k - m)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed36-{n}n", nodes=nodes, objects=(key,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def run_ed36(cfg: ED36Config | None = None) -> ED36Result:
    cfg = cfg or ED36Config()
    result = ED36Result(k=cfg.k, m=cfg.m)
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

    # --- Panel A: net correctness, goes-negative, always-available, no lost update ----------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for _ in range(cfg.k):
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cincr n0 {cfg.key}")
        for _ in range(cfg.m):
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cdecr n0 {cfg.key}")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"cget n0 {cfg.key}")
        ok = v == str(cfg.k - cfg.m)
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.net_correct_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # the value goes below zero: a cdecr on a fresh counter reads -1 (where a G-counter cannot).
    config = _config(3, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cdecr n0 {cfg.key}")
    sa, sb, _, v_neg = step_both(ref, sysb, sa, sb, f"cget n0 {cfg.key}")
    result.goes_negative = v_neg == "-1"

    # the direct concurrency contrast on 5 nodes: +2 (majority) and -1 (minority) all count.
    config = _config(5, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cincr n0 {cfg.key}")  # majority side -> P[n0]=1
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cincr n0 {cfg.key}")  # majority side -> P[n0]=2
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"cdecr n3 {cfg.key}")  # minority -> N[n3]=1
    result.always_available = s_min == "ok"  # the partitioned-minority cdecr is acknowledged (AP)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    sa, sb, _, net = step_both(ref, sysb, sa, sb, f"cget n0 {cfg.key}")
    result.no_lost_update = net == "1"  # +2 (n0) - 1 (n3): both halves merged, nothing lost

    # --- Panel B: convergence (the CRDT join over both halves) ------------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for cmd in ["partition n0 n1 n2 | n3 n4", f"cincr n0 {cfg.key}", f"cincr n0 {cfg.key}",
                    f"cdecr n3 {cfg.key}", "heal"]:
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
    nets = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"cget {nd} {cfg.key}")
        nets.append(v)
    result.gossip_converges = all(t == "1" for t in nets)
    # a second gossip pass is idempotent — the value does not change.
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    sa, sb, _, v_again = step_both(ref, sysb, sa, sb, f"cget n0 {cfg.key}")
    result.idempotent = v_again == "1"

    # anti_entropy on each node also converges every node to the net total.
    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    ae_nets = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"cget {nd} {cfg.key}")
        ae_nets.append(v)
    result.anti_entropy_converges = all(t == "1" for t in ae_nets)

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED36Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"pncounter,nets_to_k_minus_m,{result.net_correct_rate:.4f},"
                 f"k={result.k}_m={result.m}_over_{result.n_sizes}_clusters")
    lines.append(f"pncounter,goes_negative,"
                 f"{1.0 if result.goes_negative else 0.0:.4f},fresh_cdecr_reads_minus_1")
    lines.append(f"pncounter,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_cdecr_acked_AP")
    lines.append(f"pncounter,no_lost_update,"
                 f"{1.0 if result.no_lost_update else 0.0:.4f},plus2_minus1_net_1")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},epidemic_all_nodes_net_1")
    lines.append(f"converge,anti_entropy,"
                 f"{1.0 if result.anti_entropy_converges else 0.0:.4f},all_nodes_net_1")
    lines.append(f"converge,idempotent,"
                 f"{1.0 if result.idempotent else 0.0:.4f},second_gossip_unchanged")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED36 (the CRDT PN-counter: decrementable, loss-free, convergent)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed36.csv")
    parser.add_argument("--plot", type=str, default="figures/ed36.png")
    args = parser.parse_args()
    cfg = ED36Config.from_json_file(args.config) if args.config else ED36Config()
    result = run_ed36(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — decrement works, loss-free, may go negative:")
    print(f"    nets to k-m={result.k - result.m}: {result.net_correct_rate:.2f} "
          f"| goes negative (-1): {result.goes_negative} "
          f"| always available (AP): {result.always_available} "
          f"| no lost update (net 1): {result.no_lost_update}")
    print("  Panel B — convergence (the CRDT join over both halves):")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed36 import plot_ed36

        plot_ed36(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
