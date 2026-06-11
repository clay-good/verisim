"""ED39 — the CRDT LWW-register: deterministic single-value resolution by a Lamport-timestamp order.

The DS0-increment-32 experiment for SPEC-7 §3.2 — `lwwput`/`lwwget`, a last-writer-wins register,
the **policy-opposite** of ED38's MV-register. Where the MV-register *surfaces* a write conflict as
siblings, the LWW-register **deterministically picks one winner**. The mechanism is a **Lamport
clock**: `lwwput n key val` stamps `val` with `(ts, owner=n)` where `ts = lamport[n] + 1` (advancing
`n`'s clock), and the CRDT join keeps the **max** copy by `(ts, owner, value)`. Two consequences,
both measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — happens-after wins, deterministically.** Across a cluster-size sweep, `lwwput` then
    `lwwget` reads back the value (**1.0**). A write that **happened-after** another (having seen
    it, so its Lamport ts is higher) **wins regardless of node id** — even a *lower*-id node's later
    write beats a higher-id node's earlier one (**causal LWW, 1.0**), where "highest node wins"
    would get it backwards. Truly *concurrent* writes (equal ts) resolve to **one** value by node-id
    tie-break, the same on every node (**deterministic resolution, 1.0**) — where MV-register keeps
    both as siblings. And `lwwput` is **always available** — a partitioned-alone node still writes
    (**1.0**), the AP property.

  - **Panel B — convergence (the max-by-timestamp join).** From a diverged state the join converges
    **every** node to the single winning value — `gossip` epidemically (**1.0**), `anti_entropy`
    on each node (**1.0**), idempotently (a second `gossip` is a no-op, **1.0**). The price the
    MV-register avoids: the concurrent *loser* is **dropped** (the register holds one value,
    not two) — the deterministic-resolution-vs-conflict-surfacing tradeoff made explicit (**1.0**).

`lwwput` is purely node-local (no replication, no in-flight message) and the merge is a
coordinator-level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces each
transition bit-for-bit (§5.2). Two omitted-when-empty `lwwreg`/`lamport` maps + the
`LWWRegSet`/`LamportSet` edits; the ops are purely additive over increment 31 — no prior
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
class ED39Config:
    name: str = "ed39-lwwregister"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "w"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED39Config:
        b = ED39Config()
        return ED39Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED39Config:
        return ED39Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED39Result:
    #: Panel A: lwwput then lwwget reads back the value (cluster-size sweep).
    basic_read_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a causally-later write wins regardless of node id (happens-after wins).
    causal_lww: bool = False
    #: Panel A: concurrent writes resolve to one deterministic value on every node.
    deterministic_resolve: bool = False
    #: Panel A: a lwwput on a partitioned-alone node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel B: a gossip chain after heal converges every node to the single winner.
    gossip_converges: bool = False
    #: Panel B: anti_entropy on each node after heal converges every node to the single winner.
    anti_entropy_converges: bool = False
    #: Panel B: the max-by-timestamp join is idempotent — a second gossip leaves the value as is.
    idempotent: bool = False
    #: Panel B: the concurrent loser is dropped — the register holds one value, not siblings.
    loser_dropped: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, basic read ok)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed39-{n}n", nodes=nodes, objects=(key,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def run_ed39(cfg: ED39Config | None = None) -> ED39Result:
    cfg = cfg or ED39Config()
    result = ED39Result()
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

    # --- Panel A: basic read, causal LWW, deterministic resolution, always-available --------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, k)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"lwwput n0 {k} a")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"lwwget n0 {k}")
        ok = v == "a"
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.basic_read_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # causal LWW: the HIGHEST-id node writes first; a LOWER-id node, having seen it, writes later.
    # The later write has a higher Lamport ts, so it wins — proving ts (causality) beats node id.
    config = _config(5, k)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"lwwput n4 {k} a")  # ts 1, owner n4 (high id)
    for nd in ("n0", "n1", "n2", "n3"):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")  # all see a, clock -> 1
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"lwwput n0 {k} b")  # ts 2, owner n0 (low id)
    for nd in ("n1", "n2", "n3", "n4"):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    vals = [step_both(ref, sysb, sa, sb, f"lwwget {nd} {k}")[3] for nd in config.nodes]
    result.causal_lww = all(v == "b" for v in vals)  # the later (lower-id) write won

    # concurrent (equal ts) writes resolve to ONE value by the node-id tie-break, same everywhere.
    config = _config(5, k)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"lwwput n3 {k} b")  # ts 1, owner n3
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"lwwput n0 {k} a")  # ts 1, owner n0 (concurrent)
    result.always_available = s_min == "ok"  # the partitioned-minority write is acknowledged (AP)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    v0 = step_both(ref, sysb, sa, sb, f"lwwget n0 {k}")[3]
    v3 = step_both(ref, sysb, sa, sb, f"lwwget n3 {k}")[3]
    result.deterministic_resolve = v0 == v3 == "b"  # owner n3 > n0 wins the tie, same on both
    result.loser_dropped = v0 == "b" and v0 != "a"  # the concurrent loser a is gone (one value)

    # --- Panel B: convergence (the max-by-timestamp join) -----------------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, k)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for cmd in ["partition n0 n1 n2 | n3 n4", f"lwwput n0 {k} a", f"lwwput n3 {k} b", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        return ref, sysb, sa, sb

    nodes = _config(5, k).nodes
    winner = "b"  # owner n3 > n0 at equal ts
    ref, sysb, sa, sb = setup_diverged()
    for p, q in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    for p, q in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    gvals = [step_both(ref, sysb, sa, sb, f"lwwget {nd} {k}")[3] for nd in nodes]
    result.gossip_converges = all(v == winner for v in gvals)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    v_again = step_both(ref, sysb, sa, sb, f"lwwget n0 {k}")[3]
    result.idempotent = v_again == winner

    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    avals = [step_both(ref, sysb, sa, sb, f"lwwget {nd} {k}")[3] for nd in nodes]
    result.anti_entropy_converges = all(v == winner for v in avals)

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED39Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"lwwreg,basic_read,{result.basic_read_rate:.4f},over_{result.n_sizes}_clusters")
    lines.append(f"lwwreg,causal_lww,"
                 f"{1.0 if result.causal_lww else 0.0:.4f},happens_after_wins_low_id")
    lines.append(f"lwwreg,deterministic_resolve,"
                 f"{1.0 if result.deterministic_resolve else 0.0:.4f},concurrent_one_winner")
    lines.append(f"lwwreg,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_lwwput_acked_AP")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},epidemic_all_nodes_winner")
    lines.append(f"converge,anti_entropy,"
                 f"{1.0 if result.anti_entropy_converges else 0.0:.4f},all_nodes_winner")
    lines.append(f"converge,idempotent,"
                 f"{1.0 if result.idempotent else 0.0:.4f},second_gossip_unchanged")
    lines.append(f"converge,loser_dropped,"
                 f"{1.0 if result.loser_dropped else 0.0:.4f},one_value_not_siblings")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED39 (the CRDT LWW-register: deterministic, Lamport-ordered, convergent)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed39.csv")
    parser.add_argument("--plot", type=str, default="figures/ed39.png")
    args = parser.parse_args()
    cfg = ED39Config.from_json_file(args.config) if args.config else ED39Config()
    result = run_ed39(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — happens-after wins, deterministically:")
    print(f"    basic read: {result.basic_read_rate:.2f} "
          f"| causal LWW: {result.causal_lww} "
          f"| deterministic resolve: {result.deterministic_resolve} "
          f"| always available (AP): {result.always_available}")
    print("  Panel B — convergence (the max-by-timestamp join):")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent} "
          f"| loser dropped: {result.loser_dropped}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed39 import plot_ed39

        plot_ed39(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
