"""ED27 — membership change: the quorum threshold tracks the voting set, and restoring availability.

The DS0-increment-20 experiment for SPEC-7 §3.2 (the consensus family) — the `add_replica` /
`remove_replica` admin ops. They reconfigure the *consensus voting membership* (the nodes that count
toward an election/commit quorum), a leader-committed change, so the **majority threshold follows
the membership**. Two properties, measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — the quorum threshold tracks the membership.** Across a cluster-size sweep, a leader
    partitioned **alone** is a minority of the full cluster, so its `propose` is `no_quorum` (rate
    **1.0**). `remove_replica` the unreachable nodes until the leader is the sole member, and the
    *same* lone leader now commits (a majority of 1, rate **1.0**) — availability with no change in
    reachability, purely from shrinking the voting set. `add_replica` a node back raises the
    threshold again, and the lone leader is `no_quorum` once more (rate **1.0**): the threshold
    moves in both directions with the membership.

  - **Panel B — restore availability after node failure (the operational lever).** A 3-node cluster
    loses 2 nodes to crashes; the lone survivor cannot commit (`no_quorum` at majority-2-of-3, rate
    **1.0**). `remove_replica` the two dead nodes (membership → 1) and the survivor commits again
    (majority-1-of-1, rate **1.0**) — the standard way an operator restores progress after losing
    quorum. The reconfiguration is fenced for safety: the **active leader cannot be removed**
    (`is_leader`, rate **1.0** — step it down first), so a membership change never strands the
    cluster leaderless mid-write.

Membership is coordinator-level cluster metadata (like `term`/`leader`), so the autonomous-actor
system oracle (Tier-B) reproduces every reconfiguration bit-for-bit (the W1 retirement, §5.2). The
voting set is omitted from the canonical form until the first change (the empty set is the "all
nodes vote" sentinel), so the ops are purely additive — no prior golden/hash changes. (Honest
scope: membership here is the *voting* overlay — all config nodes still physically store replicas;
and the change is committed by leader fiat, where real Raft commits it as a log entry under joint
consensus to make *concurrent* reconfigurations safe — deferred.)
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
class ED27Config:
    name: str = "ed27-membership"
    #: cluster sizes for the threshold-tracking sweep (odd → clean majorities).
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "x"
    val: str = "a"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED27Config:
        b = ED27Config()
        return ED27Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            val=d.get("val", b.val),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED27Config:
        return ED27Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED27Result:
    #: Panel A: a lone leader is a minority of the full cluster (propose no_quorum).
    alone_blocked_at_full_rate: float = 0.0
    #: Panel A: after removing the other nodes, the sole member commits (majority of 1).
    sole_member_commits_rate: float = 0.0
    #: Panel A: re-adding a node raises the threshold, blocking the lone leader again.
    regrow_reblocks_rate: float = 0.0
    n_sizes: int = 0
    #: Panel B: with 1 live of 3 members, the survivor cannot commit.
    stuck_before_removal: bool = False
    #: Panel B: after removing the 2 dead nodes, the survivor commits again.
    restored_after_removal: bool = False
    #: Panel B: the active leader cannot be removed (a safety fence).
    active_leader_remove_blocked: bool = False
    #: Tier-B reproduces every reconfiguration transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, str]] = field(default_factory=list)  # (n, leader)


def _config(n: int, key: str, vals: tuple[str, ...]) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(
        name=f"ed27-{n}n", nodes=nodes, objects=(key,),
        values=vals, replication_factor=n,
    )


def run_ed27(cfg: ED27Config | None = None) -> ED27Result:
    cfg = cfg or ED27Config()
    result = ED27Result()
    vals = (cfg.val, "b", "c", "d")
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

    # --- Panel A: the quorum threshold tracks the membership --------------------------------------
    alone_blocked = 0
    sole_commits = 0
    regrow_blocks = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, vals)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        leader = nodes[0]
        others = [x for x in nodes if x != leader]
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {leader}")
        # isolate the leader alone — a strict minority of the full cluster
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"partition {leader} | {' '.join(others)}")
        _, _, s_alone, _ = step_both(ref, sysb, sa, sb, f"propose {leader} {cfg.key} {cfg.val}")
        # remove every other node from the voting set → the leader is the sole member
        for o in others:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"remove_replica {o}")
        _, _, s_sole, _ = step_both(ref, sysb, sa, sb, f"propose {leader} {cfg.key} {cfg.val}")
        # add one node back → the threshold rises to 2; the lone-reachable leader is blocked again
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"add_replica {others[0]}")
        _, _, s_regrow, _ = step_both(ref, sysb, sa, sb, f"propose {leader} {cfg.key} {cfg.val}")
        sizes += 1
        if s_alone == "no_quorum":
            alone_blocked += 1
        if s_sole == "ok":
            sole_commits += 1
        if s_regrow == "no_quorum":
            regrow_blocks += 1
        result.per_size.append((n, leader))
    result.alone_blocked_at_full_rate = alone_blocked / sizes if sizes else 0.0
    result.sole_member_commits_rate = sole_commits / sizes if sizes else 0.0
    result.regrow_reblocks_rate = regrow_blocks / sizes if sizes else 0.0
    result.n_sizes = sizes

    # --- Panel B: restore availability after node failure -----------------------------------------
    config = _config(3, cfg.key, vals)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "crash n1")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "crash n2")
    _, _, s_stuck, _ = step_both(ref, sysb, sa, sb, f"propose n0 {cfg.key} {cfg.val}")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "remove_replica n1")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "remove_replica n2")
    _, _, s_restored, _ = step_both(ref, sysb, sa, sb, f"propose n0 {cfg.key} {cfg.val}")
    result.stuck_before_removal = s_stuck == "no_quorum"
    result.restored_after_removal = s_restored == "ok"

    # the active leader cannot be removed (a safety fence)
    config2 = _config(3, cfg.key, vals)
    ref2, sysb2 = ReferenceDistOracle(config2), SystemDistOracle(config2)
    sa = sb = DistributedState.initial(config2)
    sa, sb, _, _ = step_both(ref2, sysb2, sa, sb, "elect n0")
    _, _, s_leader_rm, _ = step_both(ref2, sysb2, sa, sb, "remove_replica n0")
    result.active_leader_remove_blocked = s_leader_rm == "is_leader"

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED27Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"threshold,alone_blocked_at_full,{result.alone_blocked_at_full_rate:.4f},"
                 f"minority_of_full_cluster_over_{result.n_sizes}")
    lines.append(f"threshold,sole_member_commits,{result.sole_member_commits_rate:.4f},"
                 f"majority_of_one_after_shrink")
    lines.append(f"threshold,regrow_reblocks,{result.regrow_reblocks_rate:.4f},"
                 f"add_replica_raises_threshold")
    lines.append(f"availability,stuck_before_removal,"
                 f"{1.0 if result.stuck_before_removal else 0.0:.4f},one_live_of_three_no_quorum")
    lines.append(f"availability,restored_after_removal,"
                 f"{1.0 if result.restored_after_removal else 0.0:.4f},commits_after_removing_dead")
    lines.append(f"availability,active_leader_remove_blocked,"
                 f"{1.0 if result.active_leader_remove_blocked else 0.0:.4f},is_leader_fence")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED27 (membership change: the quorum threshold tracks the voting set)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed27.csv")
    parser.add_argument("--plot", type=str, default="figures/ed27.png")
    args = parser.parse_args()
    cfg = ED27Config.from_json_file(args.config) if args.config else ED27Config()
    result = run_ed27(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — the quorum threshold tracks the membership (over the cluster-size sweep):")
    print(f"    lone leader blocked at full membership: {result.alone_blocked_at_full_rate:.2f} "
          f"| sole member commits: {result.sole_member_commits_rate:.2f} "
          f"| re-add re-blocks: {result.regrow_reblocks_rate:.2f}")
    print("  Panel B — restore availability after node failure:")
    print(f"    stuck before removal: {result.stuck_before_removal} "
          f"| restored after removal: {result.restored_after_removal} "
          f"| active leader remove blocked: {result.active_leader_remove_blocked}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed27 import plot_ed27

        plot_ed27(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
