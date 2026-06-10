"""ED24 — voluntary step-down: the graceful handoff, and relinquish-needs-no-quorum.

The DS0-increment-17 experiment for SPEC-7 §3.2 (the consensus family) — `step_down`, the
voluntary counterpart to ED23's involuntary term-fencing. `step_down node` lets the *current*
leader hand back power on its own, leaving the cluster **leaderless at the same term** (where
`elect` deposes a leader by a *higher* term). Two properties, measured dependency-free and
confirmed Tier-A ≡ Tier-B:

  - **Panel A — the voluntary-handoff lifecycle.** Across a sweep of cluster sizes: a leader is
    elected, commits a `propose`, then `step_down`s. The cluster is now **leaderless at the same
    term**, so the same node's next `propose` is rejected (`not_leader`, rate **1.0**) — there is no
    window in which a leaderless cluster commits a consensus write. A fresh `elect` of a *different*
    node installs a successor at a **strictly higher term** (rate **1.0**), who commits (rate
    **1.0**). A clean handoff is exactly `step_down` then `elect <successor>`; the term machinery
    closes the leaderless gap the same way it fences a deposed leader (ED23).

  - **Panel B — authority + partition-independence (relinquish needs no quorum).** Only the
    *current* leader can step down: a non-leader's `step_down` is rejected (`not_leader`, rate
    **1.0**), and a second `step_down` on an already-leaderless cluster is likewise a no-op reject —
    `step_down` is idempotently safe. The sharp case: a leader **stranded in a minority partition
    can still step down** (rate **1.0**) where its `propose` there is `no_quorum` (control, 1.0).
    Giving up authority reads only the node's own leadership, never the medium, so it is always
    safe; *committing* under that authority is not. This is the asymmetry consensus rests on:
    power is cheap to drop and expensive to exercise.

`step_down` touches no replica (leadership is cluster metadata), so the autonomous-actor system
oracle (Tier-B) clears the leader byte-identically and reproduces every transition bit-for-bit (the
W1 retirement, §5.2). The consensus metadata is omitted from the canonical form until the first
election, so the `step_down` op is purely additive — no prior golden, hash, or tokenization changes.
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
class ED24Config:
    name: str = "ed24-step-down"
    #: cluster sizes for the handoff-lifecycle sweep (odd → a clean strict majority).
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "x"
    v1: str = "b"  # the first leader's committed value
    v2: str = "c"  # the successor's committed value

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED24Config:
        b = ED24Config()
        return ED24Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            v1=d.get("v1", b.v1),
            v2=d.get("v2", b.v2),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED24Config:
        return ED24Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED24Result:
    #: Panel A: after step_down the same node's propose is rejected (leaderless, no commit window).
    handoff_leaderless_rate: float = 0.0
    #: Panel A: the successor's election lands at a strictly higher term.
    reelect_higher_term_rate: float = 0.0
    #: Panel A: the legitimate successor commits.
    new_leader_commits_rate: float = 0.0
    n_sizes: int = 0
    #: Panel B: a non-leader's step_down is rejected (only the current leader may relinquish).
    nonleader_stepdown_rejected: bool = False
    #: Panel B: a second step_down on a leaderless cluster is a no-op reject (idempotently safe).
    second_stepdown_rejected: bool = False
    #: Panel B: a minority-stranded leader can still step down (relinquish needs no quorum).
    minority_leader_steps_down: bool = False
    #: Panel B: the control — that same minority leader's propose is no_quorum (commit needs one).
    minority_propose_blocked: bool = False
    #: Tier-B reproduces every step_down/elect/propose transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, int, int]] = field(default_factory=list)  # (n, term_pre, term_post)


def _config(n: int, key: str, vals: tuple[str, ...]) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(
        name=f"ed24-{n}n", nodes=nodes, objects=(key,),
        values=vals, replication_factor=n,
    )


def run_ed24(cfg: ED24Config | None = None) -> ED24Result:
    cfg = cfg or ED24Config()
    result = ED24Result()
    vals = ("a", cfg.v1, cfg.v2, "d")
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

    # --- Panel A: the voluntary-handoff lifecycle -------------------------------------------------
    handoff_leaderless = 0
    reelect_higher = 0
    new_leader_commits = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, vals)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {nodes[0]}")
        term_before = sa.term
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"propose {nodes[0]} {cfg.key} {cfg.v1}")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"step_down {nodes[0]}")
        # leaderless at the same term: the same node's next propose must be rejected.
        _, _, s_propose, _ = step_both(ref, sysb, sa, sb, f"propose {nodes[0]} {cfg.key} {cfg.v2}")
        # a fresh election of a *different* node installs a successor at a strictly higher term.
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {nodes[1]}")
        term_after = sa.term
        _, _, s_commit, _ = step_both(ref, sysb, sa, sb, f"propose {nodes[1]} {cfg.key} {cfg.v2}")
        sizes += 1
        if s_propose == "not_leader":
            handoff_leaderless += 1
        if term_after > term_before:
            reelect_higher += 1
        if s_commit == "ok":
            new_leader_commits += 1
        result.per_size.append((n, term_before, term_after))
    result.handoff_leaderless_rate = handoff_leaderless / sizes if sizes else 0.0
    result.reelect_higher_term_rate = reelect_higher / sizes if sizes else 0.0
    result.new_leader_commits_rate = new_leader_commits / sizes if sizes else 0.0
    result.n_sizes = sizes

    # --- Panel B: authority + partition-independence ----------------------------------------------
    config = _config(3, cfg.key, vals)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)

    # Only the current leader may step down: a non-leader is rejected, and a second step_down on the
    # now-leaderless cluster is likewise a no-op reject (idempotently safe).
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    _, _, s_nonleader, _ = step_both(ref, sysb, sa, sb, "step_down n1")  # n1 is not the leader
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "step_down n0")  # n0 relinquishes -> leaderless
    _, _, s_second, _ = step_both(ref, sysb, sa, sb, "step_down n0")  # no leader to step down
    result.nonleader_stepdown_rejected = s_nonleader == "not_leader"
    result.second_stepdown_rejected = s_second == "not_leader"

    # The sharp case: a leader stranded in the minority can still step down, where its propose there
    # is no_quorum — relinquishing needs no quorum, committing does.
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 | n1 n2")
    _, _, s_minority_propose, _ = step_both(ref, sysb, sa, sb, f"propose n0 {cfg.key} {cfg.v1}")
    _, _, s_minority_stepdown, _ = step_both(ref, sysb, sa, sb, "step_down n0")
    result.minority_propose_blocked = s_minority_propose == "no_quorum"
    result.minority_leader_steps_down = s_minority_stepdown == "stepped_down"

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED24Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"handoff,handoff_leaderless,{result.handoff_leaderless_rate:.4f},"
                 f"propose_rejected_after_step_down_over_{result.n_sizes}_clusters")
    lines.append(f"handoff,reelect_higher_term,{result.reelect_higher_term_rate:.4f},"
                 f"successor_term_strictly_higher_over_{result.n_sizes}_clusters")
    lines.append(f"handoff,new_leader_commits,{result.new_leader_commits_rate:.4f},"
                 f"successor_commits_over_{result.n_sizes}_clusters")
    lines.append(f"authority,nonleader_rejected,"
                 f"{1.0 if result.nonleader_stepdown_rejected else 0.0:.4f},only_leader_steps_down")
    lines.append(f"authority,second_rejected,"
                 f"{1.0 if result.second_stepdown_rejected else 0.0:.4f},idempotently_safe")
    lines.append(f"partition,minority_steps_down,"
                 f"{1.0 if result.minority_leader_steps_down else 0.0:.4f},relinquish_no_quorum")
    lines.append(f"partition,minority_propose_blocked,"
                 f"{1.0 if result.minority_propose_blocked else 0.0:.4f},commit_needs_quorum_ctl")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED24 (voluntary step-down: graceful handoff + relinquish-needs-no-quorum)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed24.csv")
    parser.add_argument("--plot", type=str, default="figures/ed24.png")
    args = parser.parse_args()
    cfg = ED24Config.from_json_file(args.config) if args.config else ED24Config()
    result = run_ed24(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — voluntary-handoff lifecycle (over the cluster-size sweep):")
    print(f"    step_down leaves cluster leaderless (propose rejected): "
          f"{result.handoff_leaderless_rate:.2f} "
          f"| successor term strictly higher: {result.reelect_higher_term_rate:.2f} "
          f"| successor commits: {result.new_leader_commits_rate:.2f}")
    print("  Panel B — authority + partition-independence:")
    print(f"    non-leader step_down rejected: {result.nonleader_stepdown_rejected} "
          f"| second step_down rejected: {result.second_stepdown_rejected}")
    print(f"    minority leader can step down: {result.minority_leader_steps_down} "
          f"(control) its propose is no_quorum: {result.minority_propose_blocked}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed24 import plot_ed24

        plot_ed24(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
