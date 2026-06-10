"""ED26 — Raft log replication: commit-on-majority, and log-matching reconciliation.

The DS0-increment-19 experiment for SPEC-7 §5.1 (the consensus family) — the replicated **log** the
spec named since increment 1, the piece the one-shot `propose` (incr 16) elided. `append node key
val` appends a `(term, index, key, value)` entry to the leader's log, replicates it to the reachable
followers (who adopt the leader's prefix, overwriting any divergent uncommitted tail), and commits
it — folding it into the KV state machine — **iff a majority holds it**. Two properties, measured
dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — commit requires a majority.** Across a cluster-size sweep, an `append` reaching a
    majority commits (the monotone `commit_index` grows, rate **1.0**), while a leader **stranded in
    the minority** appends the entry to its own log but it stays **uncommitted** (commit_index
    unchanged, rate **1.0**) — and the entry is **retained on the leader's log** (rate **1.0**), not
    lost, just not yet durable. The commit index never moves backward (monotone, rate **1.0**).

  - **Panel B — log-matching reconciliation (the safety the one-shot `propose` lacked).** A leader
    appends an entry it cannot commit (minority), is deposed by a higher-term election, and the new
    leader commits a *different* entry at the same index. While uncommitted, the stale entry is
    **never applied to the KV** (rate **1.0**) — the state machine only ever reflects committed
    entries. After the partition heals and the new leader appends again, the deposed leader's
    uncommitted entry is **overwritten** by the committed one (rate **1.0**), all live nodes hold an
    **identical log** (the log-matching property, rate **1.0**), and the rejoined node's KV
    converges to the committed value (rate **1.0**). A committed entry is permanent; an uncommitted
    one is not.

`append` reads the majority from the partition/down medium (a coordinator-level decision, like
`propose`), so the autonomous-actor system oracle (Tier-B) computes byte-identical log/commit/KV
deltas and reproduces every transition bit-for-bit (the W1 retirement, §5.2). The log and commit
index are omitted from the canonical form until the first `append`, so the op is purely additive —
no prior golden/hash/tokenization changes.
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
class ED26Config:
    name: str = "ed26-log"
    #: cluster sizes for the commit-on-majority sweep (odd → a clean strict majority).
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "x"
    #: the committed-then-contended values: a commits, b is the minority leader's lost entry,
    #: c is the higher-term leader's conflicting entry, d is the post-heal entry.
    v_a: str = "a"
    v_b: str = "b"
    v_c: str = "c"
    v_d: str = "d"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED26Config:
        b = ED26Config()
        return ED26Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            v_a=d.get("v_a", b.v_a),
            v_b=d.get("v_b", b.v_b),
            v_c=d.get("v_c", b.v_c),
            v_d=d.get("v_d", b.v_d),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED26Config:
        return ED26Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED26Result:
    #: Panel A: a majority-reachable append commits (commit_index grows).
    majority_commit_rate: float = 0.0
    #: Panel A: a minority-stranded leader's append stays uncommitted (commit_index unchanged).
    minority_uncommitted_rate: float = 0.0
    #: Panel A: the uncommitted entry is still retained on the leader's log (not lost).
    minority_entry_retained_rate: float = 0.0
    #: Panel A: the commit index never moved backward across the sweep.
    commit_index_monotone_rate: float = 0.0
    n_sizes: int = 0
    #: Panel B: while uncommitted, the stale entry is never applied to the KV state machine.
    uncommitted_not_applied: bool = False
    #: Panel B: the deposed leader's uncommitted entry is overwritten by the committed one.
    deposed_entry_overwritten: bool = False
    #: Panel B: after heal, all live nodes hold an identical log (the log-matching property).
    log_matching_after_heal: bool = False
    #: Panel B: the rejoined node's KV converges to the committed value.
    kv_reflects_committed_log: bool = False
    #: Tier-B reproduces every log/commit/KV transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, int, int]] = field(default_factory=list)  # (n, ci_maj, ci_min)


def _config(n: int, key: str, vals: tuple[str, ...]) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(
        name=f"ed26-{n}n", nodes=nodes, objects=(key,),
        values=vals, replication_factor=n,
    )


def run_ed26(cfg: ED26Config | None = None) -> ED26Result:
    cfg = cfg or ED26Config()
    result = ED26Result()
    vals = (cfg.v_a, cfg.v_b, cfg.v_c, cfg.v_d)
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

    # --- Panel A: commit requires a majority ------------------------------------------------------
    maj_commit = 0
    min_uncommitted = 0
    min_retained = 0
    monotone = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, vals)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        leader = nodes[0]
        others = [x for x in nodes if x != leader]
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {leader}")
        # a majority-reachable append commits (commit_index grows from 0 to 1)
        ci_before = sa.commit_index
        sa, sb, s_maj, _ = step_both(ref, sysb, sa, sb, f"append {leader} {cfg.key} {cfg.v_a}")
        ci_majority = sa.commit_index
        # isolate the leader; its next append cannot reach a majority -> uncommitted
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"partition {leader} | {' '.join(others)}")
        ci_pre_min = sa.commit_index
        sa, sb, s_min, idx_min = step_both(ref, sysb, sa, sb,
                                           f"append {leader} {cfg.key} {cfg.v_b}")
        ci_minority = sa.commit_index
        leader_log = sa.logs.get(leader, ())
        sizes += 1
        if s_maj == "appended" and ci_majority > ci_before:
            maj_commit += 1
        if s_min == "uncommitted" and ci_minority == ci_pre_min:
            min_uncommitted += 1
        # the uncommitted entry is on the leader's log at the reported index, carrying v_b
        if leader_log and leader_log[int(idx_min)].value == cfg.v_b:
            min_retained += 1
        if ci_majority >= ci_before and ci_minority >= ci_pre_min:
            monotone += 1
        result.per_size.append((n, ci_majority, ci_minority))
    result.majority_commit_rate = maj_commit / sizes if sizes else 0.0
    result.minority_uncommitted_rate = min_uncommitted / sizes if sizes else 0.0
    result.minority_entry_retained_rate = min_retained / sizes if sizes else 0.0
    result.commit_index_monotone_rate = monotone / sizes if sizes else 0.0
    result.n_sizes = sizes

    # --- Panel B: log-matching reconciliation -----------------------------------------------------
    config = _config(3, cfg.key, vals)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"append n0 {cfg.key} {cfg.v_a}")  # committed a@0
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 | n1 n2")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"append n0 {cfg.key} {cfg.v_b}")  # uncommitted b@1
    # while uncommitted, b is on n0's log but NOT applied to n0's KV (still the committed value a)
    n0_log_has_b = any(e.value == cfg.v_b for e in sa.logs.get("n0", ()))
    result.uncommitted_not_applied = (
        n0_log_has_b and sa.replicas[(cfg.key, "n0")].value == cfg.v_a
    )
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n1")  # term 2 on the majority side
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"append n1 {cfg.key} {cfg.v_c}")  # committed c@1
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"append n1 {cfg.key} {cfg.v_d}")  # n0 reconciles
    logs = {nd: sa.logs.get(nd, ()) for nd in config.nodes}
    # the deposed leader's uncommitted b is gone; index 1 now holds the committed c (term 2)
    n0_log = logs["n0"]
    result.deposed_entry_overwritten = (
        all(e.value != cfg.v_b for e in n0_log)
        and len(n0_log) > 1 and n0_log[1].value == cfg.v_c and n0_log[1].term == 2
    )
    # all live nodes hold an identical log (the log-matching property)
    result.log_matching_after_heal = logs["n0"] == logs["n1"] == logs["n2"]
    # the rejoined node's KV reflects the committed log (the last committed value, d)
    result.kv_reflects_committed_log = sa.replicas[(cfg.key, "n0")].value == cfg.v_d

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED26Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"commit,majority_commit,{result.majority_commit_rate:.4f},"
                 f"commit_index_grows_over_{result.n_sizes}_clusters")
    lines.append(f"commit,minority_uncommitted,{result.minority_uncommitted_rate:.4f},"
                 f"commit_index_unchanged_in_minority")
    lines.append(f"commit,minority_entry_retained,{result.minority_entry_retained_rate:.4f},"
                 f"on_the_leader_log_not_lost")
    lines.append(f"commit,commit_index_monotone,{result.commit_index_monotone_rate:.4f},"
                 f"never_backward")
    lines.append(f"reconcile,uncommitted_not_applied,"
                 f"{1.0 if result.uncommitted_not_applied else 0.0:.4f},kv_only_reflects_committed")
    lines.append(f"reconcile,deposed_entry_overwritten,"
                 f"{1.0 if result.deposed_entry_overwritten else 0.0:.4f},stale_tail_replaced")
    lines.append(f"reconcile,log_matching_after_heal,"
                 f"{1.0 if result.log_matching_after_heal else 0.0:.4f},all_live_logs_identical")
    lines.append(f"reconcile,kv_reflects_committed_log,"
                 f"{1.0 if result.kv_reflects_committed_log else 0.0:.4f},rejoined_node_converges")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED26 (Raft log replication: commit-on-majority + log-matching reconcile)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed26.csv")
    parser.add_argument("--plot", type=str, default="figures/ed26.png")
    args = parser.parse_args()
    cfg = ED26Config.from_json_file(args.config) if args.config else ED26Config()
    result = run_ed26(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — commit requires a majority (over the cluster-size sweep):")
    print(f"    majority append commits: {result.majority_commit_rate:.2f} "
          f"| minority append uncommitted: {result.minority_uncommitted_rate:.2f} "
          f"| entry retained on log: {result.minority_entry_retained_rate:.2f} "
          f"| commit index monotone: {result.commit_index_monotone_rate:.2f}")
    print("  Panel B — log-matching reconciliation:")
    print(f"    uncommitted not applied to KV: {result.uncommitted_not_applied} "
          f"| deposed entry overwritten: {result.deposed_entry_overwritten}")
    print(f"    logs identical after heal: {result.log_matching_after_heal} "
          f"| rejoined KV converges: {result.kv_reflects_committed_log}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed26 import plot_ed26

        plot_ed26(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
