"""ED31 — the config push: "will this config push break the cluster?"

The DS0-increment-24 experiment for SPEC-7 §3.2 — `config_push`, the config-management admin op that
answers the spec's *other* headline operational question (the sibling of ED29's `deploy`). Unlike
`deploy` (a node-local version *label* that gates consensus *compatibility*), a config push is a
**leader-committed, majority-replicated** cluster setting — a Raft-style config entry — so it shares
the leader-fence + majority-reachability rule of `propose`/`append`. Two properties, measured
dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — leader-committed rollout + the leader fence.** Across a cluster-size sweep, a
    `config_push` at the elected leader with full connectivity **commits and reaches every voting
    member** (rate **1.0**). The fence: a push by a **non-leader** is `not_leader` (**1.0**), and a
    push with **no leader elected** is rejected too (`not_leader`, empty current leader — **1.0**).
    Config changes go through consensus, not any node that asks.

  - **Panel B — the partition: will this config push break the cluster?** A leader **stranded in
    the minority** gets `no_quorum` — the push **cannot commit and no node's config changes** (rate
    **1.0**, the all-or-nothing rule: a minority side never installs a value it cannot hold). A
    leader on the **majority** side **commits**, but the value reaches only the reachable majority,
    so the **partitioned minority retains its stale config** — *config divergence*, the broken-
    cluster outcome (**1.0**). The repair: after `heal`, a **re-push converges every node** (1.0).

`config_push` reads its commit quorum from the partition/down medium (a coordinator-level decision,
like `propose`/`append`), so the autonomous-actor system oracle (Tier-B) reproduces every config
transition — including the divergence-under-partition — bit-for-bit (§5.2). The `config` map joins
the observable `cluster_view`, and is omitted from the canonical form until the first push, so the
op is purely additive — no prior golden/hash/tokenization changes.
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
class ED31Config:
    name: str = "ed31-config-push"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "feature"
    val: str = "on"
    val2: str = "v2"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED31Config:
        b = ED31Config()
        return ED31Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            val=d.get("val", b.val),
            val2=d.get("val2", b.val2),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED31Config:
        return ED31Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED31Result:
    #: Panel A: a leader push with full connectivity commits and reaches every voting member.
    commit_full_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a push by a non-leader is fenced (`not_leader`).
    nonleader_fenced: bool = False
    #: Panel A: a push with no leader elected is fenced too.
    noleader_fenced: bool = False
    #: Panel B: a minority-stranded leader gets `no_quorum` and changes no node's config.
    minority_no_quorum: bool = False
    #: Panel B: a majority-side push commits but the partitioned minority retains its stale config.
    minority_stale_under_partition: bool = False
    #: Panel B: after heal, a re-push converges every node to the new value.
    repush_converges: bool = False
    #: Tier-B reproduces every config transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, committed-on-all)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed31-{n}n", nodes=nodes, objects=(key,), replication_factor=n)


def run_ed31(cfg: ED31Config | None = None) -> ED31Result:
    cfg = cfg or ED31Config()
    result = ED31Result()
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

    # --- Panel A: leader-committed rollout + the leader fence -------------------------------------
    commit_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {nodes[0]}")
        push = f"config_push {nodes[0]} {cfg.key} {cfg.val}"
        sa, sb, s_push, _ = step_both(ref, sysb, sa, sb, push)
        # committed, and the value landed on *every* voting member (full connectivity).
        on_all = all(sa.config.get((nd, cfg.key)) == cfg.val for nd in nodes)
        committed = s_push == "committed" and on_all
        sizes += 1
        if committed:
            commit_count += 1
        result.per_size.append((n, committed))
    result.commit_full_rate = commit_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # the leader fence, on a 5-node cluster.
    config = _config(5, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    nodes = config.nodes
    sa = sb = DistributedState.initial(config)
    # no leader yet -> a push is rejected.
    push0 = f"config_push {nodes[0]} {cfg.key} {cfg.val}"
    sa, sb, s_noleader, _ = step_both(ref, sysb, sa, sb, push0)
    result.noleader_fenced = s_noleader == "not_leader"
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {nodes[0]}")
    # a non-leader push -> not_leader.
    sa, sb, s_nonleader, _ = step_both(
        ref, sysb, sa, sb, f"config_push {nodes[1]} {cfg.key} {cfg.val}"
    )
    result.nonleader_fenced = s_nonleader == "not_leader" and (nodes[1], cfg.key) not in sa.config

    # --- Panel B: the partition — will this config push break the cluster? ------------------------
    config = _config(5, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    nodes = config.nodes  # n0..n4
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {nodes[0]}")  # n0 leader
    # 1) a leader stranded in the minority: partition {n0,n1} | {n2,n3,n4}, n0 still leader.
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 | n2 n3 n4")
    before = dict(sa.config)
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"config_push n0 {cfg.key} {cfg.val}")
    # no_quorum, and not a single node's config changed (the all-or-nothing rule).
    result.minority_no_quorum = s_min == "no_quorum" and dict(sa.config) == before

    # 2) a leader on the majority side: heal, partition the other way (n0 on the majority).
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")  # n0 on the majority
    sa, sb, s_maj, _ = step_both(ref, sysb, sa, sb, f"config_push n0 {cfg.key} {cfg.val2}")
    majority = ("n0", "n1", "n2")
    minority = ("n3", "n4")
    majority_has_new = all(sa.config.get((nd, cfg.key)) == cfg.val2 for nd in majority)
    minority_stale = all(sa.config.get((nd, cfg.key)) != cfg.val2 for nd in minority)
    result.minority_stale_under_partition = (
        s_maj == "committed" and majority_has_new and minority_stale
    )

    # 3) the repair: heal + re-push converges every node.
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, s_re, _ = step_both(ref, sysb, sa, sb, f"config_push n0 {cfg.key} {cfg.val2}")
    result.repush_converges = s_re == "committed" and all(
        sa.config.get((nd, cfg.key)) == cfg.val2 for nd in nodes
    )

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED31Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"rollout,commit_full,{result.commit_full_rate:.4f},"
                 f"reaches_all_voters_over_{result.n_sizes}_clusters")
    lines.append(f"rollout,nonleader_fenced,"
                 f"{1.0 if result.nonleader_fenced else 0.0:.4f},non_leader_push_not_leader")
    lines.append(f"rollout,noleader_fenced,"
                 f"{1.0 if result.noleader_fenced else 0.0:.4f},push_with_no_leader_rejected")
    lines.append(f"partition,minority_no_quorum,"
                 f"{1.0 if result.minority_no_quorum else 0.0:.4f},minority_leader_cannot_commit")
    lines.append(f"partition,minority_stale,"
                 f"{1.0 if result.minority_stale_under_partition else 0.0:.4f},"
                 f"majority_commits_minority_keeps_stale_config")
    lines.append(f"partition,repush_converges,"
                 f"{1.0 if result.repush_converges else 0.0:.4f},re_push_after_heal_converges_all")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description='Run ED31 (config push: "will this config push break the cluster?").'
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed31.csv")
    parser.add_argument("--plot", type=str, default="figures/ed31.png")
    args = parser.parse_args()
    cfg = ED31Config.from_json_file(args.config) if args.config else ED31Config()
    result = run_ed31(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — leader-committed rollout + the leader fence:")
    print(f"    commit reaches all voters: {result.commit_full_rate:.2f} "
          f"| non-leader fenced: {result.nonleader_fenced} "
          f"| no-leader fenced: {result.noleader_fenced}")
    print("  Panel B — the partition (will this config push break the cluster?):")
    print(f"    minority no_quorum: {result.minority_no_quorum} "
          f"| minority stale under partition: {result.minority_stale_under_partition} "
          f"| re-push after heal converges: {result.repush_converges}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed31 import plot_ed31

        plot_ed31(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
