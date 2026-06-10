"""ED29 — the rolling upgrade: "will this deploy break the cluster?"

The DS0-increment-22 experiment for SPEC-7 §3.2 — `deploy`, the rolling-upgrade admin op that
answers the question SPEC-7 names in its introduction (§1: *"will this config push break the
cluster?"*). A
node's running software `version` gates **consensus** participation: two nodes interoperate in a
quorum only if their versions are within `max_version_skew` (the default `1` is the standard N-1
rolling-upgrade window). Two properties, measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — the safe rolling upgrade.** Across a cluster-size sweep, roll every node `v0 → v1`
    one at a time; after each single-node bump a `propose` still **commits** (rate **1.0** over all
    intermediate steps). At every instant the version spread is at most `1`, inside the
    compatibility window, so a compatible majority always exists — the upgrade never breaks the
    cluster.

  - **Panel B — the deploy that breaks the cluster (and why).** The cluster is split into two
    *incompatible* version cohorts with **no compatible majority** (e.g. 2 nodes at `v0`, 2 at `v2`,
    skew `2 > 1`): the next `propose` is **`no_quorum`** — the deploy broke the cluster (rate
    **1.0**). The diagnostic contrast isolates the cause: the *same* version assignment is **safe**
    when it stays inside the window — either a smaller spread (`v0`/`v1`, spread `1`) **commits**
    (**1.0**), or a wider configured window (`max_version_skew = 2`) **commits** (**1.0**). It is
    the spread *exceeding the compatibility window* that breaks consensus, not the mere presence of
    mixed versions.

A node's version is cluster metadata read coordinator-side (compatibility gates *consensus* only —
the best-effort KV/queue data plane is version-agnostic), so the autonomous-actor system oracle
(Tier-B) reproduces the version-driven quorum loss bit-for-bit (the W1 retirement, §5.2). Versions
are omitted from the canonical form until the first `deploy`, so the op is purely additive — no
prior golden/hash/tokenization changes.
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
class ED29Config:
    name: str = "ed29-deploy"
    #: cluster sizes for the safe-rolling-upgrade sweep.
    cluster_sizes: tuple[int, ...] = (3, 4, 5)
    key: str = "x"
    val: str = "a"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED29Config:
        b = ED29Config()
        return ED29Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            val=d.get("val", b.val),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED29Config:
        return ED29Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED29Result:
    #: Panel A: fraction of rolling-upgrade steps (propose after each single-node bump) that commit.
    rolling_commit_rate: float = 0.0
    n_sizes: int = 0
    n_steps: int = 0
    #: Panel B: an incompatible split (spread > window, no compatible majority) loses quorum.
    incompatible_split_breaks: bool = False
    #: Panel B: a within-window split (smaller spread) still commits — the spread is what matters.
    within_window_commits: bool = False
    #: Panel B: the *same* over-spread assignment commits under a wider configured window.
    wider_window_commits: bool = False
    #: Tier-B reproduces every deploy / version-quorum transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, int]] = field(default_factory=list)  # (n, commits)


def _config(n: int, key: str, vals: tuple[str, ...], skew: int = 1) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(
        name=f"ed29-{n}n-skew{skew}", nodes=nodes, objects=(key,),
        values=vals, replication_factor=n, consistency_model="quorum", max_version_skew=skew,
    )


def run_ed29(cfg: ED29Config | None = None) -> ED29Result:
    cfg = cfg or ED29Config()
    result = ED29Result()
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

    # --- Panel A: the safe rolling upgrade (v0 -> v1, one node at a time) ------------------------
    commits = 0
    steps = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, vals, skew=1)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {nodes[0]}")
        size_commits = 0
        for node in nodes:  # bump each node v0 -> v1, then probe a consensus write
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"deploy {node} 1")
            probe = f"propose {nodes[0]} {cfg.key} {cfg.val}"
            _, _, s_prop, _ = step_both(ref, sysb, sa, sb, probe)
            steps += 1
            if s_prop == "ok":
                commits += 1
                size_commits += 1
        sizes += 1
        result.per_size.append((n, size_commits))
    result.rolling_commit_rate = commits / steps if steps else 0.0
    result.n_sizes = sizes
    result.n_steps = steps

    # --- Panel B: the deploy that breaks the cluster, and the diagnostic contrast -----------------
    # An incompatible even split with no compatible majority: 4 nodes, 2 at v2 and 2 at v0 (spread 2
    # > skew 1) -> no cohort is a majority -> the next propose is no_quorum (the cluster broke).
    config = _config(4, cfg.key, vals, skew=1)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "deploy n0 2")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "deploy n1 2")  # {n0,n1}=v2 | {n2,n3}=v0, spread 2
    _, _, s_broke, _ = step_both(ref, sysb, sa, sb, f"propose n0 {cfg.key} {cfg.val}")
    result.incompatible_split_breaks = s_broke == "no_quorum"

    # within-window split: the same 2|2 shape but spread 1 (v0/v1) -> compatible majority commits.
    config = _config(4, cfg.key, vals, skew=1)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "deploy n0 1")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "deploy n1 1")  # {n0,n1}=v1 | {n2,n3}=v0, spread 1
    _, _, s_within, _ = step_both(ref, sysb, sa, sb, f"propose n0 {cfg.key} {cfg.val}")
    result.within_window_commits = s_within == "ok"

    # wider window: the same over-spread (v2/v0) assignment, but max_version_skew = 2 -> compatible.
    config = _config(4, cfg.key, vals, skew=2)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "deploy n0 2")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "deploy n1 2")
    _, _, s_wider, _ = step_both(ref, sysb, sa, sb, f"propose n0 {cfg.key} {cfg.val}")
    result.wider_window_commits = s_wider == "ok"

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED29Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"rolling,rolling_commit_rate,{result.rolling_commit_rate:.4f},"
                 f"commits_over_{result.n_steps}_upgrade_steps")
    lines.append(f"break,incompatible_split_breaks,"
                 f"{1.0 if result.incompatible_split_breaks else 0.0:.4f},spread2_skew1_no_quorum")
    lines.append(f"break,within_window_commits,"
                 f"{1.0 if result.within_window_commits else 0.0:.4f},spread_1_skew_1_ok")
    lines.append(f"break,wider_window_commits,"
                 f"{1.0 if result.wider_window_commits else 0.0:.4f},spread_2_skew_2_ok")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED29 (the rolling upgrade: will this deploy break the cluster?)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed29.csv")
    parser.add_argument("--plot", type=str, default="figures/ed29.png")
    args = parser.parse_args()
    cfg = ED29Config.from_json_file(args.config) if args.config else ED29Config()
    result = run_ed29(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — the safe rolling upgrade (v0 -> v1, one node at a time):")
    print(f"    propose commits at every step: {result.rolling_commit_rate:.2f} "
          f"(over {result.n_steps} upgrade steps)")
    print("  Panel B — the deploy that breaks the cluster (and why):")
    print(f"    incompatible split (spread 2, skew 1) breaks: {result.incompatible_split_breaks} "
          f"| within-window (spread 1) commits: {result.within_window_commits} "
          f"| wider window (skew 2) commits: {result.wider_window_commits}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed29 import plot_ed29

        plot_ed29(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
