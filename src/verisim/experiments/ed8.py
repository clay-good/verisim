"""ED8 — the OCC transaction commit/abort frontier under contention (SPEC-7 §3.2, DS0 increment 2).

The structural experiment that pins the transaction substrate. ``K`` concurrent multi-key
transactions each read-then-write a key drawn uniformly from ``M`` objects; they all open and read
the pre-state, then commit in order. Under optimistic concurrency control (first-committer-wins),
for each object exactly the first committer touching it succeeds and the rest **abort** (the OCC
``conflict``). So the committed count equals the number of *distinct objects touched*, whose
expectation is the classic **balls-in-bins occupancy law**:

    E[commits] = M · (1 − (1 − 1/M)^K)        commit_rate = E[commits] / K

ED8 measures the realized commit rate as the contention dial ``M`` (number of objects) is swept at
fixed concurrency ``K`` and shows it tracks this closed form — a from-scratch, dependency-free
verification that the OCC semantics are exactly right, not merely plausible. A second panel confirms
the transaction layer **composes with Tier-B**: the autonomous-actor system oracle (which delivers
the committed writes' replication on ``advance``) reproduces Tier-A's observable cluster bit-for-bit
across every scenario, so transactions inherit the ED7 W1 retirement for free.

Dependency-free and GPU-free: pure oracle stepping + a seeded key assignment, no torch.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import scaled_dist_config
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class ED8Config:
    name: str = "ed8"
    n_nodes: int = 3  # >1 so committed writes replicate and Tier-B's actor delivery is exercised
    n_txns: int = 8  # K — the concurrency (number of simultaneously-open transactions)
    object_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8)  # M — the contention dial (fewer = hotter)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    values: tuple[str, ...] = ("a", "b", "c", "d")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED8Config:
        b = ED8Config()
        return ED8Config(
            name=d.get("name", b.name),
            n_nodes=d.get("n_nodes", b.n_nodes),
            n_txns=d.get("n_txns", b.n_txns),
            object_counts=tuple(d.get("object_counts", b.object_counts)),
            seeds=tuple(d.get("seeds", b.seeds)),
            values=tuple(d.get("values", b.values)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED8Config:
        return ED8Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED8Result:
    rows: list[dict[str, Any]] = field(default_factory=list)


def _occupancy_rate(m: int, k: int) -> float:
    """The expected commit rate E[commits]/K = M(1-(1-1/M)^K)/K (balls-in-bins occupancy)."""
    return m * (1.0 - (1.0 - 1.0 / m) ** k) / k


def _contention_trajectory(cfg: ED8Config, m: int, rng: random.Random) -> list[str]:
    """K transactions, each reading+writing one object drawn uniformly from M; all commit in order.

    All read the pre-state before any commit, so per object the first committer wins and the rest
    conflict — exactly the OCC race the occupancy law counts.
    """
    config = scaled_dist_config(cfg.n_nodes, n_objects=m)
    node = config.nodes[0]
    keys = [f"o{rng.randrange(m)}" for _ in range(cfg.n_txns)]
    cmds = [f"begin {node} t{i}" for i in range(cfg.n_txns)]
    cmds += [f"tget {node} t{i} {keys[i]}" for i in range(cfg.n_txns)]  # all read the pre-state
    cmds += [f"tput {node} t{i} {keys[i]} {rng.choice(cfg.values)}" for i in range(cfg.n_txns)]
    cmds += [f"commit {node} t{i}" for i in range(cfg.n_txns)]
    return cmds


def _measure(cfg: ED8Config, m: int, seed: int) -> tuple[int, int, bool]:
    """Run one contention scenario; return (commits, conflicts, tier_a_equals_tier_b)."""
    config = scaled_dist_config(cfg.n_nodes, n_objects=m)
    ref = ReferenceDistOracle(config)
    sys_oracle = SystemDistOracle(config)
    rng = random.Random(seed * 1000 + m)
    cmds = _contention_trajectory(cfg, m, rng)
    s = DistributedState.initial(config)
    s_sys = DistributedState.initial(config)
    commits = conflicts = 0
    agree = True
    for cmd in cmds:
        action = parse_dist_action(cmd)
        r = ref.step(s, action)
        r_sys = sys_oracle.step(s_sys, action)
        if cluster_view(r.state) != cluster_view(r_sys.state):
            agree = False
        if r.status == "committed":
            commits += 1
        elif r.status == "conflict":
            conflicts += 1
        s, s_sys = r.state, r_sys.state
    return commits, conflicts, agree


def run_ed8(cfg: ED8Config | None = None) -> ED8Result:
    cfg = cfg or ED8Config()
    rows: list[dict[str, Any]] = []
    for m in cfg.object_counts:
        rates: list[float] = []
        all_agree = True
        commits_total = conflicts_total = 0
        for seed in cfg.seeds:
            commits, conflicts, agree = _measure(cfg, m, seed)
            rates.append(commits / cfg.n_txns)
            commits_total += commits
            conflicts_total += conflicts
            all_agree = all_agree and agree
        lo, hi = bootstrap_ci(rates, seed=0)
        rows.append({
            "objects": m,
            "commit_rate": fmean(rates),
            "ci_lo": lo,
            "ci_hi": hi,
            "occupancy_rate": _occupancy_rate(m, cfg.n_txns),
            "commits": commits_total,
            "conflicts": conflicts_total,
            "tier_b_agrees": all_agree,
        })
    return ED8Result(rows=rows)


CSV_HEADER = "objects,commit_rate,ci_lo,ci_hi,occupancy_rate,commits,conflicts,tier_b_agrees"


def write_csv(result: ED8Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.rows:
        lines.append(f"{r['objects']},{r['commit_rate']:.4f},{r['ci_lo']:.4f},{r['ci_hi']:.4f},"
                     f"{r['occupancy_rate']:.4f},{r['commits']},{r['conflicts']},{r['tier_b_agrees']}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED8 (the OCC transaction commit/abort frontier under contention)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed8.csv")
    parser.add_argument("--plot", type=str, default="figures/ed8.png")
    args = parser.parse_args()
    cfg = ED8Config.from_json_file(args.config) if args.config else ED8Config()
    result = run_ed8(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print(f"  K={cfg.n_txns} concurrent transactions; commit rate vs balls-in-bins occupancy law:")
    for r in result.rows:
        print(f"    M={r['objects']:2d} objects: measured {r['commit_rate']:.3f} "
              f"vs occupancy {r['occupancy_rate']:.3f}  "
              f"(commits {r['commits']}, conflicts {r['conflicts']}, "
              f"Tier-B agrees={r['tier_b_agrees']})")
    try:
        from figures.plot_ed8 import plot_ed8

        plot_ed8(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
