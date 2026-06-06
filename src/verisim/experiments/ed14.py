"""ED14 — the quorum (Raft-subset) consensus model: availability frontier + split-brain prevention.

The DS0-increment-7 experiment for SPEC-7 §3.4 — the `quorum` consistency model, the realistic CP
middle real consensus protocols (Raft, Paxos) occupy, between `eventual` (always available, but
split-brain-prone) and `linearizable` (no divergence, but unavailable under *any* partition).
`quorum` commits **synchronously to a reachable majority** and rejects a write that cannot reach
one; the unreachable minority catches up asynchronously. Two findings, dependency-free, on a
5-node cluster (every node replicates every object, so a strict majority is **3**):

  - **Panel A — the availability frontier.** Partition the cluster into a ``k``-node side and a
    ``5-k``-node side and issue a write from the ``k``-side coordinator. A write commits iff its
    side can reach enough replicas: **eventual** always (it never coordinates), **quorum** iff
    ``k >= 3``
    (the majority frontier), **linearizable** only at ``k = 5`` (no partition at all — it needs
    *every* replica). The step in the quorum curve at ``k = 3`` is the consensus availability
    boundary — quorum stays available on the majority side exactly where linearizable goes dark.

  - **Panel B — split-brain prevention.** Under the same partition, have **both** sides write the
    same key, then ask whether the object **forks** (two replicas hold the *same version* with
    *different values* — a divergent committed write, the split-brain ED11's version oracle catches
    black-box). **eventual** forks (both sides commit, the object diverges); **quorum** does not
    (only the majority side commits, so there is a single committed value and the minority is merely
    stale); neither does **linearizable** (neither side commits — safety bought with downtime).
    Quorum is the only model that is *both* available (on the majority side) *and* divergence-free —
    the reason real systems use majority quorums rather than all-replica synchrony.

Both panels also confirm the **Tier-B** agreement: the autonomous-actor system oracle reproduces the
quorum decision bit-for-bit (the W1 retirement, §5.2), so the availability/split-brain behavior is a
property of a real message-passing execution, not just the analytic DES.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig, scaled_dist_config
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle

CONSISTENCY_MODELS: tuple[str, ...] = ("eventual", "quorum", "linearizable")


@dataclass(frozen=True)
class ED14Config:
    name: str = "ed14-quorum"
    n_nodes: int = 5  # majority = 3
    #: the minority-side sizes to sweep (a k-node side vs a (n-k)-node side).
    minority_sizes: tuple[int, ...] = (1, 2, 3, 4)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED14Config:
        b = ED14Config()
        return ED14Config(
            name=d.get("name", b.name),
            n_nodes=d.get("n_nodes", b.n_nodes),
            minority_sizes=tuple(d.get("minority_sizes", b.minority_sizes)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED14Config:
        return ED14Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED14Result:
    #: Panel A: per model, the per-k commit status of a write from the k-node side (1 ok / 0 rej).
    availability: list[dict[str, Any]] = field(default_factory=list)
    #: Panel B: per model, the split-brain (fork) rate when both sides write the same key.
    split_brain: list[dict[str, Any]] = field(default_factory=list)
    #: the quorum majority threshold (n // 2 + 1), the frontier the availability curve steps at.
    majority: int = 0
    #: Tier-B reproduces the quorum decision bit-for-bit over every scenario.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0


def _config(cfg: ED14Config, model: str) -> DistConfig:
    return scaled_dist_config(
        cfg.n_nodes, n_objects=1, replication_factor=cfg.n_nodes, consistency_model=model
    )


def _partition_cmd(nodes: tuple[str, ...], k: int) -> str:
    """Split the cluster into a ``k``-node side and the rest: ``partition n0 .. | nk ..``."""
    left = " ".join(nodes[:k])
    right = " ".join(nodes[k:])
    return f"partition {left} | {right}"


def _forked(state: DistributedState) -> bool:
    """``True`` iff two replicas of an object hold the **same version** with **different values** —
    the split-brain signature (a divergent committed write, not mere staleness across versions)."""
    by_obj: dict[str, dict[int, str]] = {}
    for (obj, _node), r in state.replicas.items():
        seen = by_obj.setdefault(obj, {})
        if r.version in seen and seen[r.version] != r.value:
            return True
        seen[r.version] = r.value
    return False


def run_ed14(cfg: ED14Config | None = None) -> ED14Result:
    cfg = cfg or ED14Config()
    result = ED14Result(majority=cfg.n_nodes // 2 + 1)
    nodes = scaled_dist_config(cfg.n_nodes, n_objects=1).nodes
    tier_b_agree = True
    tier_b_steps = 0

    for model in CONSISTENCY_MODELS:
        config = _config(cfg, model)
        ref = ReferenceDistOracle(config)
        sysb = SystemDistOracle(config)

        # --- Panel A: a write from the k-node side, swept over k -------------------------------
        commits: dict[int, int] = {}
        for k in cfg.minority_sizes:
            script = [_partition_cmd(nodes, k), f"put {nodes[0]} o0 a"]  # nodes[0] is on the k-side
            sa = sb = DistributedState.initial(config)
            status = "ok"
            for cmd in script:
                action = parse_dist_action(cmd)
                ra, rb = ref.step(sa, action), sysb.step(sb, action)
                if cluster_view(ra.state) != cluster_view(rb.state):
                    tier_b_agree = False
                tier_b_steps += 1
                status = ra.status
                sa, sb = ra.state, rb.state
            commits[k] = 1 if status == "ok" else 0
        result.availability.append({"model": model, "commits": commits})

        # --- Panel B: both sides write the same key; does the object fork? ---------------------
        forks = 0
        for k in cfg.minority_sizes:
            # write from a k-side node AND a (n-k)-side node, then check for a divergent version
            script = [
                _partition_cmd(nodes, k),
                f"put {nodes[0]} o0 a",         # k-side
                f"put {nodes[k]} o0 b",         # other-side
            ]
            sa = sb = DistributedState.initial(config)
            for cmd in script:
                action = parse_dist_action(cmd)
                ra, rb = ref.step(sa, action), sysb.step(sb, action)
                if cluster_view(ra.state) != cluster_view(rb.state):
                    tier_b_agree = False
                tier_b_steps += 1
                sa, sb = ra.state, rb.state
            forks += _forked(sa)
        result.split_brain.append({
            "model": model,
            "forks": forks,
            "scenarios": len(cfg.minority_sizes),
            "fork_rate": forks / len(cfg.minority_sizes) if cfg.minority_sizes else 0.0,
        })

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,model,k,value,detail"


def write_csv(result: ED14Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.availability:
        for k, c in sorted(r["commits"].items()):
            lines.append(f"availability,{r['model']},{k},{c},commit_from_{k}_node_side")
    for r in result.split_brain:
        lines.append(f"split_brain,{r['model']},,{r['fork_rate']:.4f},"
                     f"forks={r['forks']}/{r['scenarios']}")
    lines.append(f"tier_b,all,,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps};majority={result.majority}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED14 (quorum consensus: availability frontier + split-brain prevention)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed14.csv")
    parser.add_argument("--plot", type=str, default="figures/ed14.png")
    args = parser.parse_args()
    cfg = ED14Config.from_json_file(args.config) if args.config else ED14Config()
    result = run_ed14(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print(f"  Panel A — write commits from a k-node side (majority = {result.majority}):")
    for r in result.availability:
        row = "  ".join(f"k={k}:{'ok' if c else 'rej'}" for k, c in sorted(r["commits"].items()))
        print(f"    [{r['model']:12s}] {row}")
    print("  Panel B — split-brain (fork) rate when both sides write:")
    for r in result.split_brain:
        verdict = "FORKS" if r["fork_rate"] > 0 else "no fork"
        print(f"    [{r['model']:12s}] {r['fork_rate']:.2f} "
              f"({r['forks']}/{r['scenarios']}) → {verdict}")
    print(f"  Tier-B reproduces the quorum decision bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed14 import plot_ed14

        plot_ed14(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
