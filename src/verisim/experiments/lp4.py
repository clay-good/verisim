"""Experiment LP4: edge-metric ablation -- reachability vs exact-state edges (H34, SPEC-12 §6).

LP3 showed the *headline* (re-grounding at landmark boundaries buys goal-space horizon). LP4
isolates *why the graph is wired on reachability* rather than on the exact state. It is the
EN10-vs-HS3 distinction (the model predicts a hop's reachability *effect* better than its delta)
lifted to the planning altitude, and pre-registered as H34: **a graph whose edges are the
reachability projection plans successfully where a graph whose edges are the exact-delta prediction
fails.**

The two arms run the *same* LP3 landmark planner over the *same* verified chain and the *same*
re-grounding boundaries -- differing only in the **projection the edge (and the goal) is wired on**:

  - **reachability edges** -- a hop succeeds / the goal is reached when the model's predicted
    *reachability signature* matches the truth (the EN10-faithful projection, LP3's arm). Within a
    hop the model free-runs the reachability effect; at the goal it must land the reach class.
  - **exact-state edges** -- a hop succeeds / the goal is reached only when the model's predicted
    *exact state* matches the truth (the HS3-pinned projection). The same free-run, graded on the
    bit-for-bit delta the structured arm cannot sustain (HS3, ``H_free ≈ 0``).

Both arms re-ground to truth at the *intermediate* landmark boundaries, so they are identical up to
the final, model-predicted goal hop; the gap there is the H34 effect. Two readouts show the
mechanism: **goal reach** vs goal-space distance ``G`` (does the reachability arm sustain where the
exact arm collapses?), and the **horizon gap** -- the model's reachability horizon (steps the
reachability projection stays correct) against its exact-state horizon (steps the bit-for-bit state
stays correct), the within-hop form of the same fact. Reduced over (difficulty x seed) cells with
bootstrap CIs (the EN10 reduction). The structured arm, the two oracles, and the shipped loop are
reused unchanged; LP4 adds only the second readout projection. CPU, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.experiments.lp3 import Journey, _chain_graph, _reground_steps, _roll_journey
from verisim.landmark.plan import execute_plan, shortest_landmark_path
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import NetConfig, scaled_net_config
from verisim.netloop.model import NetModel
from verisim.netoracle import ReferenceNetworkOracle


@dataclass(frozen=True)
class LP4Config:
    """A small, fast edge-metric-ablation (H34) measurement instance."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 2500
    model_seed: int = 0
    eval_difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    hop_length: int = 4
    goal_distances: tuple[int, ...] = (4, 8, 12, 16, 20)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP4Config:
        b = LP4Config()
        return LP4Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            model_seed=d.get("model_seed", b.model_seed),
            eval_difficulties=d.get("eval_difficulties", b.eval_difficulties),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            hop_length=d.get("hop_length", b.hop_length),
            goal_distances=tuple(d.get("goal_distances", b.goal_distances)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP4Config:
        return LP4Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class LP4Stat:
    """Goal reach + horizon for one (edge-metric, G) cell: mean + bootstrap CI over cells."""

    edge_metric: str  # "reachability" | "exact"
    goal_distance: float
    goal_reach: float
    gr_lo: float
    gr_hi: float
    horizon: float  # reachability horizon (reach arm) or exact-state horizon (exact arm)
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.edge_metric},{self.goal_distance:.6f},{self.goal_reach:.6f},"
            f"{self.gr_lo:.6f},{self.gr_hi:.6f},{self.horizon:.6f},{self.n}"
        )


CSV_HEADER = "edge_metric,goal_distance,goal_reach,gr_lo,gr_hi,horizon,n"


def _cell_edge_metrics(
    model: NetModel, journey: Journey, goal_dist: int, hop_len: int
) -> dict[str, tuple[float, float]]:
    """Run the landmark planner on one journey; read goal-reach + horizon under both projections.

    A single re-grounding rollout (identical for both arms up to the final goal hop) is graded two
    ways: ``reachability`` reads the reachability-signature projection (EN10), ``exact`` reads the
    bit-for-bit state projection (HS3). Returns ``{edge_metric: (goal_reach, horizon)}``.
    """
    graph = _chain_graph(journey, goal_dist, hop_len)
    path = shortest_landmark_path(graph, 0, graph.num_nodes - 1)
    assert path is not None  # the chain is connected by construction
    reground_at = _reground_steps(path, hop_len)
    actions = journey.actions[:goal_dist]
    truth = journey.truth[:goal_dist]
    trace = execute_plan(model, journey.start, actions, truth, reground_at, reground=True)
    return {
        "reachability": (float(trace.goal_reached), float(trace.reach_horizon)),
        "exact": (float(trace.goal_reached_exact), float(trace.full_horizon)),
    }


def run_lp4(config: LP4Config | None = None) -> list[LP4Stat]:
    """Train the graph arm, then ablate the edge projection: reachability vs exact-state (H34)."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    config = config or LP4Config()
    torch.set_num_threads(1)  # process-reproducibility (the EN1 discipline)
    oracle = ReferenceNetworkOracle()
    net: NetConfig = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)

    model = build_graph_model(
        vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds,
        seed=config.model_seed,
    )
    examples = build_graph_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    train_graph_model(model, examples, steps=config.graph_iters, seed=config.model_seed)
    model.net.eval()

    g_max = max(config.goal_distances)
    journeys = [
        _roll_journey(oracle, net, driver, seed, g_max)
        for driver in config.eval_difficulties.values()
        for seed in config.eval_seeds
    ]

    stats: list[LP4Stat] = []
    for goal_dist in config.goal_distances:
        per_arm_reach: dict[str, list[float]] = {"reachability": [], "exact": []}
        per_arm_horizon: dict[str, list[float]] = {"reachability": [], "exact": []}
        for journey in journeys:
            cell = _cell_edge_metrics(model, journey, goal_dist, config.hop_length)
            for arm, (reach, horizon) in cell.items():
                per_arm_reach[arm].append(reach)
                per_arm_horizon[arm].append(horizon)
        for arm in ("reachability", "exact"):
            stats.append(
                _reduce(arm, float(goal_dist), per_arm_reach[arm], per_arm_horizon[arm])
            )
    return stats


def _reduce(
    edge_metric: str, goal_distance: float, reach: list[float], horizon: list[float]
) -> LP4Stat:
    """Reduce a cell's per-(difficulty x seed) goal-reach + horizon to a mean + bootstrap CI."""
    lo, hi = bootstrap_ci(reach, seed=0) if reach else (float("nan"), float("nan"))
    return LP4Stat(
        edge_metric=edge_metric, goal_distance=goal_distance,
        goal_reach=fmean(reach) if reach else float("nan"),
        gr_lo=lo, gr_hi=hi,
        horizon=fmean(horizon) if horizon else float("nan"),
        n=len(reach),
    )


def _print_summary(stats: list[LP4Stat]) -> None:
    print("LP4 / H34 - edge-metric ablation: reachability vs exact-state edges:")
    print(f"  {'edge_metric':<14} {'G':>5} {'goal_reach':>12} {'95% CI':>18} {'horizon':>8}")
    for s in stats:
        print(
            f"  {s.edge_metric:<14} {s.goal_distance:>5.0f} {s.goal_reach:>12.3f} "
            f"{f'[{s.gr_lo:.3f}, {s.gr_hi:.3f}]':>18} {s.horizon:>8.2f}"
        )
    by = {(s.edge_metric, s.goal_distance): s for s in stats}
    far = max(s.goal_distance for s in stats)
    reach_far = by[("reachability", far)].goal_reach
    exact_far = by[("exact", far)].goal_reach
    verdict = (
        "reachability edges plan where exact-state edges fail - H34 supported"
        if reach_far > exact_far else
        "exact-state edges plan as well as reachability - H34 refuted"
    )
    print(f"  verdict (G={far:.0f}): reach {reach_far:.3f} vs exact {exact_far:.3f} -> {verdict}")


def _plot(stats: list[LP4Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A: goal reach vs goal-space distance G, per edge projection.
    for arm, color, label in (
        ("reachability", "#16a", "reachability edges (EN10 projection)"),
        ("exact", "#c33", "exact-state edges (HS3 projection)"),
    ):
        cells = sorted((s for s in stats if s.edge_metric == arm), key=lambda s: s.goal_distance)
        xs = [s.goal_distance for s in cells]
        ys = [s.goal_reach for s in cells]
        lo = [s.gr_lo for s in cells]
        hi = [s.gr_hi for s in cells]
        ax1.plot(xs, ys, "-o", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.set_xlabel("goal-space distance G (steps)")
    ax1.set_ylabel("goal reach")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title("reachability edges plan; exact-state edges fail")
    ax1.legend(fontsize=8)

    # Panel B: the within-hop horizon gap that causes it (reach horizon > exact horizon).
    for arm, color, label in (
        ("reachability", "#16a", "reachability horizon"),
        ("exact", "#c33", "exact-state horizon"),
    ):
        cells = sorted((s for s in stats if s.edge_metric == arm), key=lambda s: s.goal_distance)
        xs = [s.goal_distance for s in cells]
        ys = [s.horizon for s in cells]
        ax2.plot(xs, ys, "-o", color=color, label=label)
    ax2.set_xlabel("goal-space distance G (steps)")
    ax2.set_ylabel("free-run horizon (steps correct)")
    ax2.set_title("the model sustains reachability longer than exact state")
    ax2.legend(fontsize=8)

    fig.suptitle("LP4 / H34: reachability edges beat exact-state edges (EN10 over HS3)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP4 edge-metric ablation (H34).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp4_edge_metric.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP4Config.from_json_file(args.config) if args.config else LP4Config()
    stats = run_lp4(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
