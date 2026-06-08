"""Experiment LP3: long-range goal reach -- landmark planning vs flat free-running (H33, headline).

The piece LP1/LP2 set up. SPEC-10 found the structured arm's free-running horizon pinned near zero
(HS3); SPEC-12's bet (H33) is that re-grounding at *landmark boundaries* converts that zero step
horizon into long-range **goal-space** horizon -- structure buys goal reach where it could not buy
step reach -- because free-running compounds (HS3) while hops re-ground.

LP3 measures it directly. For each eval cell a seeded driver rolls a ground-truth trajectory of
length ``G`` (the *goal-space distance*); the verified landmark graph (LP2) supplies the subgoal
checkpoints every ``L`` steps, and :func:`~verisim.landmark.plan.shortest_landmark_path` (graph
search, never the model, never an LLM walking the graph -- §2.2) yields the subgoal sequence to run.
Two arms run on the *same* model and the *same* trajectory, differing only in re-grounding:

  - **flat free-running** (``ρ = 0``) -- the model rolls the whole ``G``-step path on its own (the
    HS3 reading). Goal reach = does its predicted reachability at the goal match the truth?
  - **landmark planning** (``ρ = 1/L``) -- the model free-runs each ``L``-step hop, the oracle
    re-grounds the coupled state at every *intermediate* landmark boundary (the imagine/verify loop,
    SPEC-12 §3), and the final hop to the goal is again a model prediction (never a re-ground -- the
    goal-reach is non-tautological, §10). One consult per hop, so the budget is ``ρ = 1/L < 1`` --
    the favorable regime.

Two sweeps, two panels: goal reach vs **goal-space distance** ``G`` at fixed ``L`` (does landmark
planning sustain where flat collapses?), and goal reach vs **oracle budget** ``ρ = 1/L`` at fixed
``G`` (how much re-grounding buys the reach). Reduced over (difficulty x seed) cells with bootstrap
CIs (the EN10 form). The structured arm, the two oracles, and the shipped loop are reused unchanged;
LP3 adds only the goal battery and the plan executor (:mod:`verisim.landmark.plan`). CPU,
deterministic, seeded (SPEC-2 §12).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.landmark.graph import LandmarkGraph, reach_signature
from verisim.landmark.plan import execute_plan, shortest_landmark_path
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.action import NetAction
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState
from verisim.netloop.model import NetModel
from verisim.netoracle import ReferenceNetworkOracle


@dataclass(frozen=True)
class LP3Config:
    """A small, fast goal-reach (H33) measurement instance."""

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
    hop_length: int = 4  # L for the distance sweep
    goal_distances: tuple[int, ...] = (4, 8, 12, 16, 20)  # G values (multiples of hop_length)
    budget_hop_lengths: tuple[int, ...] = (2, 4, 8)  # L values for the budget sweep (ρ = 1/L)
    budget_goal_distance: int = 16  # fixed G for the budget sweep

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP3Config:
        b = LP3Config()
        return LP3Config(
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
            budget_hop_lengths=tuple(d.get("budget_hop_lengths", b.budget_hop_lengths)),
            budget_goal_distance=d.get("budget_goal_distance", b.budget_goal_distance),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP3Config:
        return LP3Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class LP3Stat:
    """Goal reach (and reachability horizon) for one (sweep, arm, x) cell: mean + bootstrap CI."""

    sweep: str  # "distance" | "budget"
    arm: str  # "flat" | "landmark"
    x_value: float  # goal-space distance G (distance sweep) or budget ρ = 1/L (budget sweep)
    goal_reach: float
    gr_lo: float
    gr_hi: float
    reach_horizon: float
    rho: float  # consults / steps actually spent (0 for flat)
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.sweep},{self.arm},{self.x_value:.6f},{self.goal_reach:.6f},"
            f"{self.gr_lo:.6f},{self.gr_hi:.6f},{self.reach_horizon:.6f},{self.rho:.6f},{self.n}"
        )


CSV_HEADER = "sweep,arm,x_value,goal_reach,gr_lo,gr_hi,reach_horizon,rho,n"


@dataclass(frozen=True)
class Journey:
    """One ground-truth rollout: the actions + the true state after each (the re-ground source)."""

    actions: tuple[NetAction, ...]
    truth: tuple[NetworkState, ...]  # truth[t] = true state after actions[t]
    start: NetworkState


def _roll_journey(
    oracle: ReferenceNetworkOracle, net: NetConfig, driver: str, seed: int, n_steps: int
) -> Journey:
    """Roll a seeded driver ``n_steps`` from the initial state; record actions + true states."""
    from verisim.netdata import NetDriver

    drv = NetDriver(name=driver, config=net, rng=random.Random(seed))
    start = NetworkState.initial(net.hosts)
    state = start
    actions: list[NetAction] = []
    truth: list[NetworkState] = []
    for _ in range(n_steps):
        action = drv.sample(state)
        state = oracle.step(state, action).state
        actions.append(action)
        truth.append(state)
    return Journey(tuple(actions), tuple(truth), start)


def _chain_graph(journey: Journey, goal_dist: int, hop_len: int) -> LandmarkGraph:
    """The verified landmark chain over ``journey``'s every-``hop_len`` checkpoints up to the goal.

    Node 0 is the start; node ``k`` is the checkpoint after ``k`` hops (step ``k*hop_len - 1``); the
    goal is the last node. Edges are the consecutive verified hops -- a real (if linear) graph the
    high-level planner searches; branching/stitching is exercised in the unit tests.
    """
    nodes = [journey.start]
    for k in range(1, goal_dist // hop_len + 1):
        nodes.append(journey.truth[k * hop_len - 1])
    sigs = tuple(reach_signature(s) for s in nodes)
    edges = frozenset((k, k + 1) for k in range(len(nodes) - 1))
    return LandmarkGraph(nodes=tuple(nodes), signatures=sigs, edges=edges)


def _reground_steps(path: list[int], hop_len: int) -> frozenset[int]:
    """Step indices to re-ground at: the *intermediate* subgoal boundaries (the goal excluded)."""
    return frozenset(k * hop_len - 1 for k in path[1:-1])


def _cell_goal_reach(
    model: NetModel, journey: Journey, goal_dist: int, hop_len: int
) -> tuple[dict[str, float], dict[str, float], float]:
    """Run both arms on one journey at (goal_dist, hop_len); return goal-reach, horizon, ρ."""
    graph = _chain_graph(journey, goal_dist, hop_len)
    path = shortest_landmark_path(graph, 0, graph.num_nodes - 1)
    assert path is not None  # the chain is connected by construction
    reground_at = _reground_steps(path, hop_len)
    actions = journey.actions[:goal_dist]
    truth = journey.truth[:goal_dist]

    flat = execute_plan(model, journey.start, actions, truth, frozenset(), reground=False)
    landmark = execute_plan(model, journey.start, actions, truth, reground_at, reground=True)
    reach = {"flat": float(flat.goal_reached), "landmark": float(landmark.goal_reached)}
    horizon = {"flat": float(flat.reach_horizon), "landmark": float(landmark.reach_horizon)}
    rho = landmark.n_consults / goal_dist if goal_dist else 0.0
    return reach, horizon, rho


def run_lp3(config: LP3Config | None = None) -> list[LP3Stat]:
    """Train the graph arm, then measure goal reach: landmark planning vs flat free-run (H33)."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    config = config or LP3Config()
    torch.set_num_threads(1)  # process-reproducibility (the EN1 discipline)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
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

    g_max = max(max(config.goal_distances), config.budget_goal_distance)
    journeys = [
        _roll_journey(oracle, net, driver, seed, g_max)
        for driver in config.eval_difficulties.values()
        for seed in config.eval_seeds
    ]

    # -- distance sweep: fixed L, vary G ---------------------------------------------------------
    stats: list[LP3Stat] = []
    for goal_dist in config.goal_distances:
        per_arm_reach: dict[str, list[float]] = {"flat": [], "landmark": []}
        per_arm_horizon: dict[str, list[float]] = {"flat": [], "landmark": []}
        rhos: list[float] = []
        for journey in journeys:
            reach, horizon, rho = _cell_goal_reach(model, journey, goal_dist, config.hop_length)
            for arm in ("flat", "landmark"):
                per_arm_reach[arm].append(reach[arm])
                per_arm_horizon[arm].append(horizon[arm])
            rhos.append(rho)
        for arm in ("flat", "landmark"):
            stats.append(
                _reduce("distance", arm, float(goal_dist), per_arm_reach[arm],
                        per_arm_horizon[arm], 0.0 if arm == "flat" else fmean(rhos))
            )

    # -- budget sweep: fixed G, vary L (so ρ = 1/L) ----------------------------------------------
    g = config.budget_goal_distance
    flat_reach: list[float] = []
    flat_horizon: list[float] = []
    for journey in journeys:
        reach, horizon, _ = _cell_goal_reach(model, journey, g, config.budget_hop_lengths[0])
        flat_reach.append(reach["flat"])
        flat_horizon.append(horizon["flat"])
    stats.append(_reduce("budget", "flat", 0.0, flat_reach, flat_horizon, 0.0))
    for hop_len in config.budget_hop_lengths:
        lm_reach: list[float] = []
        lm_horizon: list[float] = []
        rhos = []
        for journey in journeys:
            reach, horizon, rho = _cell_goal_reach(model, journey, g, hop_len)
            lm_reach.append(reach["landmark"])
            lm_horizon.append(horizon["landmark"])
            rhos.append(rho)
        stats.append(
            _reduce("budget", "landmark", 1.0 / hop_len, lm_reach, lm_horizon, fmean(rhos))
        )

    return stats


def _reduce(
    sweep: str, arm: str, x_value: float, reach: list[float], horizon: list[float], rho: float
) -> LP3Stat:
    """Reduce a cell's per-(difficulty x seed) goal-reach + horizon to a mean + bootstrap CI."""
    lo, hi = bootstrap_ci(reach, seed=0) if reach else (float("nan"), float("nan"))
    return LP3Stat(
        sweep=sweep, arm=arm, x_value=x_value,
        goal_reach=fmean(reach) if reach else float("nan"),
        gr_lo=lo, gr_hi=hi,
        reach_horizon=fmean(horizon) if horizon else float("nan"),
        rho=rho, n=len(reach),
    )


def _print_summary(stats: list[LP3Stat]) -> None:
    print("LP3 / H33 - goal reach: landmark planning vs flat free-running:")
    print(f"  {'sweep':<9} {'arm':<9} {'x':>7} {'goal_reach':>12} {'95% CI':>18} {'ρ':>6}")
    for s in stats:
        print(
            f"  {s.sweep:<9} {s.arm:<9} {s.x_value:>7.3f} {s.goal_reach:>12.3f} "
            f"{f'[{s.gr_lo:.3f}, {s.gr_hi:.3f}]':>18} {s.rho:>6.3f}"
        )
    dist = {(s.arm, s.x_value): s for s in stats if s.sweep == "distance"}
    far = max(s.x_value for s in stats if s.sweep == "distance")
    flat_far = dist[("flat", far)].goal_reach
    lm_far = dist[("landmark", far)].goal_reach
    verdict = (
        "landmark planning reaches the far goal where flat free-running cannot - H33 supported"
        if lm_far > flat_far else
        "landmark planning does not beat flat free-running at the far goal - H33 refuted"
    )
    print(f"  verdict (G={far:.0f}): flat {flat_far:.3f} vs landmark {lm_far:.3f} -> {verdict}")


def _plot(stats: list[LP3Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A: goal reach vs goal-space distance G (fixed L).
    for arm, color, label in (
        ("flat", "#c33", "flat free-running (ρ=0)"),
        ("landmark", "#16a", "landmark planning (ρ≈1/L)"),
    ):
        cells = sorted((s for s in stats if s.sweep == "distance" and s.arm == arm),
                       key=lambda s: s.x_value)
        xs = [s.x_value for s in cells]
        ys = [s.goal_reach for s in cells]
        lo = [s.gr_lo for s in cells]
        hi = [s.gr_hi for s in cells]
        ax1.plot(xs, ys, "-o", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.set_xlabel("goal-space distance G (steps)")
    ax1.set_ylabel("goal reach (reachability at goal correct)")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title("landmark planning sustains goal reach; free-running collapses")
    ax1.legend(fontsize=8)

    # Panel B: goal reach vs oracle budget ρ = 1/L (fixed G).
    budget = sorted((s for s in stats if s.sweep == "budget" and s.arm == "landmark"),
                    key=lambda s: s.x_value)
    xs = [s.x_value for s in budget]
    ys = [s.goal_reach for s in budget]
    lo = [s.gr_lo for s in budget]
    hi = [s.gr_hi for s in budget]
    ax2.plot(xs, ys, "-o", color="#16a", label="landmark (ρ = 1/L)")
    ax2.fill_between(xs, lo, hi, color="#16a", alpha=0.15)
    flat = next(s for s in stats if s.sweep == "budget" and s.arm == "flat")
    ax2.scatter([0.0], [flat.goal_reach], color="#c33", zorder=5, label="flat free-running (ρ=0)")
    ax2.set_xlabel("oracle budget ρ = 1/L (consults per step)")
    ax2.set_ylabel("goal reach (fixed far G)")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("re-grounding budget buys goal reach")
    ax2.legend(fontsize=8)

    fig.suptitle("LP3 / H33: structure buys goal-space horizon (re-ground at landmark boundaries)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP3 goal reach: landmark planning vs flat (H33).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp3_goal_reach.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP3Config.from_json_file(args.config) if args.config else LP3Config()
    stats = run_lp3(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
