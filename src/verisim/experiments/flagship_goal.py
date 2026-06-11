"""FL3 -- the structured arm's goal-horizon on the flagship (SPEC-19 §4, H71).

FL1's headline curve uses the *flat* flagship arm -- the competent proposer whose floor SPEC-10
dissolved with scale. FL3 is about the *other* arm. SPEC-10 HS3 found the structured (graph+RSSM)
proposer's free-running step horizon pinned at `H_free=0` across capacity, data, AND world size -- a
genuine compounding wall, not a resourcing artifact. SPEC-12 LP3 then showed (on a small graph arm)
that landmark planning converts that zero *step* horizon into long-range *goal-space* horizon by
re-grounding at landmark boundaries (H33). FL3 puts both facts on **one structured flagship arm at
the compute-optimal frontier** and reports them together -- the H71 co-report:

  - **the wall survives** -- free-running `H_free ≈ 0` at exact tolerance on the trained flagship
    structured arm (the HS3 reading reproduced at flagship scale, *not* dissolved like the flat
    arm's);
  - **structure escapes it** -- landmark planning lifts goal-reach far above flat free-running on
    the *same* model, sustaining where flat collapses with goal-space distance.

H71 is *supported* when both land on one checkpoint; *refuted* if the LP3 goal-reach lift fails to
reproduce on the trained flagship arm (the planner's faithful hops decay when taken by a real
drifting model rather than LP3's smaller one) -- which would localize H33 to the smaller arm.

This is the structured sibling of FL0's flat flagship: a different proposer in its planning role
(SPEC-19 §2, the proposer is swappable). FL3 reuses LP3's goal-reach machinery verbatim
([`landmark.plan.execute_plan`](../landmark/plan.py), the chain graph, the journey roller) and adds
only the co-located `H_free` wall measurement + the H71 verdict. CPU-local; CI runs the smoke
instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.loop.policy import Never
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState
from verisim.netloop import PartialNetOracle, run_net_rollout
from verisim.netoracle import ReferenceNetworkOracle

from .lp3 import _cell_goal_reach, _roll_journey

if TYPE_CHECKING:
    from verisim.netloop.model import NetModel


@dataclass(frozen=True)
class FlagshipGoalConfig:
    """The FL3 instance: a frontier-scale structured arm + the goal-reach + wall battery."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_steps_per_traj: int = 60
    graph_d_model: int = 96  # frontier-ish structured capacity (LP3 used 64)
    graph_mp_rounds: int = 3
    graph_iters: int = 4000
    model_seed: int = 0
    eval_difficulties: tuple[str, ...] = ("weighted", "adversarial")
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    hop_length: int = 4
    goal_distances: tuple[int, ...] = (4, 8, 12, 16)
    epsilon: float = 0.0
    free_run_seeds: tuple[int, ...] = (200, 201, 202)
    free_run_steps: int = 24  # the H_free wall measurement horizon

    @staticmethod
    def smoke() -> FlagshipGoalConfig:
        return FlagshipGoalConfig(
            train_seeds=(0, 1), train_steps_per_traj=20, graph_d_model=32, graph_iters=300,
            eval_seeds=(100,), goal_distances=(4, 8), free_run_seeds=(200,), free_run_steps=10,
        )


@dataclass(frozen=True)
class GoalStat:
    """Goal reach for one (arm, goal_distance) cell: mean + bootstrap CI + the spent budget."""

    arm: str  # "flat" | "landmark"
    goal_distance: int
    goal_reach: float
    gr_lo: float
    gr_hi: float
    rho: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.goal_distance},{self.goal_reach:.6f},"
            f"{self.gr_lo:.6f},{self.gr_hi:.6f},{self.rho:.6f},{self.n}"
        )


CSV_HEADER = "arm,goal_distance,goal_reach,gr_lo,gr_hi,rho,n"


def _train_structured_arm(
    config: FlagshipGoalConfig, net: NetConfig, oracle: ReferenceNetworkOracle
) -> NetModel:
    """Train the graph+RSSM arm at the FL3 frontier config (the LP3 recipe, frontier scale)."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    torch.set_num_threads(1)
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
    return model


def measure_wall(
    model: NetModel, config: FlagshipGoalConfig, oracle: ReferenceNetworkOracle, net: NetConfig,
) -> float:
    """The structured arm's free-running `H_free` at ρ=0 -- the HS3 wall, measured on this model."""
    from .en1 import eval_actions

    partial = PartialNetOracle(oracle)
    horizons: list[float] = []
    for seed in config.free_run_seeds:
        actions = eval_actions(oracle, net, config.train_driver, seed, config.free_run_steps)
        record = run_net_rollout(
            model, partial, NetworkState.initial(net.hosts), actions, Never(),
            epsilon=config.epsilon, budget=0, seed=seed,
        )
        horizons.append(float(faithful_horizon(list(record.divergences), config.epsilon)))
    return fmean(horizons)


def measure_goal_reach(
    model: NetModel, config: FlagshipGoalConfig, net: NetConfig, oracle: ReferenceNetworkOracle
) -> list[GoalStat]:
    """Goal-reach flat vs landmark across goal-space distance (LP3's H33 battery on this model)."""
    g_max = max(config.goal_distances)
    journeys = [
        _roll_journey(oracle, net, driver, seed, g_max)
        for driver in config.eval_difficulties
        for seed in config.eval_seeds
    ]
    stats: list[GoalStat] = []
    for goal_dist in config.goal_distances:
        per_arm: dict[str, list[float]] = {"flat": [], "landmark": []}
        rhos: list[float] = []
        for journey in journeys:
            reach, _horizon, rho = _cell_goal_reach(model, journey, goal_dist, config.hop_length)
            per_arm["flat"].append(reach["flat"])
            per_arm["landmark"].append(reach["landmark"])
            rhos.append(rho)
        for arm in ("flat", "landmark"):
            lo, hi = bootstrap_ci(per_arm[arm], seed=0)
            stats.append(
                GoalStat(arm, goal_dist, fmean(per_arm[arm]), lo, hi,
                         0.0 if arm == "flat" else fmean(rhos), len(per_arm[arm]))
            )
    return stats


def run_flagship_goal(
    config: FlagshipGoalConfig | None = None, *, oracle: ReferenceNetworkOracle | None = None,
) -> tuple[float, list[GoalStat]]:
    """Train the structured flagship arm; co-measure the wall (`H_free`) and the goal-reach lift."""
    config = config or FlagshipGoalConfig()
    oracle = oracle or ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    model = _train_structured_arm(config, net, oracle)
    h_free = measure_wall(model, config, oracle, net)
    goal_stats = measure_goal_reach(model, config, net, oracle)
    return h_free, goal_stats


def h71_verdict(
    h_free: float, goal_stats: list[GoalStat], *, wall_tol: float = 1.0
) -> dict[str, Any]:
    """H71: the wall survives (`H_free` small) AND landmark beats flat at the far goal."""
    far = max((s.goal_distance for s in goal_stats), default=0)
    flat_far = next(
        (s.goal_reach for s in goal_stats if s.arm == "flat" and s.goal_distance == far), 0.0
    )
    lm_far = next(
        (s.goal_reach for s in goal_stats if s.arm == "landmark" and s.goal_distance == far), 0.0
    )
    return {
        "h_free": h_free,
        "wall_survives": h_free <= wall_tol,
        "far_goal_distance": far,
        "flat_goal_reach_far": flat_far,
        "landmark_goal_reach_far": lm_far,
        "goal_lift": lm_far - flat_far,
        "h71_supported": h_free <= wall_tol and lm_far > flat_far,
    }


def write_csv(stats: list[GoalStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="FL3 -- structured goal-horizon (SPEC-19, H71).")
    parser.add_argument("--out", type=str, default="figures/fl3_goal_horizon.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = FlagshipGoalConfig.smoke() if args.smoke else FlagshipGoalConfig()
    h_free, stats = run_flagship_goal(config)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    print(f"  H_free (the HS3 wall) = {h_free:.2f}")
    for s in stats:
        print(f"  {s.arm:9s} G={s.goal_distance:2d}  goal_reach={s.goal_reach:.3f} "
              f"[{s.gr_lo:.3f}, {s.gr_hi:.3f}]  ρ={s.rho:.3f}")
    verdict = h71_verdict(h_free, stats)
    print(f"H71: {'SUPPORTED' if verdict['h71_supported'] else 'not supported'}  {verdict}")


if __name__ == "__main__":  # pragma: no cover
    main()
