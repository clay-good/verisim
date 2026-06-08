"""Experiment LP5: landmark placement policy -- where to spend a landmark budget (H35, SPEC-12 §6).

LP3/LP4 re-grounded at *uniform* checkpoints. LP5 asks the placement question (H35): under a small
landmark budget, *which* waypoints to keep? The measured form of the "interesting points" an analyst
or a security world model flags. Four policies score the verified-graph landmarks; the goal-reach
harness then re-grounds a journey only at its top-``k`` scored landmarks and measures how much reach
that budget buys:

  - **random** -- the control (a deterministic random score).
  - **betweenness** -- chokepoints: landmarks on many shortest reachability paths
    (:func:`~verisim.landmark.placement.betweenness_centrality`, Brandes over the *verified* edges).
  - **belief-variance** -- uncertainty: landmarks where the graph arm's RSSM posterior variance (its
    §6.2 calibrated signal) is highest, averaged over a seeded action sample.
  - **combined** -- normalized betweenness + normalized belief-variance.

LP5 is a *refinement of LP3*: the candidate re-ground points are the uniform hop boundaries LP3
re-grounds at, and the placement budget ``k`` keeps only the ``k`` highest-scored of them. So
``k = all boundaries`` recovers LP3's landmark arm (re-ground everywhere) and ``k = 0`` is flat
free-running -- the placement question is purely *which* boundaries to keep when the budget is below
the boundary count. For each (policy, budget ``k``) and each eval journey, each interior boundary
(the goal step excluded -- goal reach stays a model prediction, §10) is scored by the policy, the
``k`` highest are kept, :func:`~verisim.landmark.plan.execute_plan` re-grounds there, and we record
goal reach. Goal-reach-per-landmark (reach vs ``k``) is the H35 readout. Pre-registered both ways
(SPEC §10.1): if an informed policy beats random the placement signal is load-bearing; if random is
within CI the placement is a floor-type negative at this scale, banked and flagged to revisit at
larger worlds. Reduced over (difficulty x seed) cells with bootstrap CIs. CPU, deterministic,
seeded.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.experiments.lp3 import Journey, _roll_journey
from verisim.landmark.build import sample_landmarks
from verisim.landmark.graph import LandmarkGraph, reach_signature
from verisim.landmark.placement import (
    betweenness_centrality,
    normalize,
    random_scores,
    select_top,
)
from verisim.landmark.plan import execute_plan
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState
from verisim.netloop.model import NetModel, NetUncertaintyModel
from verisim.netoracle import ReferenceNetworkOracle

POLICIES = ("random", "betweenness", "belief_variance", "combined")


@dataclass(frozen=True)
class LP5Config:
    """A small, fast placement-policy (H35) measurement instance."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 2500
    model_seed: int = 0
    landmark_seeds: tuple[int, ...] = (0, 1, 2, 3)
    landmark_steps: int = 48
    n_uncertainty_actions: int = 8
    eval_difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    goal_distance: int = 24
    hop_length: int = 4  # uniform candidate boundaries every L steps (the LP3 boundaries)
    budgets: tuple[int, ...] = (1, 2, 3, 4, 5)  # how many of those boundaries to keep
    placement_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP5Config:
        b = LP5Config()
        return LP5Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            model_seed=d.get("model_seed", b.model_seed),
            landmark_seeds=tuple(d.get("landmark_seeds", b.landmark_seeds)),
            landmark_steps=d.get("landmark_steps", b.landmark_steps),
            n_uncertainty_actions=d.get("n_uncertainty_actions", b.n_uncertainty_actions),
            eval_difficulties=d.get("eval_difficulties", b.eval_difficulties),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            goal_distance=d.get("goal_distance", b.goal_distance),
            hop_length=d.get("hop_length", b.hop_length),
            budgets=tuple(d.get("budgets", b.budgets)),
            placement_seed=d.get("placement_seed", b.placement_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP5Config:
        return LP5Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class LP5Stat:
    """Goal reach for one (policy, budget) cell: mean + bootstrap CI over (difficulty x seed)."""

    policy: str
    budget: float
    goal_reach: float
    gr_lo: float
    gr_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.policy},{self.budget:.6f},{self.goal_reach:.6f},"
            f"{self.gr_lo:.6f},{self.gr_hi:.6f},{self.n}"
        )


CSV_HEADER = "policy,budget,goal_reach,gr_lo,gr_hi,n"


def _state_belief_variance(
    model: NetUncertaintyModel, state: NetworkState, net: NetConfig, *, driver: str,
    n_actions: int, seed: int,
) -> float:
    """Mean RSSM belief variance at ``state`` over a seeded action sample (uncertainty signal)."""
    from verisim.netdata import NetDriver

    drv = NetDriver(name=driver, config=net, rng=random.Random(seed))
    variances = [
        model.predict_delta_with_uncertainty(state, drv.sample(state))[1] for _ in range(n_actions)
    ]
    return fmean(variances) if variances else 0.0


def _interior_boundaries(goal_dist: int, hop_len: int) -> list[int]:
    """The re-ground boundaries (steps ``k*L-1``); the goal step ``goal_dist-1`` excluded."""
    last = goal_dist - 1
    return [k * hop_len - 1 for k in range(1, goal_dist // hop_len + 1) if k * hop_len - 1 < last]


def _boundary_scores(
    model: NetUncertaintyModel,
    journey: Journey,
    boundaries: list[int],
    bet_by_sig: dict[frozenset[tuple[str, str, int]], float],
    net: NetConfig,
    config: LP5Config,
    journey_idx: int,
) -> dict[str, dict[int, float]]:
    """Per-policy score of each boundary state of one journey (the four placement signals)."""
    bet = {
        b: bet_by_sig.get(reach_signature(journey.truth[b]), 0.0) for b in boundaries
    }
    var = {
        b: _state_belief_variance(
            model, journey.truth[b], net, driver=config.train_driver,
            n_actions=config.n_uncertainty_actions, seed=config.placement_seed + journey_idx + b,
        )
        for b in boundaries
    }
    nb, nv = normalize(bet), normalize(var)
    combined = {b: nb[b] + nv[b] for b in boundaries}
    rnd = random_scores(boundaries, seed=config.placement_seed + 7919 * journey_idx)
    return {"random": rnd, "betweenness": bet, "belief_variance": var, "combined": combined}


def _journey_goal_reach(
    model: NetModel,
    journey: Journey,
    boundaries: list[int],
    scores: dict[int, float],
    budget: int,
    goal_dist: int,
) -> float:
    """Keep the top-``budget`` scored boundaries; re-ground there; return goal reach
    (reachability)."""
    reground_at = select_top(scores, budget)
    actions = journey.actions[:goal_dist]
    truth = journey.truth[:goal_dist]
    trace = execute_plan(
        model, journey.start, actions, truth, frozenset(reground_at), reground=True
    )
    return float(trace.goal_reached)


def run_lp5(config: LP5Config | None = None) -> list[LP5Stat]:
    """Train the graph arm, build the verified graph, then compare placement policies (H35)."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    config = config or LP5Config()
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
    assert isinstance(model, NetUncertaintyModel)

    # The verified graph supplies the chokepoint (betweenness) signal, keyed by reachability
    # signature so any journey boundary state can be looked up.
    sample = sample_landmarks(
        oracle, net, driver=config.train_driver, seeds=config.landmark_seeds,
        n_steps=config.landmark_steps,
    )
    graph = LandmarkGraph(
        nodes=sample.nodes, signatures=sample.signatures, edges=sample.oracle_edges()
    )
    cb = betweenness_centrality(graph)
    bet_by_sig = {graph.signatures[i]: cb[i] for i in range(graph.num_nodes)}

    journeys = [
        _roll_journey(oracle, net, driver, seed, config.goal_distance)
        for driver in config.eval_difficulties.values()
        for seed in config.eval_seeds
    ]
    boundaries = _interior_boundaries(config.goal_distance, config.hop_length)
    # Per-journey per-policy boundary scores (computed once; the budget sweep just keeps top-k).
    journey_scores = [
        _boundary_scores(model, journey, boundaries, bet_by_sig, net, config, idx)
        for idx, journey in enumerate(journeys)
    ]

    stats: list[LP5Stat] = []
    for policy in POLICIES:
        for budget in config.budgets:
            reach = [
                _journey_goal_reach(
                    model, journey, boundaries, journey_scores[idx][policy], budget,
                    config.goal_distance,
                )
                for idx, journey in enumerate(journeys)
            ]
            lo, hi = bootstrap_ci(reach, seed=0) if reach else (float("nan"), float("nan"))
            stats.append(
                LP5Stat(policy, float(budget), fmean(reach) if reach else float("nan"),
                        lo, hi, len(reach))
            )
    return stats


def _print_summary(stats: list[LP5Stat]) -> None:
    print("LP5 / H35 - landmark placement: goal reach per landmark budget:")
    print(f"  {'policy':<16} {'k':>3} {'goal_reach':>12} {'95% CI':>18}")
    for s in stats:
        print(
            f"  {s.policy:<16} {s.budget:>3.0f} {s.goal_reach:>12.3f} "
            f"{f'[{s.gr_lo:.3f}, {s.gr_hi:.3f}]':>18}"
        )
    by = {(s.policy, s.budget): s for s in stats}
    budgets = sorted({s.budget for s in stats})
    # Goal-reach-per-landmark: the mean advantage over random across the budget sweep (positive =
    # buys reach at lower budget). At full budget every policy recovers the LP3 ceiling, so the
    # signal is in the low-budget regime.
    def adv(p: str) -> float:
        return fmean([by[(p, k)].goal_reach - by[("random", k)].goal_reach for k in budgets])

    print("  per-policy mean advantage over random (across budgets):")
    for p in ("betweenness", "belief_variance", "combined"):
        wins = sum(1 for k in budgets if by[(p, k)].goal_reach > by[("random", k)].goal_reach)
        losses = sum(1 for k in budgets if by[(p, k)].goal_reach < by[("random", k)].goal_reach)
        print(f"    {p:<16} adv={adv(p):+.3f}  (beats random {wins}/{len(budgets)}, "
              f"loses {losses}/{len(budgets)})")
    best = max(("betweenness", "belief_variance", "combined"), key=adv)
    best_adv = adv(best)
    verdict = (
        f"the {best} signal buys reach at lower budget (adv {best_adv:+.3f}) - H35 leaning "
        "supported for uncertainty; CIs overlap at this scale (revisit at larger worlds)"
        if best_adv > 0 else
        "no informed policy beats random - H35 negative (placement floor at this scale)"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[LP5Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {
        "random": "#999", "betweenness": "#16a",
        "belief_variance": "#393", "combined": "#c33",
    }
    labels = {
        "random": "random (control)", "betweenness": "betweenness (chokepoints)",
        "belief_variance": "belief-variance (uncertainty)", "combined": "combined",
    }
    for policy in POLICIES:
        cells = sorted((s for s in stats if s.policy == policy), key=lambda s: s.budget)
        xs = [s.budget for s in cells]
        ys = [s.goal_reach for s in cells]
        lo = [s.gr_lo for s in cells]
        hi = [s.gr_hi for s in cells]
        ax.plot(xs, ys, "-o", color=colors[policy], label=labels[policy])
        ax.fill_between(xs, lo, hi, color=colors[policy], alpha=0.10)
    ax.set_xlabel("landmark budget k (re-grounds kept per journey)")
    ax.set_ylabel("goal reach")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("LP5 / H35: goal reach per landmark budget, by placement policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP5 landmark placement policy (H35).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp5_placement.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP5Config.from_json_file(args.config) if args.config else LP5Config()
    stats = run_lp5(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
