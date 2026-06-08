"""Experiment LP6: replanning policy -- when to re-ground at equal budget (H36, SPEC-12 §6).

The RQ2 axis (does *when* you consult beat *how often*?) lifted to the planning altitude. LP5 chose
*which static landmarks* to keep; LP6 chooses *which moments to re-plan* via runtime model signals,
at **equal oracle budget** ``B`` -- the EH2 question (calibrated triggers beat fixed-interval where
flat entropy could not) carried to the planner. Three replanning triggers, each spending exactly the
same ``B`` re-grounds over a goal-distance-``G`` journey:

  - **fixed-interval** -- re-ground at ``B`` evenly spaced steps (LP3's policy; the dumb control).
  - **reachability-triggered** -- re-ground at the ``B`` steps where the model's *predicted
    reachability* changes the most (the "a subgoal was reached or has become unreachable" signal,
    SPEC-12 §3, computed online from the free-running prediction -- no oracle).
  - **belief-variance-triggered** -- re-ground at the ``B`` steps of highest RSSM posterior variance
    (the §6.2 calibrated uncertainty, where the model is least sure -- so the re-ground buys most).

All three read their trigger from a single undisturbed free-run of the model (so the budget is equal
by construction and the comparison is clean), keep the top-``B`` steps, and
:func:`~verisim.landmark.plan.execute_plan` re-grounds there. Goal reach vs budget ``B`` is the H36
readout. Pre-registered both ways (SPEC §10.1): if a triggered policy beats fixed-interval at equal
``B`` the trigger signal is calibrated at plan altitude (the EH2 lift); if fixed-interval ties or
wins the trigger is not calibrated here (the ED2-smart negative lifted) -- both banked. Reduced over
(difficulty x seed) cells with bootstrap CIs. CPU, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.experiments.lp3 import Journey, _roll_journey
from verisim.landmark.graph import ReachSig, reach_signature
from verisim.landmark.placement import select_top
from verisim.landmark.plan import execute_plan
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import NetConfig, scaled_net_config
from verisim.netdelta.apply import apply
from verisim.netloop.model import NetUncertaintyModel
from verisim.netoracle import ReferenceNetworkOracle

POLICIES = ("fixed", "reachability", "belief_variance")


@dataclass(frozen=True)
class LP6Config:
    """A small, fast replanning-policy (H36) measurement instance."""

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
    goal_distance: int = 24
    budgets: tuple[int, ...] = (1, 2, 3, 4, 6)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP6Config:
        b = LP6Config()
        return LP6Config(
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
            goal_distance=d.get("goal_distance", b.goal_distance),
            budgets=tuple(d.get("budgets", b.budgets)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP6Config:
        return LP6Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class LP6Stat:
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


def _free_run_signals(
    model: NetUncertaintyModel, journey: Journey, goal_dist: int
) -> tuple[list[float], list[float]]:
    """Free-run the model once; return per-step (reachability-change magnitude, belief variance).

    The reachability-change at step ``t`` is the size of the symmetric difference between the
    model's *predicted* reachability signature at ``t`` and at ``t-1`` -- the online "a subgoal was
    reached or became unreachable" signal (no oracle). The belief variance is the RSSM posterior
    variance (§6.2). Both are read from the *undisturbed* free-run, so a budget-``B`` top-``B``
    selection spends exactly ``B`` consults for every policy (the equal-budget guarantee).
    """
    state = journey.start
    prev_sig: ReachSig = reach_signature(state)
    reach_change: list[float] = []
    variances: list[float] = []
    for t in range(goal_dist):
        delta, var = model.predict_delta_with_uncertainty(state, journey.actions[t])
        state = apply(state, delta)
        sig = reach_signature(state)
        reach_change.append(float(len(sig ^ prev_sig)))
        variances.append(var)
        prev_sig = sig
    return reach_change, variances


def _fixed_steps(goal_dist: int, budget: int) -> set[int]:
    """``budget`` evenly spaced interior steps over a goal-distance-``goal_dist`` journey."""
    interior = goal_dist - 1  # steps 0 .. goal_dist-2 are re-groundable; goal_dist-1 is the goal
    if budget <= 0 or interior <= 0:
        return set()
    return {
        min(interior - 1, max(0, round((i + 1) * goal_dist / (budget + 1)) - 1))
        for i in range(budget)
    }


def _select(
    policy: str, signals: tuple[list[float], list[float]], goal_dist: int, budget: int
) -> set[int]:
    """The ``budget`` re-ground steps a policy keeps (interior steps; equal budget across all)."""
    reach_change, variances = signals
    interior = list(range(goal_dist - 1))  # exclude the goal step
    if policy == "fixed":
        return _fixed_steps(goal_dist, budget)
    if policy == "reachability":
        return select_top({t: reach_change[t] for t in interior}, budget)
    return select_top({t: variances[t] for t in interior}, budget)


def run_lp6(config: LP6Config | None = None) -> list[LP6Stat]:
    """Train the graph arm, then compare replanning triggers at equal oracle budget (H36)."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    config = config or LP6Config()
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

    journeys = [
        _roll_journey(oracle, net, driver, seed, config.goal_distance)
        for driver in config.eval_difficulties.values()
        for seed in config.eval_seeds
    ]
    signals = [_free_run_signals(model, journey, config.goal_distance) for journey in journeys]

    stats: list[LP6Stat] = []
    for policy in POLICIES:
        for budget in config.budgets:
            reach: list[float] = []
            for journey, sig in zip(journeys, signals, strict=True):
                steps = _select(policy, sig, config.goal_distance, budget)
                actions = journey.actions[: config.goal_distance]
                truth = journey.truth[: config.goal_distance]
                trace = execute_plan(
                    model, journey.start, actions, truth, frozenset(steps), reground=True
                )
                reach.append(float(trace.goal_reached))
            lo, hi = bootstrap_ci(reach, seed=0) if reach else (float("nan"), float("nan"))
            stats.append(
                LP6Stat(policy, float(budget), fmean(reach) if reach else float("nan"),
                        lo, hi, len(reach))
            )
    return stats


def _print_summary(stats: list[LP6Stat]) -> None:
    print("LP6 / H36 - replanning policy: goal reach at equal oracle budget:")
    print(f"  {'policy':<16} {'B':>3} {'goal_reach':>12} {'95% CI':>18}")
    for s in stats:
        print(
            f"  {s.policy:<16} {s.budget:>3.0f} {s.goal_reach:>12.3f} "
            f"{f'[{s.gr_lo:.3f}, {s.gr_hi:.3f}]':>18}"
        )
    by = {(s.policy, s.budget): s for s in stats}
    budgets = sorted({s.budget for s in stats})
    print("  per-trigger mean advantage over fixed-interval (across budgets):")
    best_adv = -1.0
    best = "fixed"
    for p in ("reachability", "belief_variance"):
        adv = fmean([by[(p, b)].goal_reach - by[("fixed", b)].goal_reach for b in budgets])
        wins = sum(1 for b in budgets if by[(p, b)].goal_reach > by[("fixed", b)].goal_reach)
        losses = sum(1 for b in budgets if by[(p, b)].goal_reach < by[("fixed", b)].goal_reach)
        print(f"    {p:<16} adv={adv:+.3f}  (beats fixed {wins}/{len(budgets)}, "
              f"loses {losses}/{len(budgets)})")
        if adv > best_adv:
            best_adv, best = adv, p
    verdict = (
        f"the {best}-triggered policy beats fixed-interval at equal budget (adv {best_adv:+.3f}) - "
        "H36 leaning supported; CIs overlap at this scale (revisit at larger worlds)"
        if best_adv > 0 else
        "no trigger beats fixed-interval - H36 negative (the trigger is not calibrated at plan "
        "altitude here, the ED2-smart negative lifted)"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[LP6Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"fixed": "#999", "reachability": "#16a", "belief_variance": "#393"}
    labels = {
        "fixed": "fixed-interval (control)",
        "reachability": "reachability-triggered",
        "belief_variance": "belief-variance-triggered",
    }
    for policy in POLICIES:
        cells = sorted((s for s in stats if s.policy == policy), key=lambda s: s.budget)
        xs = [s.budget for s in cells]
        ys = [s.goal_reach for s in cells]
        lo = [s.gr_lo for s in cells]
        hi = [s.gr_hi for s in cells]
        ax.plot(xs, ys, "-o", color=colors[policy], label=labels[policy])
        ax.fill_between(xs, lo, hi, color=colors[policy], alpha=0.10)
    ax.set_xlabel("oracle budget B (re-grounds per journey, equal across policies)")
    ax.set_ylabel("goal reach")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("LP6 / H36: replanning trigger vs fixed-interval at equal budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP6 replanning policy (H36).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp6_replanning.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP6Config.from_json_file(args.config) if args.config else LP6Config()
    stats = run_lp6(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
