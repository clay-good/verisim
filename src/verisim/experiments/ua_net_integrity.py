"""UA10 -- network flow-integrity: the cross-world confirmation (SPEC-20 §7, H82).

UA8/UA9 drew the SPEC-20 boundary law and the useful knee on the **host** world (content =
file-writes). UA10 asks whether both reproduce on the **network** world, whose content dimension is
**flows** (the network flagship drifts ~0.252 on the live-flow set, faithful on structure). The
content-keyed task is flow-integrity defense: an adversarial workload opens connections; the
defender predicts which flows will be live and protects the budget flows it predicts. Two sweeps,
the two prior findings reproduced on the network:

  - **the positive (mirror of UA8/H80)** -- a faithful predictor (oracle rollout) vs a free
    predictor (raw network `M_θ` rollout), over the workload horizon: faithful > free, the gap
    widening with horizon as flow drift compounds.
  - **the useful knee (mirror of UA9/H81)** -- the ρ-grounded predictor sweeps the catch rate from
    the free floor to the faithful ceiling, recovering it at sub-linear oracle budget.

**H82 (the law is cross-world).** Both reproduce on the network world: the content-keyed positive
(faithful beats free, widening with horizon) *and* the useful knee (catch monotone in ρ, ceiling
recovered below full budget). *Refuted if* the network free predictor already catches the flows (the
model is faithful on flows too, so the content/structure split is host-specific) or the ρ-curve is
flat (grounding buys nothing on the network content task). No training; CPU-local; CI runs smoke.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.acd.net_integrity import (
    grounded_flow_defense_reward,
    make_net_workload,
    model_step,
    oracle_step,
    predictive_flow_defense_reward,
)
from verisim.metrics.aggregate import bootstrap_ci
from verisim.netoracle.base import NetOracle
from verisim.netoracle.reference import ReferenceNetworkOracle

if TYPE_CHECKING:
    from verisim.netmodel import NeuralNetworkWorldModel


@dataclass(frozen=True)
class NetIntegrityConfig:
    """The UA10 sweeps: the faithful-vs-free horizon sweep + the ρ-grounded useful-knee sweep."""

    horizons: tuple[int, ...] = (8, 14, 20, 28)
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    knee_horizon: int = 20  # the horizon the ρ-sweep is run at (deep enough that free drifts)
    budget: int = 2
    workload_seeds: tuple[int, ...] = tuple(range(800, 824))
    driver: str = "weighted"  # the in-distribution driver the flagship was trained/evaluated on
    knee_frac: float = 0.9

    @staticmethod
    def smoke() -> NetIntegrityConfig:
        return NetIntegrityConfig(
            horizons=(6, 12), rhos=(0.0, 0.5, 1.0), knee_horizon=12,
            budget=2, workload_seeds=(800, 801, 802),
        )


@dataclass(frozen=True)
class IntegrityPoint:
    """One horizon cell: faithful vs free mean catch + bootstrap CI."""

    predictor: str  # "faithful" | "free"
    horizon: int
    reward: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"horizon,{self.predictor},{self.horizon},{self.reward:.6f},"
            f"{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


@dataclass(frozen=True)
class KneePoint:
    """One ρ cell: mean catch + bootstrap CI + mean oracle-call cost."""

    rho: float
    reward: float
    ci_lo: float
    ci_hi: float
    oracle_calls: float
    n: int

    def csv_row(self) -> str:
        return (
            f"rho,{self.rho:.4f},{self.reward:.6f},{self.ci_lo:.6f},"
            f"{self.ci_hi:.6f},{self.oracle_calls:.4f},{self.n}"
        )


CSV_HEADER = "kind,col1,col2,col3,col4,col5,col6"


def run_horizon_sweep(
    model: NeuralNetworkWorldModel, config: NetIntegrityConfig, *, oracle: NetOracle,
) -> list[IntegrityPoint]:
    """The UA8 mirror: faithful vs free predictor over the workload horizon."""
    faithful = oracle_step(oracle)
    free = model_step(model)
    true_step = oracle_step(oracle)
    points: list[IntegrityPoint] = []
    for horizon in config.horizons:
        workloads = [
            make_net_workload(s, horizon, driver=config.driver, oracle=oracle)
            for s in config.workload_seeds
        ]
        for name, predictor in (("faithful", faithful), ("free", free)):
            rewards = [
                predictive_flow_defense_reward(predictor, true_step, start, actions, config.budget)
                for start, actions in workloads
            ]
            lo, hi = bootstrap_ci(rewards, seed=0)
            points.append(IntegrityPoint(name, horizon, fmean(rewards), lo, hi, len(rewards)))
    return points


def run_knee_sweep(
    model: NeuralNetworkWorldModel, config: NetIntegrityConfig, *, oracle: NetOracle,
) -> list[KneePoint]:
    """The UA9 mirror: the ρ-grounded predictor's catch rate + cost over the consultation budget."""
    workloads = [
        make_net_workload(s, config.knee_horizon, driver=config.driver, oracle=oracle)
        for s in config.workload_seeds
    ]
    points: list[KneePoint] = []
    for rho in config.rhos:
        rewards: list[float] = []
        calls: list[int] = []
        for start, actions in workloads:
            r, c = grounded_flow_defense_reward(model, oracle, start, actions, config.budget, rho)
            rewards.append(r)
            calls.append(c)
        lo, hi = bootstrap_ci(rewards, seed=0)
        points.append(KneePoint(rho, fmean(rewards), lo, hi, fmean(calls), len(rewards)))
    return points


def h82_verdict(
    horizon_points: list[IntegrityPoint], knee_points: list[KneePoint], knee_frac: float = 0.9,
) -> dict[str, Any]:
    """H82: the content-keyed positive AND the useful knee both reproduce on the network world."""
    by_h: dict[tuple[str, int], float] = {
        (p.predictor, p.horizon): p.reward for p in horizon_points
    }
    horizons = sorted({p.horizon for p in horizon_points})
    gaps = {h: by_h[("faithful", h)] - by_h[("free", h)] for h in horizons if ("free", h) in by_h}
    hmin, hmax = horizons[0], horizons[-1]
    positive = gaps.get(hmax, 0.0) > 0.02 and gaps.get(hmax, 0.0) > gaps.get(hmin, 0.0)

    by_r = {p.rho: p for p in knee_points}
    rhos = sorted(by_r)
    floor, ceil = by_r[rhos[0]].reward, by_r[rhos[-1]].reward
    ceil_calls = by_r[rhos[-1]].oracle_calls
    rewards = [by_r[r].reward for r in rhos]
    tol = max((p.ci_hi - p.ci_lo) / 2 for p in knee_points)
    monotone = all(rewards[i + 1] >= rewards[i] - tol for i in range(len(rewards) - 1))
    threshold = knee_frac * ceil
    knee = next((r for r in rhos if by_r[r].reward >= threshold), rhos[-1])
    knee_useful = (
        ceil - floor > 0.02 and monotone
        and knee < rhos[-1] and by_r[knee].oracle_calls < ceil_calls
    )

    return {
        "horizons": horizons,
        "faithful_minus_free": gaps,
        "gap_at_min_horizon": gaps.get(hmin, 0.0),
        "gap_at_max_horizon": gaps.get(hmax, 0.0),
        "content_positive": positive,
        "rhos": rhos,
        "reward_by_rho": {r: by_r[r].reward for r in rhos},
        "free_floor": floor,
        "faithful_ceiling": ceil,
        "monotone_in_rho": monotone,
        "knee_rho": knee,
        "knee_calls": by_r[knee].oracle_calls,
        "ceiling_calls": ceil_calls,
        "useful_knee": knee_useful,
        # H82: the law (the positive) AND the knee both reproduce on the network world
        "h82_supported": positive and knee_useful,
    }


def write_csv(
    horizon_points: list[IntegrityPoint], knee_points: list[KneePoint], path: str | Path,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER, *(p.csv_row() for p in horizon_points), *(p.csv_row() for p in knee_points)]
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="UA10 -- network flow-integrity: the cross-world confirmation (H82)."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/ua10_net_integrity.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = NetIntegrityConfig.smoke() if args.smoke else NetIntegrityConfig()
    oracle: NetOracle = ReferenceNetworkOracle()
    if args.smoke:
        from verisim.experiments.flagship import FlagshipConfig, train_flagship

        model, _ = train_flagship(FlagshipConfig.smoke())
    else:
        from verisim.experiments.flagship import load_checkpoint

        model = load_checkpoint(args.checkpoint).world_model

    horizon_points = run_horizon_sweep(model, config, oracle=oracle)
    knee_points = run_knee_sweep(model, config, oracle=oracle)
    write_csv(horizon_points, knee_points, args.out)

    print("the content-keyed positive (faithful vs free predictor over horizon):")
    for h in sorted({p.horizon for p in horizon_points}):
        c = {p.predictor: p.reward for p in horizon_points if p.horizon == h}
        print(f"  horizon={h:2d}:  faithful={c['faithful']:.3f}  free={c['free']:.3f}  "
              f"gap={c['faithful'] - c['free']:+.3f}")
    print("the useful knee (ρ-grounded predictor):")
    for p in knee_points:
        print(f"  ρ={p.rho:.2f}:  catch={p.reward:.3f}  oracle_calls={p.oracle_calls:.1f}")
    v = h82_verdict(horizon_points, knee_points, config.knee_frac)
    print(
        f"H82 (the law is cross-world): {'SUPPORTED' if v['h82_supported'] else 'no'}\n"
        f"  content positive: gap h={v['horizons'][0]} {v['gap_at_min_horizon']:+.3f} -> "
        f"h={v['horizons'][-1]} {v['gap_at_max_horizon']:+.3f} ({v['content_positive']})\n"
        f"  useful knee: free {v['free_floor']:.3f} -> faithful {v['faithful_ceiling']:.3f}, "
        f"ρ={v['knee_rho']:.2f} at {v['knee_calls']:.1f} calls vs {v['ceiling_calls']:.1f} "
        f"({'sub-linear' if v['useful_knee'] else 'no knee'})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
