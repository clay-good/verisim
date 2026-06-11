"""UA8 -- predictive file-integrity defense on the host world (SPEC-20 §7, H80; the positive side).

The content-keyed task the cross-world law predicts is the exception. A faithful predictor (oracle
rollout) vs a free predictor (raw host `M_θ` rollout) each protect the budget files they expect
corrupted; both scored against the *true* corruptions. Sweeping the workload horizon is the knob:
short horizons sit inside the host faithful horizon (`H_free≈9`) so faithful ≈ free; longer horizons
compound the model's content drift (~25% on the written-file set) so the free predictor protects the
wrong files. H80: the faithful predictor beats the free one and the gap widens with horizon -- the
positive that closes the SPEC-20 boundary from the other side (control keyed on the *content* the
model drifts on *does* need faithfulness). No training; CPU-local; CI runs the smoke instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.acd.host_integrity import (
    make_workload,
    model_step,
    oracle_step,
    predictive_defense_reward,
)
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.metrics.aggregate import bootstrap_ci

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel


@dataclass(frozen=True)
class HostIntegrityConfig:
    """The UA8 sweep: a budget-limited predictive-defense over the workload horizon."""

    horizons: tuple[int, ...] = (6, 10, 14, 18)
    budget: int = 2  # files the defender can protect (binds, so prediction quality matters)
    workload_seeds: tuple[int, ...] = tuple(range(700, 724))
    driver: str = "forky"

    @staticmethod
    def smoke() -> HostIntegrityConfig:
        return HostIntegrityConfig(horizons=(4, 8), budget=2, workload_seeds=(700, 701, 702))


@dataclass(frozen=True)
class IntegrityPoint:
    """One (predictor, horizon) cell: mean predictive-defense reward + bootstrap CI."""

    predictor: str  # "faithful" | "free"
    horizon: int
    reward: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.predictor},{self.horizon},{self.reward:.6f},"
            f"{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


CSV_HEADER = "predictor,horizon,reward,ci_lo,ci_hi,n"


def run_host_integrity(
    model: NeuralHostWorldModel, config: HostIntegrityConfig | None = None, *,
    oracle: HostOracle | None = None,
) -> list[IntegrityPoint]:
    """Sweep the workload horizon; score the faithful vs free predictor's protective defense."""
    config = config or HostIntegrityConfig()
    oracle = oracle or ReferenceHostOracle()
    faithful = oracle_step(oracle)
    free = model_step(model)
    true_step = oracle_step(oracle)

    points: list[IntegrityPoint] = []
    for horizon in config.horizons:
        workloads = [
            make_workload(s, horizon, driver=config.driver, oracle=oracle)
            for s in config.workload_seeds
        ]
        for predictor_name, predictor in (("faithful", faithful), ("free", free)):
            rewards = [
                predictive_defense_reward(predictor, true_step, start, actions, config.budget)
                for start, actions in workloads
            ]
            lo, hi = bootstrap_ci(rewards, seed=0)
            points.append(
                IntegrityPoint(predictor_name, horizon, fmean(rewards), lo, hi, len(rewards))
            )
    return points


def h80_verdict(points: list[IntegrityPoint]) -> dict[str, Any]:
    """H80: the faithful predictor beats the free one, and the gap widens with horizon."""
    by: dict[tuple[str, int], float] = {(p.predictor, p.horizon): p.reward for p in points}
    horizons = sorted({p.horizon for p in points})
    gaps = {
        h: by[("faithful", h)] - by[("free", h)] for h in horizons if ("free", h) in by
    }
    hmin, hmax = horizons[0], horizons[-1]
    return {
        "horizons": horizons,
        "faithful_minus_free": gaps,
        "gap_at_min_horizon": gaps.get(hmin, 0.0),
        "gap_at_max_horizon": gaps.get(hmax, 0.0),
        # H80: faithfulness helps (positive gap at depth) AND the gap grows with the horizon
        "h80_supported": gaps.get(hmax, 0.0) > 0.02 and gaps.get(hmax, 0.0) > gaps.get(hmin, 0.0),
        "widens_with_horizon": gaps.get(hmax, 0.0) > gaps.get(hmin, 0.0),
    }


def write_csv(points: list[IntegrityPoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="UA8 -- predictive file-integrity defense (H80).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/host-l")
    parser.add_argument("--out", type=str, default="figures/ua8_host_integrity.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = HostIntegrityConfig.smoke() if args.smoke else HostIntegrityConfig()
    if args.smoke:
        from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship

        model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    else:
        from verisim.experiments.host_flagship import load_checkpoint

        model = load_checkpoint(args.checkpoint).world_model

    points = run_host_integrity(model, config)
    write_csv(points, args.out)
    for h in sorted({p.horizon for p in points}):
        c = {p.predictor: p.reward for p in points if p.horizon == h}
        print(f"  horizon={h:2d} (H_free~9):  faithful={c['faithful']:.3f}  free={c['free']:.3f}  "
              f"gap={c['faithful'] - c['free']:+.3f}")
    v = h80_verdict(points)
    print(f"H80 (content-keyed control needs faithfulness): "
          f"{'SUPPORTED' if v['h80_supported'] else 'no'}  "
          f"(gap h={v['horizons'][0]}: {v['gap_at_min_horizon']:+.3f} "
          f"-> h={v['horizons'][-1]}: {v['gap_at_max_horizon']:+.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
