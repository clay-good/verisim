"""UA9 -- the useful knee: buying content-keyed faithfulness at budget ρ (SPEC-20 §7, H81).

UA8 (H80) proved faithfulness is *load-bearing* for the content-keyed file-integrity task -- but it
only measured the two extremes: the faithful predictor (the oracle every step, ρ=1, catch 1.000) and
the free predictor (raw `M_θ`, ρ=0, catch 0.50-0.73). The program's whole thesis is that you do not
have to choose: the oracle-in-the-loop *buys back* faithfulness cheaply at a consultation budget ρ
(SPEC-19's `H_ε(ρ)` curve). That curve has never been measured on the *downstream* task where
faithfulness actually matters -- on the structural-control tasks (UA2-UA7) there was no advantage
for ρ to recover (six nulls; H76/UA4 found the grounding advantage *flat* in ρ), so the curve was
the degenerate flat line. UA9 measures it where the advantage exists:

    the ρ-grounded predictor (re-anchor `M_θ` to the oracle's truth every round(1/ρ) steps) sweeps
    the catch rate from the free floor (ρ=0) to the faithful ceiling (ρ=1), on the content task.

**H81 (the useful knee -- the synthesis).** On the content-keyed task, the grounded predictor's
catch rate is (a) monotone non-decreasing in ρ -- the H76/UA4 *mirror*: the grounding advantage UA4
found flat on structural control *is* monotone here, on the task whose optimal policy depends on the
content the model drifts on -- and (b) recovers ≥90% of the faithful ceiling at *sub-linear* budget
(ρ < 1, fewer oracle calls than the every-step faithful predictor) -- the "useful knee," SPEC-19's
headline mechanism demonstrated on SPEC-20's downstream task. *Refuted if* the catch rate is flat in
ρ (a little grounding is as good as none -> the H81 null mirrors UA4 and faithfulness, though
load-bearing, is not cheaply buyable here) or only the ρ=1 every-step predictor recovers the ceiling
(content drift so fast that partial grounding never helps -> faithfulness is load-bearing *and*
uncheaply-buyable, the negative that says the cheap-faithful story fails where it counts).

No training; CPU-local; CI runs the smoke instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.acd.host_integrity import grounded_defense_reward, make_workload
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.metrics.aggregate import bootstrap_ci

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel


@dataclass(frozen=True)
class GroundedKneeConfig:
    """The UA9 sweep: catch rate vs consultation budget ρ at a fixed content-keyed horizon."""

    horizon: int = 14  # where the free predictor is weakest (UA8: free=0.50) -> most headroom
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    budget: int = 2
    workload_seeds: tuple[int, ...] = tuple(range(700, 724))
    driver: str = "forky"
    knee_frac: float = 0.9  # "recovers the ceiling" = ≥ knee_frac of the faithful (ρ=1) catch rate

    @staticmethod
    def smoke() -> GroundedKneeConfig:
        return GroundedKneeConfig(
            horizon=8, rhos=(0.0, 0.5, 1.0), budget=2, workload_seeds=(700, 701, 702)
        )


@dataclass(frozen=True)
class KneePoint:
    """One ρ cell: mean catch rate + bootstrap CI + mean oracle-call cost."""

    rho: float
    reward: float
    ci_lo: float
    ci_hi: float
    oracle_calls: float  # mean consultations the policy spent (the budget actually paid)
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.rho:.4f},{self.reward:.6f},{self.ci_lo:.6f},"
            f"{self.ci_hi:.6f},{self.oracle_calls:.4f},{self.n}"
        )


CSV_HEADER = "rho,reward,ci_lo,ci_hi,oracle_calls,n"


def run_grounded_knee(
    model: NeuralHostWorldModel, config: GroundedKneeConfig | None = None, *,
    oracle: HostOracle | None = None,
) -> list[KneePoint]:
    """Sweep the budget ρ; score the ρ-grounded predictive-defense catch rate + cost."""
    config = config or GroundedKneeConfig()
    oracle = oracle or ReferenceHostOracle()
    workloads = [
        make_workload(s, config.horizon, driver=config.driver, oracle=oracle)
        for s in config.workload_seeds
    ]
    points: list[KneePoint] = []
    for rho in config.rhos:
        rewards: list[float] = []
        calls: list[int] = []
        for start, actions in workloads:
            r, c = grounded_defense_reward(model, oracle, start, actions, config.budget, rho)
            rewards.append(r)
            calls.append(c)
        lo, hi = bootstrap_ci(rewards, seed=0)
        points.append(KneePoint(rho, fmean(rewards), lo, hi, fmean(calls), len(rewards)))
    return points


def h81_verdict(points: list[KneePoint], knee_frac: float = 0.9) -> dict[str, Any]:
    """H81: catch rate is monotone in ρ (the H76 mirror) and recovers the ceiling cheaply.

    - ``monotone_in_rho``: non-decreasing catch rate (within a small CI-noise tolerance) -- the
      grounding advantage UA4/H76 found *flat* on structural control is recovered here.
    - ``recoverable_gap``: faithful (ρ=1) beats free (ρ=0) -- faithfulness is load-bearing (inherits
      H80) and so there is something for ρ to buy back.
    - ``knee_rho`` / ``knee_calls``: the smallest ρ (and its mean oracle-call cost) that reaches
      ``knee_frac`` of the faithful ceiling -- the "useful knee."
    - ``h81_supported``: a recoverable gap, monotone recovery, and a *sub-linear* knee (ρ < 1).
    """
    by = {p.rho: p for p in points}
    rhos = sorted(by)
    floor, ceil = by[rhos[0]].reward, by[rhos[-1]].reward  # ρ=0 free, ρ=1 faithful
    ceil_calls = by[rhos[-1]].oracle_calls
    rewards = [by[r].reward for r in rhos]
    # monotone within CI-noise tolerance (a dip no larger than the widest half-CI is allowed)
    tol = max((p.ci_hi - p.ci_lo) / 2 for p in points)
    monotone = all(rewards[i + 1] >= rewards[i] - tol for i in range(len(rewards) - 1))
    recoverable_gap = ceil - floor
    threshold = knee_frac * ceil
    knee = next((r for r in rhos if by[r].reward >= threshold), rhos[-1])
    return {
        "rhos": rhos,
        "reward_by_rho": {r: by[r].reward for r in rhos},
        "calls_by_rho": {r: by[r].oracle_calls for r in rhos},
        "free_floor": floor,
        "faithful_ceiling": ceil,
        "recoverable_gap": recoverable_gap,
        "monotone_in_rho": monotone,
        "knee_rho": knee,
        "knee_calls": by[knee].oracle_calls,
        "ceiling_calls": ceil_calls,
        "sublinear_knee": knee < rhos[-1] and by[knee].oracle_calls < ceil_calls,
        # H81: faithfulness is load-bearing (gap), recovery tracks ρ (monotone), bought sub-linearly
        "h81_supported": (
            recoverable_gap > 0.02
            and monotone
            and knee < rhos[-1]
            and by[knee].oracle_calls < ceil_calls
        ),
    }


def write_csv(points: list[KneePoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="UA9 -- the useful knee: buying content-keyed faithfulness at budget ρ (H81)."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/host-l")
    parser.add_argument("--out", type=str, default="figures/ua9_grounded_knee.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = GroundedKneeConfig.smoke() if args.smoke else GroundedKneeConfig()
    if args.smoke:
        from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship

        model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    else:
        from verisim.experiments.host_flagship import load_checkpoint

        model = load_checkpoint(args.checkpoint).world_model

    points = run_grounded_knee(model, config)
    write_csv(points, args.out)
    for p in points:
        print(f"  ρ={p.rho:.2f}:  catch={p.reward:.3f}  "
              f"[{p.ci_lo:.3f}, {p.ci_hi:.3f}]  oracle_calls={p.oracle_calls:.1f}")
    v = h81_verdict(points, config.knee_frac)
    print(
        f"H81 (the useful knee -- content-keyed faithfulness bought at budget ρ): "
        f"{'SUPPORTED' if v['h81_supported'] else 'no'}\n"
        f"  free floor (ρ=0) {v['free_floor']:.3f} -> faithful ceiling (ρ=1) "
        f"{v['faithful_ceiling']:.3f} (recoverable gap {v['recoverable_gap']:+.3f}); "
        f"monotone_in_ρ={v['monotone_in_rho']}\n"
        f"  useful knee: ρ={v['knee_rho']:.2f} reaches ≥{int(config.knee_frac * 100)}% of the "
        f"ceiling at {v['knee_calls']:.1f} oracle calls vs {v['ceiling_calls']:.1f} for the "
        f"every-step faithful predictor "
        f"({'sub-linear' if v['sublinear_knee'] else 'NOT sub-linear'})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
