"""HFL1 -- the host flagship `H_ε(ρ)` curve: does the smart-scheduling win cross to the host world?

SPEC-19's FL1 produced the headline flagship result on the *network* world: on a real trained `M_θ`,
the strict sub-linear-knee bar is not met, but a consultation policy that triggers on the model's
own **decode entropy** nearly *doubles* fixed-interval consultation at equal budget (+57% at ρ=0.2,
+94% at ρ=0.5) -- smart scheduling decisively beats the clock. That result is network-only. HFL0
froze a *host* flagship checkpoint (the harder world: `H_free` ~9 vs the network's ~18, `p`=0.70 vs
0.88) but never ran the curve. HFL1 runs it: the same four-arm `H_ε(ρ)` sweep (floor / fixed-ρ /
composed-ρ / ceiling) on the frozen host checkpoint, with the composed policy triggering on the
*host* model's real decode entropy (the SPEC-6 §8.1 uncertainty signal, the host analogue).

**H84 (the smart-scheduling win is cross-world).** On the harder host world, the composed
decode-entropy-triggered policy beats fixed-interval consultation at equal budget -- the FL1/H69
mechanism (FL6/H77: the real signal *ranks* drift well enough to schedule, even when it cannot
*calibrate* a coverage bound) reproduces where the model is materially less faithful. *Refuted if*
composed ties or loses to fixed on the host world -- which would localize the FL1 scheduling win to
the network world, a sharp negative that says the signal's drift-ranking quality is world-dependent
(the less-faithful host model's decode entropy does not order its drift well enough to schedule).
Either branch is bankable (SPEC.md §10.1). CPU-local; CI runs the smoke instance; the frontier run
is the committed figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.acd.host_integrity import make_workload
from verisim.conformal.calibrate import calibrate_threshold
from verisim.host.delta import apply
from verisim.hostloop import (
    PartialHostOracle,
    budget_for_rho,
    ground_truth_rollout,
    run_host_rollout,
)
from verisim.hostmetrics.divergence import divergence
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import (
    ConsultationPolicy,
    FixedInterval,
    Never,
    StepContext,
    fixed_interval_for_rho,
)
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel


@dataclass(frozen=True)
class HostComposedConsult:
    """The composed consultation policy on the host world (the FL1 `ComposedConsult` analogue).

    Consult when the model's per-step decode entropy exceeds the conformal threshold ``tau`` (the
    coverage-driven early consult, SPEC-15) *or* when the speculative draft window has elapsed (the
    bounded-drift backstop, SPEC-13). The runner's budget cap enforces the exact ``ρ`` on top, so
    the comparison against fixed-``ρ`` is at truly equal budget.
    """

    tau: float
    window: int

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    def should_consult(self, ctx: StepContext) -> bool:
        return ctx.signal > self.tau or (ctx.step + 1) % self.window == 0


@dataclass(frozen=True)
class HostFlagshipCurveConfig:
    """The HFL1 sweep: budgets, tolerance, and how the host conformal trigger is calibrated."""

    rhos: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)
    epsilon: float = 0.0
    driver: str = "forky"  # the in-distribution driver the host flagship was trained/evaluated on
    eval_seeds: tuple[int, ...] = (500, 501, 502, 503)
    eval_steps: int = 48
    cal_seeds: tuple[int, ...] = (600, 601, 602, 603)
    cal_steps: int = 32
    alpha: float = 0.1  # conformal target undetected-breach rate (SPEC-15)
    window: int = 8  # speculative max draft length (SPEC-13)
    num_threads: int = 1

    @staticmethod
    def smoke() -> HostFlagshipCurveConfig:
        return HostFlagshipCurveConfig(
            rhos=(0.1, 0.3), eval_seeds=(500, 501), eval_steps=12, cal_seeds=(600,),
            cal_steps=10, window=4,
        )


@dataclass(frozen=True)
class CurvePoint:
    """One (arm, ρ) cell: mean faithful horizon over eval seeds + bootstrap CI."""

    arm: str
    rho: float
    h_mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.rho:.4f},{self.h_mean:.6f},"
            f"{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


CSV_HEADER = "arm,rho,h_mean,ci_lo,ci_hi,n"


def build_calibration(
    model: NeuralHostWorldModel, config: HostFlagshipCurveConfig, oracle: HostOracle,
) -> tuple[list[float], list[int]]:
    """Free-run the *real* host model and record (decode entropy, breach@ε) per step (SPEC-15 CF1).

    The calibration distribution is the deployment distribution -- the model's own drifted
    free-running states -- so the conformal threshold is honest about where it is used. The signal
    is the host flagship's real decode entropy, not a stand-in (SPEC-19 §1, CF6).
    """
    partial = PartialHostOracle(oracle)
    scores: list[float] = []
    breaches: list[int] = []
    for seed in config.cal_seeds:
        start, actions = make_workload(seed, config.cal_steps, driver=config.driver, oracle=oracle)
        truth = ground_truth_rollout(partial, start, actions)
        state = start
        for t, action in enumerate(actions):
            delta, signal = model.predict_delta_with_uncertainty(state, action)
            predicted = apply(state, delta)
            scores.append(signal)
            breaches.append(1 if divergence(truth[t + 1], predicted) > config.epsilon else 0)
            state = predicted  # free-run (the deployment regime)
    return scores, breaches


def _horizons(
    model: NeuralHostWorldModel, policy: ConsultationPolicy, config: HostFlagshipCurveConfig,
    oracle: HostOracle, *, budget: int | None,
) -> list[float]:
    """Faithful horizon of one policy over the eval seeds (budget caps total consultations)."""
    partial = PartialHostOracle(oracle)
    out: list[float] = []
    for seed in config.eval_seeds:
        start, actions = make_workload(seed, config.eval_steps, driver=config.driver, oracle=oracle)
        record = run_host_rollout(
            model, partial, start, actions, policy,
            epsilon=config.epsilon, budget=budget, seed=seed,
        )
        out.append(float(faithful_horizon(list(record.divergences), config.epsilon)))
    return out


def _cell(arm: str, rho: float, horizons: list[float]) -> CurvePoint:
    lo, hi = bootstrap_ci(horizons, seed=0)
    return CurvePoint(arm, rho, fmean(horizons), lo, hi, len(horizons))


def run_host_flagship_curve(
    model: NeuralHostWorldModel, config: HostFlagshipCurveConfig | None = None, *,
    oracle: HostOracle | None = None,
) -> list[CurvePoint]:
    """The HFL1 four-arm sweep on the frozen host flagship: floor, fixed-ρ, composed-ρ, ceiling."""
    config = config or HostFlagshipCurveConfig()
    oracle = oracle or ReferenceHostOracle()

    if config.num_threads > 0:  # bit-determinism for the headline numbers
        import torch

        torch.set_num_threads(config.num_threads)

    scores, breaches = build_calibration(model, config, oracle)
    tau = calibrate_threshold(scores, breaches, config.alpha).tau

    points: list[CurvePoint] = []
    points.append(_cell("floor", 0.0, _horizons(model, Never(), config, oracle, budget=0)))
    points.append(
        _cell("ceiling", 1.0,
              _horizons(model, FixedInterval(1), config, oracle, budget=config.eval_steps))
    )
    for rho in config.rhos:
        budget = budget_for_rho(rho, config.eval_steps)
        fixed = _horizons(model, fixed_interval_for_rho(rho), config, oracle, budget=budget)
        composed = _horizons(
            model, HostComposedConsult(tau, config.window), config, oracle, budget=budget
        )
        points.append(_cell("fixed", rho, fixed))
        points.append(_cell("composed", rho, composed))
    return points


def headline_verdict(points: list[CurvePoint]) -> dict[str, Any]:
    """H84: does *composed* beat *fixed* at equal budget on the host world (FL1 win cross-world)?"""
    by: dict[tuple[str, float], float] = {(p.arm, p.rho): p.h_mean for p in points}
    floor = next((p.h_mean for p in points if p.arm == "floor"), 0.0)
    ceiling = next((p.h_mean for p in points if p.arm == "ceiling"), 0.0)
    rhos = sorted({p.rho for p in points if p.arm == "composed"})
    lifts: dict[float, float] = {}
    rel: dict[float, float] = {}
    for rho in rhos:
        c, f = by.get(("composed", rho), 0.0), by.get(("fixed", rho), 0.0)
        lifts[rho] = c - f
        rel[rho] = (c - f) / f if f > 0 else 0.0
    composed_beats_fixed = (
        all(lifts[rho] >= 0 for rho in rhos) and any(lifts[rho] > 0 for rho in rhos)
    )
    return {
        "floor": floor,
        "ceiling": ceiling,
        "rhos": rhos,
        "composed_minus_fixed": lifts,
        "composed_rel_lift": rel,
        "best_rel_lift": max(rel.values()) if rel else 0.0,
        "composed_beats_fixed": composed_beats_fixed,
    }


def write_csv(points: list[CurvePoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="HFL1 -- the host flagship H_ε(ρ) curve: smart scheduling, harder world (H84)."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/host-l")
    parser.add_argument("--out", type=str, default="figures/hfl1_host_curve.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = HostFlagshipCurveConfig.smoke() if args.smoke else HostFlagshipCurveConfig()
    if args.smoke:
        from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship

        model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    else:
        from verisim.experiments.host_flagship import load_checkpoint

        model = load_checkpoint(args.checkpoint).world_model

    points = run_host_flagship_curve(model, config)
    write_csv(points, args.out)
    v = headline_verdict(points)
    print(f"  floor (ρ=0): {v['floor']:.2f}   ceiling (ρ=1): {v['ceiling']:.2f}")
    for rho in v["rhos"]:
        c = next(p.h_mean for p in points if p.arm == "composed" and p.rho == rho)
        f = next(p.h_mean for p in points if p.arm == "fixed" and p.rho == rho)
        print(f"  ρ={rho:.2f}:  composed={c:.2f}  fixed={f:.2f}  "
              f"lift={v['composed_minus_fixed'][rho]:+.2f} ({v['composed_rel_lift'][rho]:+.0%})")
    print(f"H84 (smart scheduling beats the clock on the harder host world): "
          f"{'SUPPORTED' if v['composed_beats_fixed'] else 'no'}  "
          f"(best relative lift {v['best_rel_lift']:+.0%})")


if __name__ == "__main__":  # pragma: no cover
    main()
