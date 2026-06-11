"""FL1 -- the headline `H_ε(ρ)` curve on the frozen flagship M_θ (SPEC-19 §3, milestone FL1).

This is the deliverable: the single figure the program has promised since SPEC.md §3 -- faithful
horizon vs oracle budget on a **real trained network world model**, end-to-end, against the exact
oracle, with the *composed* consultation policy. Every prior version of this curve was on a
controlled stand-in (the SR/CF experiments) or measured only the `ρ=0` floor (SPEC-10). FL1 runs the
whole loop on the one frozen checkpoint FL0 produced.

Four arms on one `H_ε(ρ)` axis (SPEC-19 §3):

  - **floor** -- `ρ=0`, no oracle (the SPEC-10 free-running number, the left anchor);
  - **fixed-ρ** -- consult on a clock (`fixed_interval_for_rho`), the naive baseline;
  - **composed π_c** -- the program's best policy: a conformal trigger on the model's **real**
    uncertainty signal (SPEC-15) OR a speculative draft-window cap (SPEC-13), budget-capped;
  - **ceiling** -- `ρ=1`, oracle every step (`H_ε=T`).

H69 (the headline) asks whether *composed* clears ≥80% of the ceiling horizon at ≤20% budget on a
real model. The honest-negative branch -- floor+cliff survives the whole stack -- is first-class
and, per SPEC-19 §0, the more consequential outcome if it lands. **The crux nobody else can see:**
the conformal arm rides the flagship's *real* decode-entropy signal, and SPEC-15 CF6 already found
the real network `belief_var` is **not conformalizable** (validity holds, efficiency null). So FL1
is the real test of whether composing the methods on a real model beats fixed -- or whether CF6's
negative dominates and the smart policy collapses to the clock. Either way the figure is exact,
because the oracle labels every breach for free.

The composed policy drops into the shipped per-step runner unchanged (no runner edits, SPEC-19 §2):
the conformal OR-clause supplies the early, coverage-driven consult; the window clause caps the
draft length (the per-step runner's budgeted approximation of SPEC-13's accept-longest-prefix -- the
full window-verify rollout lives in :mod:`verisim.experiments.sr_common`). CPU-local; CI runs the
smoke instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from verisim.conformal.calibrate import calibrate_threshold
from verisim.loop.policy import (
    ConsultationPolicy,
    FixedInterval,
    Never,
    StepContext,
    fixed_interval_for_rho,
)
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon
from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.net.state import NetworkState
from verisim.netdelta.apply import apply
from verisim.netloop import PartialNetOracle, budget_for_rho, ground_truth_rollout, run_net_rollout
from verisim.netmetrics.divergence import divergence
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .en1 import eval_actions
from .flagship import load_checkpoint

if TYPE_CHECKING:
    from verisim.net.action import NetAction
    from verisim.netdelta.edits import NetDelta


@runtime_checkable
class UncertainNetModel(Protocol):
    """A proposer that exposes BOTH a delta prediction and a per-step uncertainty signal.

    The flagship arms (flat ``NeuralNetworkWorldModel`` and structured ``GraphRSSMWorldModel``) both
    satisfy this; the combined protocol is the honest type for FL1/FL4, where the loop needs
    ``predict_delta`` (the runner) and the conformal calibration needs the uncertainty signal. The
    shipped ``NetModel`` / ``NetUncertaintyModel`` protocols each declare only one of the two.
    """

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta: ...

    def predict_delta_with_uncertainty(
        self, state: NetworkState, action: NetAction
    ) -> tuple[NetDelta, float]: ...


@dataclass(frozen=True)
class ComposedConsult:
    """The composed consultation policy: conformal trigger OR speculative draft-window cap (FL1).

    ``tau`` is the conformal threshold (SPEC-15, calibrated on the model's real signal); ``window``
    is the speculative max draft length (SPEC-13). Consult when the model's per-step uncertainty
    exceeds the calibrated threshold (the coverage-driven early consult) *or* when the draft window
    has elapsed (the bounded-drift backstop). The runner's budget cap + spend-down then enforce the
    exact ``ρ`` on top, so the comparison against fixed-``ρ`` is at truly equal budget.
    """

    tau: float
    window: int

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    def should_consult(self, ctx: StepContext) -> bool:
        return ctx.signal > self.tau or (ctx.step + 1) % self.window == 0


@dataclass(frozen=True)
class FlagshipCurveConfig:
    """The FL1 sweep: which budgets, what tolerance, and how the conformal trigger is calibrated."""

    rhos: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)  # interior budgets (anchors implicit)
    epsilon: float = 0.0
    eval_driver: str = "weighted"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    eval_steps: int = 96
    cal_driver: str = "weighted"
    cal_seeds: tuple[int, ...] = (300, 301, 302, 303)
    cal_steps: int = 64
    alpha: float = 0.1  # conformal target undetected-breach rate (SPEC-15)
    window: int = 8  # speculative max draft length (SPEC-13)
    num_threads: int = 1

    @staticmethod
    def smoke() -> FlagshipCurveConfig:
        return FlagshipCurveConfig(
            rhos=(0.1, 0.3), eval_seeds=(100, 101), eval_steps=16, cal_seeds=(300,),
            cal_steps=12, window=4,
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
    world_model: UncertainNetModel, config: FlagshipCurveConfig, oracle: NetOracle,
    net: NetConfig,
) -> tuple[list[float], list[int]]:
    """Free-run the *real* model and record (uncertainty signal, breach@ε) per step (SPEC-15 CF1).

    The calibration distribution is the deployment distribution -- the model's own drifted
    free-running states -- so the conformal threshold is honest about where it will be used (the
    non-exchangeable regime CF2/ACI is built for; FL1 uses the static threshold as the SPEC-15 §2.1
    baseline). The signal is the flagship's real decode entropy, *not* a controlled stand-in --
    which is the whole point (SPEC-19 §1, CF6).
    """
    partial = PartialNetOracle(oracle)
    scores: list[float] = []
    breaches: list[int] = []
    for seed in config.cal_seeds:
        actions = eval_actions(oracle, net, config.cal_driver, seed, config.cal_steps)
        truth = ground_truth_rollout(partial, NetworkState.initial(net.hosts), actions)
        state = NetworkState.initial(net.hosts)
        for t, action in enumerate(actions):
            delta, signal = world_model.predict_delta_with_uncertainty(state, action)
            predicted = apply(state, delta)
            scores.append(signal)
            breaches.append(1 if divergence(truth[t + 1], predicted) > config.epsilon else 0)
            state = predicted  # free-run (the deployment regime)
    return scores, breaches


def _horizons(
    world_model: UncertainNetModel, policy: ConsultationPolicy, config: FlagshipCurveConfig,
    oracle: NetOracle, net: NetConfig, *, budget: int | None,
) -> list[float]:
    """Faithful horizon of one policy over the eval seeds (budget caps total consultations)."""
    partial = PartialNetOracle(oracle)
    out: list[float] = []
    for seed in config.eval_seeds:
        actions = eval_actions(oracle, net, config.eval_driver, seed, config.eval_steps)
        record = run_net_rollout(
            world_model, partial, NetworkState.initial(net.hosts), actions, policy,
            epsilon=config.epsilon, budget=budget, seed=seed,
        )
        out.append(float(faithful_horizon(list(record.divergences), config.epsilon)))
    return out


def _cell(arm: str, rho: float, horizons: list[float]) -> CurvePoint:
    lo, hi = bootstrap_ci(horizons, seed=0)
    return CurvePoint(arm, rho, fmean(horizons), lo, hi, len(horizons))


def run_flagship_curve(
    world_model: UncertainNetModel, config: FlagshipCurveConfig | None = None, *,
    oracle: NetOracle | None = None,
) -> list[CurvePoint]:
    """The FL1 four-arm sweep on the frozen flagship: floor, fixed-ρ, composed-ρ, ceiling."""
    config = config or FlagshipCurveConfig()
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG

    if config.num_threads > 0:  # bit-determinism for the headline numbers
        import torch

        torch.set_num_threads(config.num_threads)

    scores, breaches = build_calibration(world_model, config, oracle, net)
    tau = calibrate_threshold(scores, breaches, config.alpha).tau

    points: list[CurvePoint] = []
    # floor (ρ=0) and ceiling (ρ=1) are budget anchors, measured once
    floor_h = _horizons(world_model, Never(), config, oracle, net, budget=0)
    points.append(_cell("floor", 0.0, floor_h))
    ceil_budget = config.eval_steps
    points.append(
        _cell("ceiling", 1.0,
              _horizons(world_model, FixedInterval(1), config, oracle, net, budget=ceil_budget))
    )
    for rho in config.rhos:
        budget = budget_for_rho(rho, config.eval_steps)
        fixed_pol = fixed_interval_for_rho(rho)
        fixed = _horizons(world_model, fixed_pol, config, oracle, net, budget=budget)
        composed = _horizons(
            world_model, ComposedConsult(tau, config.window), config, oracle, net, budget=budget
        )
        points.append(_cell("fixed", rho, fixed))
        points.append(_cell("composed", rho, composed))
    return points


def headline_verdict(
    points: list[CurvePoint], *, target_frac: float = 0.8, max_rho: float = 0.2
) -> dict[str, Any]:
    """H69: does *composed* clear ``target_frac`` of the ceiling horizon at ``ρ ≤ max_rho``?"""
    ceiling = next((p.h_mean for p in points if p.arm == "ceiling"), 0.0)
    floor = next((p.h_mean for p in points if p.arm == "floor"), 0.0)
    composed = [p for p in points if p.arm == "composed" and p.rho <= max_rho]
    fixed = {p.rho: p.h_mean for p in points if p.arm == "fixed"}
    best = max(composed, key=lambda p: p.h_mean) if composed else None
    target = target_frac * ceiling
    beats_fixed = (
        best is not None and best.rho in fixed and best.h_mean > fixed[best.rho]
    )
    return {
        "ceiling": ceiling,
        "floor": floor,
        "target_horizon": target,
        "best_composed_at_low_rho": None if best is None else best.h_mean,
        "best_composed_rho": None if best is None else best.rho,
        "h69_supported": best is not None and best.h_mean >= target,
        "composed_beats_fixed_at_best_rho": beats_fixed,
    }


def write_csv(points: list[CurvePoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="FL1 -- the flagship H_ε(ρ) curve (SPEC-19).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/fl1_flagship_curve.csv")
    parser.add_argument("--plot", type=str, default="figures/fl1_flagship_curve.png")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    config = FlagshipCurveConfig.smoke() if args.smoke else FlagshipCurveConfig()
    points = run_flagship_curve(ckpt.world_model, config)
    path = write_csv(points, args.out)
    print(f"wrote {len(points)} rows to {path}")
    verdict = headline_verdict(points)
    for p in sorted(points, key=lambda p: (p.arm, p.rho)):
        print(f"  {p.arm:9s} ρ={p.rho:.2f}  H={p.h_mean:6.2f}  [{p.ci_lo:.2f}, {p.ci_hi:.2f}]")
    print(f"H69: {'SUPPORTED' if verdict['h69_supported'] else 'not supported'}  {verdict}")
    try:
        from figures.plot_flagship_curve import plot_flagship_curve

        plot_flagship_curve(points, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
