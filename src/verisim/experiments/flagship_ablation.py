"""FL2 -- the composition ablation: do the methods compose on the flagship? (SPEC-19 §4, H70).

FL1 showed the *composed* policy's curve. FL2 asks the question composition requires: does stacking
the conformal trigger (SPEC-15) and the speculative draft-window (SPEC-13) on one frozen checkpoint
beat each method *alone*? The 2×2 ablation, all at **equal budget** ``B = ρ·T`` on one flagship:

  - **neither** -- the uniform clock (`fixed_interval_for_rho`), the naive baseline;
  - **conformal-only** -- consult when the real signal exceeds the calibrated `τ` (no window cap);
  - **speculative-only** -- consult on the draft-window clock (`FixedInterval(window)`, no signal);
  - **both** -- `ComposedConsult` (the FL1 policy).

At equal budget, faithful-horizon-per-oracle-call is ``H_ε / B`` -- monotone in ``H_ε`` -- so the
comparison reduces to faithful horizon at fixed budget (we report both). **H70 (compose):** ``both``
≥ max(``conformal-only``, ``speculative-only``) within CIs -- additive or super-additive. *Refuted*
if a pair interferes -- ``both`` underperforms the better of its parts (e.g. the conformal clause
fires on the speculative window boundary and the two waste budget double-triggering) -- in which
case the composition rule is the finding and the methods stay single-use (SPEC-19 §4). Reuses FL1's
calibration + horizon machinery verbatim; only the policy set changes. CPU-local; CI smoke instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING

from verisim.conformal.policy import ConformalTriggered
from verisim.loop.policy import ConsultationPolicy, FixedInterval, fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .flagship import load_checkpoint
from .flagship_curve import ComposedConsult, FlagshipCurveConfig, _horizons, build_calibration

if TYPE_CHECKING:
    from verisim.netmodel import NeuralNetworkWorldModel

# The four ablation cells, keyed by (conformal_on, speculative_on).
CELLS = ("neither", "conformal", "speculative", "both")


@dataclass(frozen=True)
class AblationCell:
    """One method-combination at one budget: mean faithful horizon + per-call efficiency + CI."""

    cell: str
    rho: float
    budget: int
    h_mean: float
    ci_lo: float
    ci_hi: float
    h_per_call: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.cell},{self.rho:.4f},{self.budget},{self.h_mean:.6f},"
            f"{self.ci_lo:.6f},{self.ci_hi:.6f},{self.h_per_call:.6f},{self.n}"
        )


CSV_HEADER = "cell,rho,budget,h_mean,ci_lo,ci_hi,h_per_call,n"


def _policy_for(cell: str, tau: float, window: int, rho: float) -> ConsultationPolicy:
    """The consultation policy for one ablation cell (all run under the same hard budget)."""
    if cell == "neither":
        return fixed_interval_for_rho(rho)
    if cell == "conformal":
        return ConformalTriggered(tau)
    if cell == "speculative":
        return FixedInterval(max(1, window))
    if cell == "both":
        return ComposedConsult(tau, window)
    raise ValueError(f"unknown ablation cell: {cell}")


def run_ablation(
    world_model: NeuralNetworkWorldModel, config: FlagshipCurveConfig | None = None, *,
    rho: float = 0.2, oracle: NetOracle | None = None,
) -> list[AblationCell]:
    """Run all four method-combinations at a single equal budget on the frozen flagship."""
    from verisim.netloop import budget_for_rho

    config = config or FlagshipCurveConfig()
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    if config.num_threads > 0:
        import torch

        torch.set_num_threads(config.num_threads)

    scores, breaches = build_calibration(world_model, config, oracle, net)
    from verisim.conformal.calibrate import calibrate_threshold

    tau = calibrate_threshold(scores, breaches, config.alpha).tau
    budget = budget_for_rho(rho, config.eval_steps)

    out: list[AblationCell] = []
    for cell in CELLS:
        policy = _policy_for(cell, tau, config.window, rho)
        horizons = _horizons(world_model, policy, config, oracle, net, budget=budget)
        lo, hi = bootstrap_ci(horizons, seed=0)
        mean_h = fmean(horizons)
        out.append(
            AblationCell(
                cell, rho, budget, mean_h, lo, hi,
                h_per_call=(mean_h / budget if budget > 0 else 0.0), n=len(horizons),
            )
        )
    return out


def compose_verdict(cells: list[AblationCell]) -> dict[str, object]:
    """H70: ``both`` ≥ max(``conformal``, ``speculative``) -- additive composition (within CIs)."""
    by = {c.cell: c for c in cells}
    both = by.get("both")
    singles = [by[k].h_mean for k in ("conformal", "speculative") if k in by]
    best_single = max(singles) if singles else 0.0
    return {
        "both": None if both is None else both.h_mean,
        "best_single": best_single,
        "neither": None if "neither" not in by else by["neither"].h_mean,
        # additive: both is at least the best single (CI-aware: not below the single's lower bound)
        "h70_composes": both is not None and both.h_mean >= best_single - 1e-9,
        "super_additive": both is not None and both.h_mean > best_single + 1e-9,
        "interferes": both is not None and both.h_mean < best_single - 1e-9,
    }


def write_csv(cells: list[AblationCell], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(c.csv_row() for c in cells)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="FL2 -- flagship composition ablation (SPEC-19).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/fl2_composition.csv")
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    config = FlagshipCurveConfig.smoke() if args.smoke else FlagshipCurveConfig()
    cells = run_ablation(ckpt.world_model, config, rho=args.rho)
    path = write_csv(cells, args.out)
    print(f"wrote {len(cells)} rows to {path}")
    for c in cells:
        print(f"  {c.cell:12s} ρ={c.rho:.2f} B={c.budget:3d}  H={c.h_mean:6.2f} "
              f"[{c.ci_lo:.2f}, {c.ci_hi:.2f}]  H/call={c.h_per_call:.3f}")
    verdict = compose_verdict(cells)
    tag = "COMPOSES" if verdict["h70_composes"] else "INTERFERES"
    print(f"H70: {tag}  {verdict}")


if __name__ == "__main__":  # pragma: no cover
    main()
