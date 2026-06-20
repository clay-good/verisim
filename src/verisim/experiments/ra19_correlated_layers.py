"""Experiment RA19 -- the framing-transfer trap and layer saturation (H151, recast).

Computes, for every subset of {alignment, monitor, oracle} and every harm class, the residual
missed- harm under both ambient and direct framing -- calibrated to verisim's measured per-layer
rates (RA7-RA12). Honest headlines (after a 4-reviewer adversarial pass): (1) the oracle is the
unique framing-robust zero-residual closer; (2) the triad is flat by SATURATION (rho inert), not
correlation, and only under the ambient assumption -- under the monitor's measured direct recall
the stack is NOT flat; (3) the deployer's framing-transfer trap is a sensitivity (up to ~20x,
degrading to ~10x at 0.5), not a point; (4) genuine correlation non-factorization lives only off-
triad on disguised-ops. Hermetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.ra19_correlated_layers import (
    HARM_CLASSES,
    deployer_underestimate,
    ra19_verdict,
    run_ra19,
    subsets,
    trap_sensitivity,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA19 -- framing-transfer trap (H151).")
    parser.add_argument("--out", type=str, default="figures/ra19_correlated_layers.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    result = run_ra19()
    v = ra19_verdict(result)

    print("\nRA19 / H151 -- the framing-transfer trap and layer saturation (recast)\n")
    print("  residual_true grid (rho=1, AMBIENT framing):\n")
    labels = {s: ("+".join(x[:4] for x in s) if s else "none") for s in subsets()}
    print(f"    {'subset':16s} " + " ".join(f"{h[:9]:>9s}" for h in HARM_CLASSES))
    for s in subsets():
        print(f"    {labels[s]:16s} "
              + " ".join(f"{result.at(s, h, 1.0).residual_true:9.2f}" for h in HARM_CLASSES))

    du = deployer_underestimate("relational")
    print("\n  the framing-transfer trap (relational, S = alignment + monitor):")
    print(f"    deployer naive (monitor DIRECT recall, independent) = {du['naive_direct']:.3f}")
    print(f"    TRUE under ambient framing (shared blind spot) = {du['true_ambient']:.3f}")
    print(f"    => trap factor = {du['trap_factor']:.0f}x  (sensitivity: "
          f"0.55->{trap_sensitivity(0.55):.0f}x, 0.5->{trap_sensitivity(0.5):.0f}x)")
    print(f"    within-framing factor (factors on triad) = {du['same_framing_factor']:.1f}x")

    print("\n  verdict (honest recast):")
    for k in ("oracle_unique_framing_robust_closer", "triad_flat_by_saturation_under_ambient",
              "triad_NOT_flat_under_direct", "rho_inert_on_triad",
              "factorization_holds_within_framing_on_triad",
              "genuine_correlation_gap_on_disguised_ops", "inversion_on_explicit"):
        print(f"    {k} = {v[k]}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra19 import plot_ra19

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra19(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
