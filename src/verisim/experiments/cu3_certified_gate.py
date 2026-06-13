"""Experiment CU3 -- the certified safety gate (SPEC-22 §5, H95).

The deepest synthesis of the program: an agent that doesn't just *act* safely (CU1/CU2) but
*certifies* its safety -- a distribution-free, finite-sample guarantee ``P(missed danger) ≤ α`` on
its gate, calibrated for free by the oracle (the SPEC-15 conformal idea applied to the agent's
allow/abort decision) -- and the measurement of what that guarantee costs.

The run ([`acd/certified_gate.py`](../acd/certified_gate.py)): a battery of host plans on the v0 fs
world, each labeled breach/safe by the oracle; an ensemble of ρ-grounded previews gives each plan a
risk score; `conformal.calibrate_threshold` sets the abort threshold certifying missed-danger ≤ α;
and the certificate's validity (split-averaged test missed-danger) + its cost (false-block on safe
plans) are measured across the consultation budget ρ.

The headline (H95): the certificate is **valid at every ρ** (missed-danger ≤ α), but its false-block
cost **collapses with faithfulness** -- a drifting preview can only honor the guarantee by aborting
*everything* (false-block ≈ 1, safe-but-useless), while the oracle-grounded preview certifies the
*same* guarantee while allowing the safe plans (false-block → 0 by ρ≈0.2). *Any* model can be made
safe by being useless; only a *faithful* one is safe **and** useful, ρ buys that ≈ free. CPU-only,
torch-free, seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.certified_gate import (
    CU3Config,
    cu3_verdict,
    run_cu3,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU3 -- the certified safety gate (SPEC-22 H95)."
    )
    parser.add_argument("--out", type=str, default="figures/cu3_certified_gate.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU3Config.smoke() if args.smoke else CU3Config()
    result = run_cu3(config)
    verdict = cu3_verdict(result)

    print(f"CU3 / H95 -- the certified safety gate (P(missed danger) <= alpha={result.alpha}, "
          f"{result.n_unsafe}/{result.n_plans} plans truly unsafe):")
    print(f"  {'rho':>5} {'missed-danger':>14} {'false-block':>12} {'abort':>7}")
    for c in result.cells:
        flag = "OK" if c.missed_danger <= result.alpha + 1e-9 else "VIOLATED"
        print(f"  {c.rho:>5.2f} {c.missed_danger:>10.3f} [{flag}] {c.false_block:>11.3f} "
              f"{c.abort_rate:>7.2f}")
    print(f"\n  certificate valid at every rho (missed-danger <= alpha): "
          f"{verdict['certificate_valid']}")
    print(f"  false-block cost falls with faithfulness (rho): "
          f"{verdict['false_block_falls_with_rho']} "
          f"({verdict['false_block_at_rho0']:.2f} -> {verdict['false_block_at_rho1']:.2f})")
    print(f"  the safety guarantee becomes ~free at rho = {verdict['cheap_certificate_rho']:.2f} "
          f"(false-block <= 0.05)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu3 import plot_cu3

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu3(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
