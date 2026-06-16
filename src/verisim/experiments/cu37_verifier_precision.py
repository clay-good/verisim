"""Experiment CU37 -- the verifier precision tax: the utility half of the dual coverage law (H130).

CU36 grounded the *safety* half of CU35's verifier-fidelity law against real verifiers (a verifier
is as safe as the oracle iff it observes the channel the danger mutates). CU37 grounds the *utility*
half with the most ordinary host defense -- a file-integrity monitor that watches a set of paths and
blocks any write to one of them, serving as both the targeting surface and the verifier. Its two
structural properties are independent: COVERAGE of the danger file /cfg sets safety (as-safe-as-the-
oracle iff /cfg is watched, however coarse, and as-leaky-as-no-gate otherwise), and PRECISION sets
utility (over-watching benign files costs a false-block tax that rises monotonically with coarseness
while safety stays pinned at 0). So a defender can coarsen the monitor freely -- for cheapness or
for robustness to an incomplete inventory -- at zero safety cost, paying only a bounded utility tax.
Torch-free (worst-case omitter + exact host oracle); ~5s.
"""

from __future__ import annotations

import argparse

from verisim.acd.footprintless_targeting import CU34Config
from verisim.acd.verifier_precision import (
    cu37_verdict,
    orthogonality_2x2,
    run_cu37,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU37 -- the verifier precision tax (H130).")
    parser.add_argument("--out", type=str, default="figures/cu37_verifier_precision.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU34Config.smoke() if args.smoke else CU34Config()
    result = run_cu37(config)
    v = cu37_verdict(result)

    print("\nCU37 / H130 -- the verifier precision tax (the utility half of the dual law):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}\n")
    print("  COVERING monitors (all watch /cfg) -- safety FLAT at 0, utility tax rises:")
    print(f"    {'watched files':>13s} {'adv_breach':>11s} {'false_blocks':>13s} {'calls':>7s}")
    for p in result.covering:
        print(f"    {p.granularity:13d} {p.adversarial_breach:11.3f} "
              f"{p.mean_false_blocks:13.3f} {p.mean_calls:7.2f}")
    print("\n  SUB-COVERAGE monitors (miss /cfg) -- leak however coarse (precision != safety):")
    print(f"    {'watched files':>13s} {'adv_breach':>11s} {'false_blocks':>13s}")
    for p in result.subcoverage:
        print(f"    {p.granularity:13d} {p.adversarial_breach:11.3f} {p.mean_false_blocks:13.3f}")

    print("\n  the orthogonality 2x2 {covers /cfg?} x {watches benign files?}:")
    print(f"    {'corner':18s} {'adv_breach':>11s} {'false_blocks':>13s}")
    for c in orthogonality_2x2(result):
        print(f"    {c.label:18s} {c.adversarial_breach:11.3f} {c.mean_false_blocks:13.3f}")

    print(f"\n  covering is safe at every precision (coverage sets safety) = "
          f"{v['covering_safe_at_every_precision']}")
    print(f"  utility tax rises monotonically with coarseness (precision sets utility) = "
          f"{v['utility_tax_rises_with_coarseness']}")
    print(f"  sub-coverage leaks at every precision (the coverage cliff) = "
          f"{v['subcoverage_leaks_at_every_precision']}")
    print(f"  faithful_on_surface == covers /cfg in every cell (the a-priori predictor) = "
          f"{v['faithful_iff_covers']}")
    print(f"  ORTHOGONAL: breach tracks coverage only = {v['breach_tracks_coverage_only']}, "
          f"utility tracks precision only = {v['utility_tracks_precision_only']}")

    out = write_csv(result, args.out)
    print(f"\n  wrote {out}")

    if args.plot:
        from figures.plot_cu37 import plot_cu37

        path = plot_cu37(result, args.plot)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
