"""Experiment CU7 -- verify-before-commit (SPEC-22 §5, H99).

CU6 warned that free replanning amplifies harm. CU7 gives the constructive fix an agent builder can
act on: **the harm only happens at the moment of commit**, so verification belongs at exactly one
place -- the route the agent is about to execute. A verify-before-commit agent replans freely (cheap
model search) and spends one oracle call to verify the route it commits to; it verifies the model's
"yes" and trusts its "no" (an abort can never cause harm).

The finding (H99): verify-before-commit reaches the **zero-harm guarantee at a fraction of a
full-verification agent's oracle cost** -- because a uniform "verify-everything" agent wastes most
of its calls verifying routes it was going to abort anyway, and that waste grows with how
adversarial the environment is. Where you verify beats how much. CPU-only, torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.closed_loop_verify import (
    CU7Config,
    cu7_verdict,
    run_cu7,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU7 -- verify-before-commit (SPEC-22 H99).")
    parser.add_argument("--out", type=str, default="figures/cu7_verify_before_commit.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU7Config.smoke() if args.smoke else CU7Config()
    result = run_cu7(config)
    verdict = cu7_verdict(result)

    print(f"CU7 / H99 -- verify-before-commit (phi={result.phi}, {result.n_goals} goals, "
          f"{result.n_routes} routes/goal):")
    print("  budgeted replanner frontier:")
    print(f"    {'rho':>5} {'calls':>7} {'harm':>7} {'success':>8}")
    for c in result.budgeted:
        print(f"    {c.rho:>5.2f} {c.calls:>7.2f} {c.harm_rate:>7.3f} {c.success_rate:>8.3f}")
    for p in (result.full_verify, result.vbc):
        print(f"  {p.label}: calls={p.calls:.2f} harm={p.harm_rate:.3f} "
              f"success={p.success_rate:.3f}")

    print(f"\n  the free agent is unsafe: {verdict['free_agent_unsafe']} "
          f"(harm {result.budgeted[0].harm_rate:.3f})")
    print(f"  verify-before-commit reaches the zero-harm guarantee: {verdict['vbc_zero_harm']}")
    print(f"  ...at {verdict['cost_saving_factor']:.1f}x lower oracle cost than full verification "
          f"({verdict['vbc_calls']:.2f} vs {verdict['full_verify_calls']:.2f} calls/goal)")
    print(f"  fraction of full verification that is wasted on 'no' decisions: "
          f"{verdict['wasted_fraction_of_full_verify']:.0%}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu7 import plot_cu7

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu7(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
