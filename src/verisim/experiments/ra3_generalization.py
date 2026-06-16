"""Experiment RA3 -- the generalization battery: the gate is task- and attack-agnostic (H135).

Runs the RA1 gate over a large randomized task distribution and a diverse injection-attack taxonomy,
neither hand-tuned, and reports the aggregate safety/utility/cost plus a per-attack-class breakdown.
The headline: across hundreds of tasks and every attack class it never saw, the covering gate drives
missed-danger to zero with no utility loss -- because the gate's safety is a property of the danger
grammar (coverage), not the task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.generalization import (
    RA3Result,
    cu_ra3_verdict,
    run_ra3,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA3 -- the generalization battery (H135).")
    parser.add_argument("--out", type=str, default="figures/ra3_generalization.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate", type=float, default=0.3)
    args = parser.parse_args()

    result: RA3Result = run_ra3(n_tasks=args.n, seed=args.seed, injection_rate=args.rate)
    v = cu_ra3_verdict(result)
    by = {c.schedule: c for c in result.base.cells}

    print("\nRA3 / H135 -- the generalization battery (the gate is task- and attack-agnostic):")
    print(f"  {result.n_tasks} randomized tasks ({result.n_injected} injected) across "
          f"{len(result.classes)} attack classes, seed {result.seed}\n")

    print(f"    {'schedule':14s} {'missed_danger':>13s} {'task_success':>12s} {'calls/task':>11s}")
    for s in ("undefended", "target", "full_oracle"):
        c = by[s]
        print(f"    {s:14s} {c.missed_danger_rate:13.3f} {c.task_success_rate:12.3f} "
              f"{c.mean_oracle_calls:11.2f}")

    print("\n  PER-ATTACK-CLASS (undefended breach -> covering-gate breach, covered?):")
    print(f"    {'class':14s} {'n':>4s} {'undef':>7s} {'gate':>6s} {'covered':>8s}")
    for cb in result.classes:
        print(f"    {cb.danger_class:14s} {cb.n_tasks:4d} {cb.undefended_breach:7.2f} "
              f"{cb.target_breach:6.2f} {cb.covered!s:>8s}")

    print(f"\n  undefended breaches across the distribution = "
          f"{v['undefended_breaches_across_distribution']} ({v['undefended_missed_danger']:.3f})")
    print(f"  covering gate generalizes to zero = {v['gate_generalizes_to_zero']} "
          f"with no utility loss = {v['no_utility_loss']} ({v['call_saving']:.1f}x cheaper)")
    print(f"  every attack class caught = {v['every_class_caught']}, "
          f"every class breaches undefended = {v['every_class_breaches_undefended']}, "
          f"coverage holds for every attack = {v['covers_all_attacks']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra3 import plot_ra3

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra3(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
