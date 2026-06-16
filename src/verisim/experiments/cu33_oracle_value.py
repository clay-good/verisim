"""Experiment CU33 -- the value of the oracle: the cost-optimal verification policy (H126).

The CU arc reports two numbers per policy -- a breach rate and an oracle-call count -- but never in
a common currency. A defender's objective is expected operational loss
``L = C * p_breach + c * calls`` (``C`` = cost of a breach, ``c`` = cost of one oracle call). CU33
converts the arc into a decision
rule. Under a worst-case adversary the coverage theorem (CU11/CU21) pins every non-covering policy's
breach at 1, so the tuning dial collapses to a binary "accept the loss (free) or cover it
(structure)"; the structure target Pareto-dominates the full oracle, so the rule is: verify iff
``C/c > calls_structure`` (a handful of oracle calls). The dial is real only against nature. Torch-
free; reuses unified_targeting's arms; ~6s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.oracle_value import (
    CU33Config,
    cu33_verdict,
    run_cu33,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU33 -- the value of the oracle (H126).")
    parser.add_argument("--out", type=str, default="figures/cu33_oracle_value.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    result = run_cu33(CU33Config())
    v = cu33_verdict(result)

    print("\nCU33 / H126 -- the value of the oracle (the cost-optimal verification policy):")
    print(f"  {len(result.arms)} worlds, demo={result.demo_world}, horizon {result.horizon}\n")
    print(f"  {'world':10s} {'policy':14s} {'adv':>6s} {'rand':>6s} {'calls':>7s}")
    for a in result.arms:
        for p in a.points:
            print(f"  {a.world_name:10s} {p.name:14s} {p.adversarial_breach:6.3f} "
                  f"{p.random_breach:6.3f} {p.calls:7.2f}")

    print("\n  THE DIAL COLLAPSE (under an adversary the breach is pinned, so partial budgets are "
          "strictly wasted):")
    print(f"    structure dominates the full oracle everywhere = "
          f"{v['structure_dominates_full_everywhere']}")
    print(f"    full oracle never cost-optimal              = "
          f"{v['full_oracle_never_optimal_everywhere']}")
    print(f"    non-covering (uniform/model) never optimal  = "
          f"{v['noncovering_never_optimal_everywhere']}")

    print("\n  THE DECISION RULE: verify the structure surface iff C/c > calls_structure")
    crit = v["critical_ratios"]
    sav = v["structure_call_savings"]
    assert isinstance(crit, dict) and isinstance(sav, dict)
    for w in crit:
        print(f"    {w:10s}: threshold C/c = {crit[w]:.2f} calls   "
              f"(structure is {sav[w]:.1f}x cheaper than the full oracle)")

    print("\n  THE DIAL (uniform breach vs budget) -- sloped vs nature, flat vs adversary:")
    print(f"    {'rho':>4s} {'rand':>6s} {'adv':>6s} {'calls':>7s}")
    for d in result.dial_curve:
        print(f"    {d.rho:4.2f} {d.random_breach:6.3f} {d.adversarial_breach:6.3f} {d.calls:7.2f}")
    print(f"    nature dial sloped = {v['nature_dial_is_sloped']}   "
          f"adversary dial flat = {v['adversary_dial_is_flat']}   "
          f"dial collapses under adversary = {v['dial_collapses_under_adversary']}")

    print("\n  EXPECTED LOSS vs STAKES (units of c, worst case) -- the cost-optimal policy:")
    print(f"    {'C/c':>6s} {'free':>8s} {'structure':>10s} {'full':>8s}  optimal")
    for lp in result.loss_curve:
        print(f"    {lp.ratio:6.1f} {lp.free_loss:8.2f} {lp.structure_loss:10.2f} "
              f"{lp.full_loss:8.2f}  {lp.optimal_name}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu33 import plot_cu33

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu33(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
