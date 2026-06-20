"""Experiment RA16 -- the oracle gate as an AI-control protocol: the unsubvertible monitor (H148).

Sweeps the blue team's audit budget and reports the safety-usefulness control frontier for four
protocols (no gate / trusted model-monitor / permission denylist / oracle coverage), under both a
non-adaptive and an adaptive red team. The result: the model-monitor's safety against a non-adaptive
attacker rises with the audit budget (the textbook control tradeoff) but collapses under an adaptive
attacker (arXiv 2510.09462), reaching safety=1 only at audit-everything (usefulness 0); the oracle
coverage gate sits in the (safety=1, usefulness=1) corner at a sparse, fixed audit budget under BOTH
attackers, because its verdict is read from the exact state, not predicted -- so there is no blind
spot to aim at. This is "complete mediation made cheap" stated as an AI-control frontier.

Hermetic engine (torch-free, seeded). The external number is the same live lane as RA5/RA6.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.control_eval import (
    ControlEvalResult,
    cu_ra16_verdict,
    run_control_eval,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA16 -- the oracle as a control protocol (H148).")
    parser.add_argument("--out", type=str, default="figures/ra16_control_eval.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    result: ControlEvalResult = run_control_eval()
    v = cu_ra16_verdict(result)

    print("\nRA16 / H148 -- the oracle gate as an AI-control protocol (the unsubvertible monitor)")
    print(f"  monitor fidelity phi={result.phi}, oracle audit budget = surface density "
          f"{result.surface_density:.3f}\n")
    print(f"    {'protocol':16s} {'bgt':>5s} {'audit':>6s} {'Srnd':>5s} {'Sadv':>5s} {'util':>5s}")
    for p in result.points:
        print(f"    {p.protocol:16s} {p.budget:7.2f} {p.audit_rate:7.3f} {p.safety_random:7.2f} "
              f"{p.safety_adaptive:7.2f} {p.usefulness:7.2f}")

    rng = v["monitor_random_safety_range"]
    lo_f, hi_f = float(rng[0]), float(rng[1])  # type: ignore[index]
    print("\n  the control frontier:")
    print(f"    monitor safety vs a NON-adaptive attacker rises with budget = "
          f"{v['monitor_random_safety_rises_with_budget']} ({lo_f:.2f} -> {hi_f:.2f})")
    print(f"    monitor safety vs an ADAPTIVE attacker is capped below 1 while it keeps autonomy = "
          f"{v['monitor_adaptive_capped_below_1_with_autonomy']} "
          f"(best {v['monitor_best_adaptive_safety_with_autonomy']:.2f})")
    corner = v["oracle_adaptive_safe"] and v["oracle_random_safe"] and v["oracle_full_usefulness"]
    print(f"    oracle is in the (safety=1, usefulness=1) corner under BOTH attackers = {corner}")
    print(f"    ...at a sparse fixed audit budget = {v['oracle_audit_budget']:.3f} "
          f"(sparse = {v['oracle_budget_is_sparse']})")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra16 import plot_ra16

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra16(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
