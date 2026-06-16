"""Experiment CU35 -- the verifier-fidelity condition: the dual coverage law (H128).

The targeting arc hard-codes one assumption it never names: when the schedule consults, the oracle
returns the EXACT verdict. Its only variable was the target (which actions to consult), and its
theorem is that the target must COVER the danger surface. CU35 turns the other knob -- the
verifier's own fidelity -- and finds the matching dual condition: the verifier must be FAITHFUL ON
the danger surface. On-surface fidelity is load-bearing (sloped vs nature, a cliff vs the
adversary); off-surface fidelity is irrelevant to safety (a verifier globally wrong but exact on the
danger grammar is as safe as a perfect oracle, buying only false blocks). Both coverage conditions
are independently necessary. Torch-free (worst-case omitter + CU21 net/host/segmentation); ~8s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.verifier_fidelity import (
    CU35Config,
    arm_verdict,
    cu35_verdict,
    run_cu35,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU35 -- the verifier-fidelity condition (H128).")
    parser.add_argument("--out", type=str, default="figures/cu35_verifier_fidelity.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU35Config.smoke() if args.smoke else CU35Config()
    result = run_cu35(config)
    v = cu35_verdict(result)

    print("\nCU35 / H128 -- the verifier-fidelity condition (the dual of CU21 target coverage):\n")
    for arm in result.arms:
        a = arm_verdict(arm)
        print(f"  {arm.world_name} (target covers={arm.target_covers}, n={arm.n_scenarios}):")
        print(f"    {'φ on-surface':>13s}  {'random':>7s}  {'adversarial':>11s}")
        for p in arm.on_surface:
            print(f"    {p.fidelity:13.2f}  {p.random_breach:7.3f}  {p.adversarial_breach:11.3f}")
        print(f"    {'ψ off-surface':>13s}  {'adv brch':>8s}  {'false blocks':>12s}")
        for p in arm.off_surface:
            print(f"    {p.fidelity:13.2f}  {p.adversarial_breach:8.3f}  "
                  f"{p.mean_false_blocks:12.2f}")
        print(f"    horizon vs nature={a['random_axis_is_a_horizon']} / cliff vs adversary="
              f"{a['adversarial_is_a_cliff']}; off-surface flat-safe="
              f"{a['offsurface_breach_flat_safe']} "
              f"costs-utility={a['offsurface_drift_costs_utility']}\n")

    g = result.grid
    print(f"  THE 2x2 ({g.world_name}) -- adversarial breach across coverage x verifier fidelity:")
    print(f"    {'':22s} {'target covers':>14s} {'target non-cov':>15s}")
    print(f"    {'verifier exact':22s} {g.covers_exact:14.3f} {g.leak_exact:15.3f}")
    print(f"    {'verifier blind':22s} {g.covers_blind:14.3f} {g.leak_blind:15.3f}")
    print(f"    both conditions independently necessary = {v['both_conditions_necessary']}")
    print(f"\n  exact-verifier safe everywhere = {v['exact_verifier_safe_everywhere']}; "
          f"on-surface-blind leaks everywhere = {v['surface_blind_leaks_everywhere']}")
    print(f"  off-surface fidelity irrelevant to safety = "
          f"{v['offsurface_fidelity_irrelevant_to_safety']} (costs utility somewhere = "
          f"{v['offsurface_drift_costs_utility_somewhere']})")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu35 import plot_cu35

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu35(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
