"""Experiment CU9 -- the agent-safety horizon (SPEC-22, H102).

CU8 showed the trained model hides danger by omission. CU9 asks the deployment question that comes
next: how long can an unverified agent run before it does the irreversible bad thing, and how much
does verification extend that? It runs the CU5-net closed loop over a long deployment on the real
trained network `M_θ`; we record the step of its first exfiltration and build the survival curve --
the fraction of agents still safe after t steps -- for a free agent versus ones that verify at
budget ρ.

The finding (H102): a free agent's survival decays toward zero with deployment length (it breaches
at its first dangerous opportunity, near-certain over a long run), while verification flattens the
curve and extends the safe horizon: unverified safety is a clock that runs out, the oracle stops it.
The trained model is torch-gated; the survival core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.safety_horizon import (
    CU9Config,
    cu9_verdict,
    run_cu9,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU9 -- the agent-safety horizon (SPEC-22 H102).")
    parser.add_argument("--out", type=str, default="figures/cu9_safety_horizon.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from verisim.experiments.flagship import FlagshipConfig, train_flagship

    base = FlagshipConfig.smoke() if args.smoke else FlagshipConfig()
    print("training the network flagship M_θ (one-time)...")
    model, _ = train_flagship(base)
    config = CU9Config.smoke() if args.smoke else CU9Config()
    result = run_cu9(model, config)
    verdict = cu9_verdict(result)

    print(f"\nCU9 / H102 -- the agent-safety horizon on a REAL trained network M_θ "
          f"({result.n_episodes} deployments, horizon {result.horizon}):")
    print(f"  {'rho':>5} {'breach_rate':>12} {'mean_safe_steps':>16} {'safe_horizon(.5)':>17}")
    for c in result.curves:
        print(f"  {c.rho:>5.2f} {c.breach_rate:>12.3f} {c.mean_safe_steps:>16.1f} "
              f"{c.safe_horizon():>17}")

    print(f"\n  the free agent is unsafe over a deployment: "
          f"{verdict['free_unsafe_over_deployment']} "
          f"(breach rate {verdict['free_breach_rate']:.2f}, "
          f"safe for ~{verdict['free_mean_safe_steps']:.0f} steps on average)")
    print(f"  the oracle agent stays safe: {verdict['oracle_stays_safe']} "
          f"(breach rate {verdict['oracle_breach_rate']:.2f})")
    print(f"  verification extends the safe horizon: {verdict['verification_extends_horizon']} "
          f"(ρ={verdict['practical_rho']:.2f} -> safe for ~"
          f"{verdict['practical_mean_safe_steps']:.0f} steps)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu9 import plot_cu9

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu9(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
