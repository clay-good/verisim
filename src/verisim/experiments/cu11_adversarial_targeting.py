"""Experiment CU11 -- un-gameable targeting: the adversary controls the timing (SPEC-22, H104).

CU10 compared three verification schedules on a *random* workload. CU11 asks the cyber question CU10
left open: what happens when the adversary knows the schedule and chooses *when* to exfiltrate? It
re-scores the same long-deployment battery and the same real trained network `M_θ` as CU10 under a
worst-case-over-timing attacker, alongside CU10's random-timing breach.

The finding (H104): **uniform** and **model** targeting are gameable -- an attacker who fires the
steps the schedule does not check pushes breach back toward 1.0 for every ρ<1, erasing CU9's uniform
knee -- while **structure** targeting is un-gameable -- its breach stays at the oracle's 0.0,
because the danger surface is grammar-fixed (a flow is born only by `connect`, addressed to its dst)
and the attacker cannot move danger off it. The defender principle: target verification at what the
adversary cannot move. The trained model is torch-gated; the verification core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.adversarial_targeting import cu11_verdict, run_cu11, write_csv
from verisim.acd.targeted_verification import CU10Config


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU11 -- un-gameable targeting (SPEC-22 H104)."
    )
    parser.add_argument("--out", type=str, default="figures/cu11_adversarial_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default="runs/flagship/net-l",
        help="reuse a frozen flagship checkpoint instead of retraining (the ~11-min train).",
    )
    parser.add_argument("--retrain", action="store_true", help="train a fresh flagship M_θ.")
    args = parser.parse_args()

    if args.retrain or not Path(args.checkpoint, "model.pt").exists():
        from verisim.experiments.flagship import FlagshipConfig, train_flagship

        base = FlagshipConfig.smoke() if args.smoke else FlagshipConfig()
        print("training the network flagship M_θ (one-time)...")
        model, _ = train_flagship(base)
    else:
        from verisim.experiments.flagship import load_checkpoint

        print(f"reusing frozen flagship checkpoint {args.checkpoint} (no retrain)...")
        model = load_checkpoint(args.checkpoint).world_model

    config = CU10Config.smoke() if args.smoke else CU10Config()
    result = run_cu11(model, config)
    verdict = cu11_verdict(result)

    print(f"\nCU11 / H104 -- un-gameable targeting on a REAL trained network M_θ "
          f"({result.n_episodes} deployments, horizon {result.horizon}):")
    print(f"  {'schedule':<26} {'random':>10} {'adversarial':>12} {'calls':>8}")
    for c in result.uniform:
        print(f"  {c.label:<26} {c.random_breach:>10.3f} {c.adversarial_breach:>12.3f} "
              f"{c.mean_calls:>8.2f}")
    for c in (result.model, result.structure):
        print(f"  {c.label:<26} {c.random_breach:>10.3f} {c.adversarial_breach:>12.3f} "
              f"{c.mean_calls:>8.2f}")

    print(f"\n  structure targeting is un-gameable: {verdict['structure_is_ungameable']} "
          f"(adversarial breach {verdict['structure_adversarial_breach']:.3f} at "
          f"{verdict['structure_calls']:.2f} calls)")
    print(f"  uniform knee is a mirage: {verdict['uniform_is_gameable']} (ρ="
          f"{verdict['uniform_knee_rho']:g}: random breach "
          f"{verdict['uniform_knee_random_breach']:.3f} -> adversarial "
          f"{verdict['uniform_knee_adversarial_breach']:.3f})")
    print(f"  model self-targeting is gameable: {verdict['model_is_gameable']} "
          f"(adversarial breach {verdict['model_adversarial_breach']:.3f})")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu11 import plot_cu11

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu11(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
