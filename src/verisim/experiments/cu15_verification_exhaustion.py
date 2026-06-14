"""Experiment CU15 -- the verification-exhaustion attack: the cost axis under an adversary (H108).

CU11 proved structure targeting is un-gameable on the *safety* axis. CU15 carries the worst-case
threat model to the *cost* axis: an adversary who cannot make structure breach floods its danger
surface (every ``connect``-to-jewel) with benign-looking activity to exhaust the verification budget
(alert fatigue / denial of budget). It sweeps the attacker's saturation on the same long-deployment
battery and the same real trained network ``M_θ`` as CU10/CU11, reading each schedule on both axes.

The finding (H108): an adversary can move **exactly one axis** of a sub-oracle schedule --
structure's *cost* (a bill: it climbs toward the full oracle as the flood grows) or uniform's
*safety* (a breach: its clock-fixed cost cannot be inflated, but off-clock exfils push breach up).
Prefer the schedule
whose movable axis is the bill: structure's cost stays bounded by and weakly dominates the full
oracle, and the attack is self-limiting -- each attacker action adds one defender call and zero
breaches. The trained model is torch-gated; the verification core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.verification_exhaustion import CU15Config, cu15_verdict, run_cu15, write_csv


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU15 -- the verification-exhaustion attack (SPEC-22 H108)."
    )
    parser.add_argument("--out", type=str, default="figures/cu15_verification_exhaustion.csv")
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

    config = CU15Config.smoke() if args.smoke else CU15Config()
    result = run_cu15(model, config)
    verdict = cu15_verdict(result)

    print(f"\nCU15 / H108 -- the verification-exhaustion attack on a REAL trained network M_θ "
          f"({result.n_episodes} deployments, horizon {result.horizon}):")
    print(f"  {'saturation':>10}  {'schedule':<24} {'breach':>8} {'calls':>8}")
    for s in config.saturations:
        for c in (c for c in result.cells if c.saturation == s):
            print(f"  {s:>10.2f}  {c.label:<24} {c.breach_rate:>8.3f} {c.mean_calls:>8.2f}")

    print(f"\n  structure SAFETY is immovable: {verdict['structure_safety_immovable']} "
          f"(breach <= {verdict['structure_breach_max']:.3f} at every saturation)")
    print(f"  structure COST is gameable: {verdict['structure_cost_gameable']} "
          f"({verdict['structure_calls_free']:.2f} -> {verdict['structure_calls_saturated']:.2f} "
          f"calls, full oracle = {verdict['full_oracle_calls']:.2f})")
    print(f"  structure weakly dominates the full oracle: "
          f"{verdict['structure_dominates_full_oracle']}")
    print(f"  the attack is self-limiting: {verdict['cost_per_attacker_action']:.2f} defender "
          f"calls per attacker action, {verdict['breaches_bought']:.3f} breaches bought")
    print(f"  uniform is the mirror: cost immovable {verdict['uniform_cost_immovable']} "
          f"({verdict['uniform_calls_free']:.2f} -> {verdict['uniform_calls_saturated']:.2f}), "
          f"safety gameable {verdict['uniform_safety_gameable']} "
          f"({verdict['uniform_breach_free']:.3f} -> {verdict['uniform_breach_saturated']:.3f})")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu15 import plot_cu15

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu15(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
