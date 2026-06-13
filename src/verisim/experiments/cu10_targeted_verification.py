"""Experiment CU10 -- targeted verification: what to verify beats how much (SPEC-22, H103).

CU9 showed an agent on a blind, uniform verification schedule only reaches zero breach at the full
oracle. CU10 asks the practitioner's follow-up: which steps should a limited budget verify? It
compares three schedules on the same long-deployment battery and the same real trained network
`M_θ` as CU9: **uniform** (the blind CU9 budget), **model** self-targeting (consult when the model
expects activity), and **structure** targeting (consult the `connect`-to-protected actions -- the
defender's crown-jewel knowledge).

The finding (H103): uniform needs ≈the full oracle to reach zero breach; model self-targeting fails
(the omission bias of CU8 hides the danger from the model, so it consults almost never on the steps
that matter); structure targeting reaches the oracle's zero-breach safety at a small fraction of the
calls -- danger is concentrated, so it is cheap to defend if you know where to look. The trained
model is torch-gated; the verification core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.targeted_verification import (
    CU10Config,
    cu10_verdict,
    run_cu10,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU10 -- targeted verification (SPEC-22 H103)."
    )
    parser.add_argument("--out", type=str, default="figures/cu10_targeted_verification.csv")
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
    result = run_cu10(model, config)
    verdict = cu10_verdict(result)

    print(f"\nCU10 / H103 -- targeted verification on a REAL trained network M_θ "
          f"({result.n_episodes} deployments, horizon {result.horizon}):")
    print(f"  {'schedule':<26} {'breach_rate':>12} {'mean_calls':>11}")
    for c in result.uniform:
        print(f"  {c.label:<26} {c.breach_rate:>12.3f} {c.mean_calls:>11.2f}")
    for c in (result.model, result.structure):
        print(f"  {c.label:<26} {c.breach_rate:>12.3f} {c.mean_calls:>11.2f}")

    print(f"\n  structure targeting reaches the oracle's safety: {verdict['structure_is_safe']} "
          f"(breach {verdict['structure_breach_rate']:.3f} at "
          f"{verdict['structure_calls']:.2f} calls vs the full oracle's "
          f"{verdict['full_oracle_calls']:.2f}) -- {verdict['structure_call_saving']:.1f}x cheaper")
    print(f"  model self-targeting fails: {verdict['model_self_targeting_fails']} "
          f"(breach {verdict['model_breach_rate']:.3f} at only "
          f"{verdict['model_calls']:.2f} calls -- it cannot see its own omissions)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu10 import plot_cu10

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu10(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
