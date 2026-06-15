"""Experiment CU25 -- the composite under real drift: high per-leg foresight is not safety (H118).

CU24 proved the composition theorem on the worst-case omitter. CU25 closes the trained-arm rigor
gap (the CU5-net / CU19 / CU20 tradition): it re-runs the composite network threat model on the real
trained ``M_theta`` (the frozen SPEC-19/20 flagship ``runs/flagship/net-l``, no retrain) and
measures what the model actually self-governs, leg by leg. The headline: the model's per-leg
self-governance recall is heterogeneous (it foresees the structural outage ~0.78 and exposure ~0.57
legs but is blind to the content exfil ~0.07 leg), yet model self-targeting is adversarially
breached on *every* leg -- including the high-recall structural legs at ~1.000 -- because
the worst-case adversary needs a single blind spot. Only the model-free union target is safe and
un-gameable on every leg (model-independently, the composition theorem on the real model).
Torch-gated (LP7).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.availability_targeting import CU22Config
from verisim.acd.composite_targeting import LEG_NAMES
from verisim.acd.composite_trained import cu25_verdict, run_cu25, write_csv


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU25 -- the composite under real drift (H118).")
    parser.add_argument("--out", type=str, default="figures/cu25_composite_trained.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default="runs/flagship/net-l",
        help="reuse the frozen flagship checkpoint (no retrain -- the ~11-min train).",
    )
    args = parser.parse_args()

    from verisim.experiments.flagship import load_checkpoint

    print(f"reusing frozen flagship checkpoint {args.checkpoint} (no retrain)...")
    model = load_checkpoint(args.checkpoint).world_model

    config = (
        CU22Config(horizon=24, n_seeds=200, max_episodes=24)
        if args.smoke else CU22Config()
    )
    result = run_cu25(model, config)
    v = cu25_verdict(result)

    print(f"\nCU25 / H118 -- the composite on a REAL trained network M_θ "
          f"({result.n_episodes} deployments, horizon {result.horizon}):\n")
    print("  the boundary law at the composite (per-leg self-governance under real drift):")
    print(f"    {'leg':10s} {'recall':>8s} {'model-self-target adv breach':>30s}")
    for leg in LEG_NAMES:
        print(f"    {leg:10s} {result.self_gov_recall[leg]:8.3f} "
              f"{result.model_adv_breach[leg]:30.3f}")
    print(f"\n  FORESIGHT HETEROGENEOUS (blind on content exfil, partial on structural legs)="
          f"{v['foresight_heterogeneous']}")
    print(f"  HIGH FORESIGHT IS NOT SAFETY: outage recall {result.self_gov_recall['outage']:.2f} "
          f"yet adversarially breached {result.model_adv_breach['outage']:.3f} = "
          f"{v['high_foresight_leg_still_breached']}")
    print(f"  model self-targeting fails EVERY leg={v['model_self_targeting_fails_every_leg']} "
          f"(composite adv breach {result.model_composite_adv:.3f})")
    print(f"\n  the model-free UNION TARGET (composition theorem on the real model): "
          f"safe on every leg={v['union_target_safe_on_every_leg']}  covers={v['union_covers']}  "
          f"at {result.union_target_calls:.2f} calls = "
          f"{v['composite_call_saving']:.1f}x cheaper than the full oracle "
          f"({result.full_oracle_calls:.0f})")

    write_csv(result, args.out)
    print(f"\nwrote {args.out}")
    try:
        from figures.plot_cu25 import plot_cu25

        plot_path = Path(args.plot) if args.plot else Path(args.out).with_suffix(".png")
        plot_cu25(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
