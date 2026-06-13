"""Experiment CU12 -- knowledge-free targeting: target the grammar, not the assets (SPEC-22, H105).

CU10/CU11 made targeted verification cheap and un-gameable by spending the oracle on the
``connect``-to-crown-jewel actions -- assuming the crown-jewel inventory is complete. CU12
asks what that cheap target costs when the inventory is wrong. It compares, against the *true*
sensitive set on the same battery and real trained `M_θ` as CU10: **uniform** (the blind baseline),
**asset-indexed** targeting at several inventory-completeness levels (verify ``connect`` to the
*known* jewels), and **grammar-indexed** targeting (verify *every* ``connect``).

The finding (H105): asset-indexed targeting degrades toward the unverified breach as the inventory
becomes incomplete -- a false sense of security, and fully gameable by an adversary who exfiltrates
to the unflagged host -- while grammar-indexed targeting reaches the oracle's zero breach
*inventory-independently* and is still far cheaper than the full oracle. When you cannot trust your
asset inventory, target the grammar, not the assets. The trained model is torch-gated; the core is
torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.knowledge_free_targeting import (
    CU12Config,
    cu12_verdict,
    run_cu12,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU12 -- knowledge-free targeting (SPEC-22 H105)."
    )
    parser.add_argument("--out", type=str, default="figures/cu12_knowledge_free_targeting.csv")
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

    config = CU12Config.smoke() if args.smoke else CU12Config()
    result = run_cu12(model, config)
    verdict = cu12_verdict(result)

    print(f"\nCU12 / H105 -- knowledge-free targeting on a REAL trained network M_θ "
          f"({result.n_episodes} deployments, horizon {result.horizon}, "
          f"|T|={result.n_true_sensitive}):")
    print(f"  {'defense':<26} {'breach(rand)':>12} {'breach(adv)':>12} {'calls':>8}")
    for c in (*result.uniform, *result.asset, result.grammar):
        print(f"  {c.label:<26} {c.breach_random:>12.3f} {c.breach_adversarial:>12.3f} "
              f"{c.mean_calls:>8.2f}")

    print(f"\n  asset-indexed degrades with the inventory: {verdict['asset_target_degrades']}"
          f" (K frac {verdict['incomplete_inventory_frac']:.2f}: breach "
          f"{verdict['incomplete_breach_random']:.3f}, adversarial "
          f"{verdict['incomplete_breach_adversarial']:.3f} -- gameable)")
    print(f"  grammar-indexed targeting is knowledge-free and safe: {verdict['grammar_is_safe']} "
          f"(breach {verdict['grammar_breach_random']:.3f} at {verdict['grammar_calls']:.2f} calls "
          f"vs the full oracle's {verdict['full_oracle_calls']:.2f} -- "
          f"{verdict['grammar_saving_vs_full']:.1f}x cheaper)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu12 import plot_cu12

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu12(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
