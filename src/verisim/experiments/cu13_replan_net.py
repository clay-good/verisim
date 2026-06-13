"""Experiment CU13 -- capability under real drift: the false-alarm channel prices CU6 and CU7.

CU6 (replanning amplifies harm) and CU7 (verify-before-commit saves 2.1x) were measured on the
two-sided synthetic stand-in. CU13 re-runs both on the *real* trained network ``M_θ`` -- whose
drift is one-sided (it omits exfil flows, recall ~0, and never hallucinates one, false-alarm ~0) --
and isolates the mechanism with two dials on a synthetic net model:

  - a **false-alarm sweep** (recall fixed at 0) on which CU6's harm-amplification rises from zero;
  - a **recall sweep** (false-alarm fixed at 0) on which CU7's verify-before-commit saving rises
    from 1x.

The real ``M_θ`` anchors at the origin of *both* (its measured false-alarm and recall are ~0), so
it shows no amplification and no saving -- exactly where the mechanism predicts. The unifying
finding (H106): CU6's amplification is priced by the model's *wrong* "no"s and CU7's saving by its
*right* ones; a real omission-biased world model says "yes" to everything, so both are artifacts of
two-sided drift. What survives is the verify-before-commit *zero-harm guarantee* and the omission
danger itself (the agent stays unsafe), which only the oracle -- or the CU10-CU12 structural
targeting -- removes. The trained model is torch-gated; the replanning core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.closed_loop_replan_net import CU13Config, cu13_verdict, run_cu13, write_csv


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU13 -- capability under real drift (SPEC-22 H106)."
    )
    parser.add_argument("--out", type=str, default="figures/cu13_replan_net.csv")
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

    config = CU13Config.smoke() if args.smoke else CU13Config()
    result = run_cu13(model, config)
    verdict = cu13_verdict(result)

    print(f"\nCU13 / H106 -- capability under real drift on a REAL trained network M_θ "
          f"({result.n_goals} goals, {result.n_routes} routes):")

    print("\n  CU6 dial -- harm-amplification is priced by the FALSE-ALARM rate (recall=0):")
    print(f"    {'false-alarm':>12} {'1-shot harm':>12} {'replan harm':>12} {'amplification':>13}")
    for c in result.fa_sweep:
        print(f"    {c.false_alarm_rate:>12.3f} {c.one_shot_harm:>12.3f} {c.replanner_harm:>12.3f} "
              f"{c.amplification:>13.3f}")

    print("\n  CU7 dial -- verify-before-commit saving is priced by the RECALL (false-alarm=0):")
    print(f"    {'recall':>12} {'VBC calls':>12} {'full calls':>12} {'saving x':>10} {'wasted':>8}")
    for c in result.recall_sweep:
        print(f"    {c.recall_rate:>12.3f} {c.vbc_calls:>12.2f} {c.full_verify_calls:>12.2f} "
              f"{c.cost_saving:>10.2f} {c.wasted_fraction:>8.2f}")

    real = result.real
    assert real is not None
    print(f"\n  the REAL M_θ anchor (measured false-alarm {real.false_alarm_rate:.4f}, "
          f"recall {real.recall_rate:.4f} -- says 'yes' to everything):")
    print(f"    amplification {real.amplification:+.3f} (CU6's +0.06: a two-sided artifact)")
    print(f"    cost-saving   {real.cost_saving:.3f}x  (CU7's 2.1x: a recall-dependent artifact)")
    print(f"    but the agent is STILL unsafe: one-shot harm {real.one_shot_harm:.3f}; "
          f"verify-before-commit stays zero-harm ({real.vbc_harm:.3f}).")

    print(f"\n  amplification priced by false alarm: "
          f"{verdict['amplification_priced_by_false_alarm']} "
          f"(0 -> {verdict['fa_hi_amplification']:.3f})")
    print(f"  saving priced by recall: {verdict['saving_priced_by_recall']} "
          f"(1.0 -> {verdict['rc_hi_saving']:.2f}x)")
    print(f"  real model at the origin of both: amplification {verdict['real_amplification']:.3f}, "
          f"saving {verdict['real_cost_saving']:.3f}x")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu13 import plot_cu13

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu13(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
