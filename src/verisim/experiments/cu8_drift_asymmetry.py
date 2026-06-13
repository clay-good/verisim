"""Experiment CU8 -- the drift asymmetry (SPEC-22, H101).

CU5-net found the trained network model's drift is one-sided. CU8 characterizes it: a teacher-forced
probe of the real trained `M_θ`, classifying every per-step flow-prediction error as an **omission**
(the oracle opened a flow the model missed -- a hidden danger) or a **hallucination** (the model
invented a flow the oracle did not open -- a false alarm), split by protected (danger) vs work
(benign) host.

The finding (H101): drift is **omission-biased** -- the model under-predicts consequences far more
than it over-predicts them -- and on the protected hosts the asymmetry is extreme (it omits real
exfil flows while hallucinating almost none). The safety implication: the catastrophic missed-danger
cell is the one drift inflates, so verification corrects the error mode the model is structurally
biased toward. The trained model is torch-gated; the probe core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.drift_asymmetry import (
    CU8Config,
    cu8_verdict,
    run_drift_asymmetry,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU8 -- the drift asymmetry (SPEC-22 H101).")
    parser.add_argument("--out", type=str, default="figures/cu8_drift_asymmetry.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from verisim.experiments.flagship import FlagshipConfig, train_flagship

    base = FlagshipConfig.smoke() if args.smoke else FlagshipConfig()
    print("training the network flagship M_θ (one-time)...")
    model, _ = train_flagship(base)
    config = CU8Config.smoke() if args.smoke else CU8Config()
    result = run_drift_asymmetry(model, config)
    verdict = cu8_verdict(result)

    print(f"\nCU8 / H101 -- the drift asymmetry on a REAL trained network M_θ "
          f"({result.n_workloads} workloads, {result.n_steps} steps):")
    print(f"  {'class':>10} {'true_opens':>11} {'omitted':>8} {'hallucinated':>13} {'recall':>7}")
    for cls in ("protected", "work", "other"):
        print(f"  {cls:>10} {result.true_opens[cls]:>11} {result.omitted[cls]:>8} "
              f"{result.hallucinated[cls]:>13} {result.recall(cls):>7.2f}")

    print(f"\n  drift is omission-biased: {verdict['drift_is_omission_biased']} "
          f"({verdict['total_omitted']} omissions vs "
          f"{verdict['total_hallucinated']} hallucinations)")
    print(f"  danger is hidden by omission: {verdict['danger_hidden_by_omission']} "
          f"(missed exfil {verdict['omitted_protected']} vs false exfil alarms "
          f"{verdict['hallucinated_protected']})")
    print(f"  the model foresaw only {verdict['protected_recall']:.0%} of true exfil flows")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu8 import plot_cu8

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu8(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
