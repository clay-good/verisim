"""Experiment CU22 -- the generative test: the framework predicts a defense it never saw (H115).

CU21 proved the four targeting defenses are one model-free covering rule. The skeptic's objection is
that CU21 *unified results we already had*. CU22 answers it: it applies the CU21
``unified_targeting`` engine, verbatim, to a danger the whole arc never studied --
**availability** (a self-inflicted
outage: an automated containment agent disconnecting a critical work service). The framework's
``covers`` invariant is used as an a-priori predictor: it classifies the existing target catalogue
(connect-to-jewel CU10, exposure-closure CU17, a syntactic disconnect rule) as non-covering ->
leaks, and the framework-derived disconnect-closure as covering -> safe + cheap + un-gameable. The
worst-case-omitter run then confirms every prediction. Torch-free; ~80s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.availability_targeting import (
    CU22Config,
    cu22_verdict,
    run_cu22,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU22 -- the generative test (H115).")
    parser.add_argument("--out", type=str, default="figures/cu22_availability_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU22Config.smoke() if args.smoke else CU22Config()
    result = run_cu22(config)
    v = cu22_verdict(result)

    print("\nCU22 / H115 -- the generative test (availability danger, worst-case omitter):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}, "
          f"free breach {v['free_breach_rate']:.3f}\n")
    print("  candidate targets (covers = the a-priori prediction, before any deployment runs):")
    for c in result.candidates:
        print(f"    {c.label:42s}  covers={c.covers!s:5s}  "
              f"random {c.random_breach:.3f}  adversarial {c.adversarial_breach:.3f}  "
              f"{c.mean_calls:.2f} calls")
    print(f"\n  derived disconnect-closure: safe={v['closure_is_safe']}  "
          f"un-gameable={v['closure_is_ungameable']}  "
          f"= {v['closure_call_saving']:.1f}x cheaper than full oracle "
          f"({v['full_oracle_calls']:.0f} calls)")
    print(f"  carried-over catalogue (CU10 + CU17): break coverage="
          f"{v['carried_over_all_break_coverage']}  all leak={v['carried_over_all_leak']}")
    print(f"  uniform knee rho={v['uniform_knee_rho']:g}: random "
          f"{v['uniform_knee_random_breach']:.3f} -> adversarial "
          f"{v['uniform_knee_adversarial_breach']:.3f} (gameable={v['uniform_is_gameable']})")
    print(f"  model self-targeting fails={v['model_self_targeting_fails']}  "
          f"perfect model self-governs={v['oracle_self_governs']}")
    print(f"\n  GENERATIVE HEADLINE (H115): covers predicted every candidate's fate; confirmed = "
          f"{v['framework_predicts_every_candidate']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu22 import plot_cu22

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu22(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
