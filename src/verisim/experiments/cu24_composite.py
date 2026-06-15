"""Experiment CU24 -- the composite defense: defending the whole threat model at once (H117).

Every targeting milestone (CU10-CU23) defends one danger. CU24 asks the defender's real question:
can the unified target defend several at once? It builds three coexisting network dangers on CU22's
provisioned-work battery -- exfil to a crown jewel (confidentiality), a jewel exposed to the
untrusted set (segmentation), a work service disconnected (availability) -- and runs every point /
partial / union schedule against the composite threat model. The composition theorem (the union of
covering targets covers the union danger) predicts the union is safe + un-gameable at the union of
the surfaces; the boundary predicts every partial leaks exactly its omitted leg, and ``covers``
calls all of it a priori. Torch-free; worst-case-omitter substrate; ~4 min (full), seconds (smoke).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.composite_targeting import (
    CU24Config,
    cu24_verdict,
    run_cu24,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU24 -- the composite defense (H117).")
    parser.add_argument("--out", type=str, default="figures/cu24_composite_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU24Config.smoke() if args.smoke else CU24Config()
    result = run_cu24(config)
    v = cu24_verdict(result)

    print("\nCU24 / H117 -- the composite defense (whole threat model, worst-case omitter):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}, "
          f"free composite breach {v['free_breach_rate']:.3f}\n")
    print("  the defender's coverage matrix (adversarial breach per leg; covers = a-priori):")
    print(f"    {'schedule':32s} {'covers':6s} {'exfil':>6s} {'expos':>6s} {'outage':>6s} "
          f"{'calls':>6s}")
    for c in result.candidates:
        pl = c.per_leg_adversarial
        print(f"    {c.label:32s} {c.covers_composite!s:6s} "
              f"{pl['exfil']:6.2f} {pl['exposure']:6.2f} {pl['outage']:6.2f} {c.mean_calls:6.2f}")
    print(f"\n  COMPOSITION HEADLINE (H117): union target safe on every leg="
          f"{v['composite_safe_on_every_leg']}  covers={v['composite_covers']}  "
          f"= {v['composite_call_saving']:.1f}x cheaper than full oracle "
          f"({v['full_oracle_calls']:.0f} calls)")
    print(f"  union surface = the union of per-leg surfaces {v['single_leg_calls']} "
          f"(sum {v['sum_single_leg_calls']:.2f}); "
          f"subadditive={v['composite_surface_subadditive']}")
    print(f"  BOUNDARY: every partial breaks coverage={v['all_partials_break_coverage']}  "
          f"leaks exactly its omitted leg={v['partial_leaks_exactly_omitted_leg']}")
    print(f"  the most-quoted point defense (exfil): safe on its own leg="
          f"{v['exfil_only_safe_on_own_leg']}  gameable on the composite="
          f"{v['exfil_only_gameable_on_composite']}")
    print(f"  uniform knee rho={v['uniform_knee_rho']:g}: gameable={v['uniform_is_gameable']}  "
          f"model self-targeting fails={v['model_self_targeting_fails']}  "
          f"perfect model self-governs={v['oracle_self_governs']}")
    print(f"\n  covers predicts every candidate's fate (generative on composition) = "
          f"{v['covers_predicts_every_candidate']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu24 import plot_cu24

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu24(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
