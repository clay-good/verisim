"""Experiment CU38 -- the heterogeneous verifier ensemble: the dual of CU24's composite (H131).

CU24 proved the target-side composition theorem (the union target covers the union danger) with a
perfect verifier; CU35/CU36 then showed no single cheap monitor a defender can run is faithful on
the whole CIA triad. CU38 composes them: it holds the union target fixed (covering -- every leg is
consulted) and varies the VERIFIER from a single partial monitor to an ENSEMBLE that OR-combines a
panel of cheap monitors. The result is the deployment-grade conclusion of the verifier sub-arc -- no
single monitor is safe on the composite (each blind on >= 1 leg, the danger consulted-but-waved-
through), the ensemble whose members' faithful surfaces jointly tile CIA is exactly as safe as a
perfect oracle (adversarial breach 0.000, == the exact oracle), and dropping a member re-opens
exactly the leg it was the only one faithful on. You assemble cheap monitors to tile the danger
surface instead of building a perfect oracle. Torch-free (worst-case omitter + exact host oracle).
"""

from __future__ import annotations

import argparse

from verisim.acd.footprintless_targeting import CU34Config
from verisim.acd.verifier_ensemble import (
    _LEGS,
    _PANEL,
    _cell,
    cu38_verdict,
    run_cu38,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU38 -- the heterogeneous verifier ensemble (H131)."
    )
    parser.add_argument("--out", type=str, default="figures/cu38_verifier_ensemble.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU34Config.smoke() if args.smoke else CU34Config()
    result = run_cu38(config)
    v = cu38_verdict(result)

    print("\nCU38 / H131 -- the heterogeneous verifier ensemble (the dual of CU24's composite):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}\n")
    scopes = (*_LEGS, "composite")
    print("  adversarial breach by verifier x scope (0.000 = as safe as the perfect oracle):")
    print(f"    {'verifier':22s} " + " ".join(f"{s[:7]:>8s}" for s in scopes))
    for name in _PANEL:
        row = " ".join(f"{_cell(result, name, s).adversarial_breach:8.3f}" for s in scopes)
        print(f"    {name:22s} {row}")

    print(f"\n  no single partial monitor is safe on the composite = "
          f"{v['no_single_partial_safe_composite']}")
    print(f"  state-diff leaks ONLY confidentiality (footprintless) = "
          f"{v['state_diff_leaks_only_confidentiality']}")
    print(f"  ENSEMBLE {{state-diff, read-audit}} safe on composite, == oracle = "
          f"{v['ensemble_safe_composite']} / {v['ensemble_matches_exact_composite']}")
    print(f"  drop read-audit -> ONLY confidentiality re-opens = "
          f"{v['drop_read_reopens_only_confidentiality']}")
    print(f"  drop state-diff -> integrity + availability re-open = "
          f"{v['drop_statediff_reopens_integrity_availability']}")
    print(f"  composition theorem (ensemble faithful iff a member is) = {v['composition_theorem']}")
    print(f"  structural predictor exact (faithful iff channel observed) = "
          f"{v['structural_predictor_exact']}")
    print(f"  union target covers every scope (CU24 condition held) = {v['all_targets_cover']}")

    out = write_csv(result, args.out)
    print(f"\n  wrote {out}")

    if args.plot:
        from figures.plot_cu38 import plot_cu38

        path = plot_cu38(result, args.plot)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
