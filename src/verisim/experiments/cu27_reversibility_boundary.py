"""Experiment CU27 -- the reversibility boundary: *when* to verify, not *what* (H120).

The targeting arc (CU10-CU26) answers WHAT to verify. CU27 opens the orthogonal axis -- WHEN must
the verification preview happen at all? A reversible danger (a segmentation posture you can
re-segment) is safe **model-free** via verify-after-commit: execute, observe the realized state
(free), roll back on a breach -- zero oracle previews, un-gameable. An irreversible danger (an exfil
*send* that already left the boundary) has no safe after-commit policy, so it needs verify-before-
commit (the oracle/targeting preview). Route by reversibility: the costly trustworthy preview is
load-bearing only on the irreversible slice, and the price of it is exactly the irreversibility you
face. Torch-free; ~1s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.reversibility_boundary import (
    CU27Config,
    cu27_verdict,
    run_cu27,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU27 -- the reversibility boundary (H120).")
    parser.add_argument("--out", type=str, default="figures/cu27_reversibility_boundary.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU27Config.smoke() if args.smoke else CU27Config()
    result = run_cu27(config)
    v = cu27_verdict(result)

    print("\nCU27 / H120 -- the reversibility boundary (verify-after vs verify-before commit):")
    print(f"  {result.n_reversible} reversible (exposure) + {result.n_irreversible} irreversible "
          f"(exfil) deployments, horizon {result.horizon}\n")
    print(f"  {'policy':22s} {'class':13s} {'rand':>6s} {'adv':>6s} {'oracle':>7s} {'rolls':>6s}")
    for c in result.cells:
        print(f"  {c.policy:22s} {c.danger_class:13s} {c.random_breach:6.3f} "
              f"{c.adversarial_breach:6.3f} {c.mean_oracle_calls:7.2f} {c.mean_rollbacks:6.2f}")

    print("\n  REVERSIBLE class: after-commit is FREE + model-free + un-gameable "
          f"(safe={v['after_commit_reversible_safe']}, "
          f"zero oracle previews={v['after_commit_reversible_is_free']})")
    print("  IRREVERSIBLE class: after-commit FAILS -- a send cannot be rolled back "
          f"(adversarial breach {v['after_commit_irreversible_adv_breach']:.3f})")
    print(f"  ROUTED: safe everywhere={v['routed_safe_everywhere']}, reversible half "
          f"free={v['routed_reversible_is_free']}, before-commit oracle "
          f"{v['routed_oracle_calls']:.2f} calls = {v['routed_call_saving']:.1f}x cheaper than "
          f"verify-everything ({v['verify_all_oracle_calls']:.0f})")

    print("\n  COST LAW (the price of trustworthy world-modeling = the irreversibility fraction):")
    for p in result.cost_law:
        print(f"    f={p.irreversible_fraction:.2f}:  routed before-commit "
              f"{p.routed_oracle_calls:5.2f} calls   after-commit residual breach "
              f"{p.after_commit_breach:.3f}   verify-all {p.verify_all_oracle_calls:.0f}")
    print(f"    price tracks irreversibility={v['price_is_irreversibility']}  "
          f"after-commit breach tracks irreversibility="
          f"{v['after_commit_breach_tracks_irreversibility']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu27 import plot_cu27

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu27(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
