"""Experiment CU34 -- the footprintless danger: host confidentiality completes the CIA triad (H127).

The targeting arc localized every danger on a model-free surface, but every danger so far MUTATED a
protected resource (a flow opens, a file changes, a process dies, a host becomes reachable) -- which
is exactly what an after-the-fact detector watches for. CU34 exhibits the danger with no
footprint: a host ``read`` of a secret discloses its content and leaves the state unchanged. So
an after-the-fact state-diff detector catches integrity and availability but is structurally blind
to confidentiality, and the danger must be verified BEFORE COMMIT on the read-to-secret-fd surface
(the dual of CU16's write-to-protected-fd). The covering target wins; the CU16 write target carried
over leaks; the host CIA union covers all three while dropping confidentiality leaks it. Torch-free
(worst-case omitter + exact host oracle); ~7s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.footprintless_targeting import (
    CU34Config,
    cu34_verdict,
    run_cu34,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU34 -- the footprintless danger (H127).")
    parser.add_argument("--out", type=str, default="figures/cu34_footprintless.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU34Config.smoke() if args.smoke else CU34Config()
    result = run_cu34(config)
    v = cu34_verdict(result)

    print("\nCU34 / H127 -- the footprintless danger (confidentiality completes the CIA triad):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}\n")

    print("  AFTER-THE-FACT DETECTION vs BEFORE-COMMIT TARGET, per CIA leg:")
    print(f"    {'leg':42s} {'detect':>7s} {'n':>5s} {'tgt_adv':>8s} {'calls':>6s}")
    for c in result.contrasts:
        print(f"    {c.name:42s} {c.after_the_fact_catch_rate:7.3f} {c.n_realizing:5d} "
              f"{c.before_commit_adversarial:8.3f} {c.target_calls:6.2f}")
    print(f"    after-the-fact detector blind to confidentiality ONLY = "
          f"{v['after_the_fact_blind_to_confidentiality_only']}")

    print("\n  THE CONFIDENTIALITY FRONTIER (the footprintless leg, read-to-secret-fd target):")
    arm = result.confidentiality
    print(f"    {'policy':36s} {'rand':>6s} {'adv':>6s} {'calls':>7s}")
    cells = [*arm.uniform, arm.model, arm.target, arm.full_oracle]
    if arm.shortcut is not None:
        cells.append(arm.shortcut)
    for cell in cells:
        print(f"    {cell.label:36s} {cell.random_breach:6.3f} {cell.adversarial_breach:6.3f} "
              f"{cell.mean_calls:7.2f}")
    print(f"    read-target safe+un-gameable={v['confidentiality_target_is_safe']}/"
          f"{v['confidentiality_target_is_ungameable']} at "
          f"{v['confidentiality_call_saving']:.1f}x cheaper than the full oracle; "
          f"write shortcut leaks={v['write_shortcut_leaks']} (covers={v['write_shortcut_covers']})")

    print("\n  THE HOST CIA TRIAD (union of the three model-free surfaces):")
    print(f"    union (all three) adversarial breach   = {result.union_breach:.3f} "
          f"(covers all three = {v['union_covers_all_three']})")
    print(f"    drop confidentiality -> adversarial    = {result.drop_confidentiality_breach:.3f} "
          f"(leaks the footprintless leg = {v['drop_confidentiality_leaks']})")
    print(f"    drop integrity (control) -> adversarial = {result.drop_integrity_breach:.3f}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu34 import plot_cu34

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu34(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
