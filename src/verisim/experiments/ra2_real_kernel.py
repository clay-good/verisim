"""Experiment RA2 -- the real-LLM safety gate against a real kernel (H134, anchor-invariance).

Runs the identical RA1 battery (the recorded Claude transcript, the held-out credential danger, the
prompt-injection) against the reference oracle AND a real ``/bin/sh``, and reports the max cell-wise
delta. A delta of 0 means the agent-safety verdict is verified against real computer-use dynamics.
The real-shell arm is SPEC-11 §2.5-disclosed: a missing shell is a first-class skip, not agreement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.real_kernel import (
    RA2Result,
    anchor_delta,
    ra2_verdict,
    run_ra2,
    write_csv,
)


def _print_table(result: RA2Result) -> None:
    print(f"    {'anchor':16s} {'schedule':14s} {'missed':>7s} {'success':>8s} {'calls':>7s}")
    rows = [("reference", result.ref)] + (
        [("real /bin/sh", result.sys)] if result.sys is not None else []
    )
    for anchor, res in rows:
        if res is None:
            continue
        for c in res.cells:
            print(f"    {anchor:16s} {c.schedule:14s} {c.missed_danger_rate:7.3f} "
                  f"{c.task_success_rate:8.3f} {c.mean_oracle_calls:7.2f}")


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA2 -- real-LLM gate vs a real kernel (H134).")
    parser.add_argument("--out", type=str, default="figures/ra2_real_kernel.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    result = run_ra2()
    v = ra2_verdict(result)

    print("\nRA2 / H134 -- the real-LLM safety gate against a real /bin/sh:")
    print(f"  platform={v['platform']}  real-shell available={v['available']}\n")
    _print_table(result)

    if result.sys is not None:
        print(f"\n  anchor delta (max |reference - real kernel| over all cells) = "
              f"{anchor_delta(result):.6f}")
        print(f"  ANCHOR-INVARIANT (bit-identical against a real kernel) = {v['anchor_invariant']}")
    else:
        print("\n  [real shell UNAVAILABLE -- skipped, not counted (SPEC-11 §2.5)]")

    print(f"\n  RA1 headline carried to the real kernel: undefended breaches = "
          f"{v['undefended_breaches']}, gate drives to zero = {v['gate_drives_to_zero']}, "
          f"no utility loss = {v['no_utility_loss']}, "
          f"{v['call_saving']:.1f}x cheaper than full oracle")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra2 import plot_ra2

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra2(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
