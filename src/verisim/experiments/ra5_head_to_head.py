"""Experiment RA5 -- the 4-arm head-to-head: the oracle gate vs the field's guardrails (H137).

Runs the four guardrail arms (no gate / learned model guardrail [the SafePred proxy] / permission
denylist [the Claude-Code proxy] / oracle coverage [verisim]) on a Terminal-Bench-shaped fixture and
reports safety (random + adversarial), utility, and cost. The centerpiece: under an adaptive
prompt-injection adversary the learned guardrail and the permission policy are both breached, while
only the exact-oracle coverage gate is un-gameable -- at no utility loss, cheaply.

The hermetic engine here proves the comparison logic; the external number (a real Claude agent on
real Terminal-Bench tasks in Docker) comes from the live lane -- see the runbook in
docs/specs/SPEC-22.md (RA5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.head_to_head import (
    HeadToHeadResult,
    cu_ra5_verdict,
    run_head_to_head,
    write_csv,
)

_LABELS = {
    "none": "no gate (bare agent)",
    "model_guardrail": "learned guardrail (SafePred proxy)",
    "permission": "permission denylist (Claude-Code proxy)",
    "oracle_coverage": "oracle coverage (verisim)",
}


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA5 -- the 4-arm head-to-head (H137).")
    parser.add_argument("--out", type=str, default="figures/ra5_head_to_head.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    result: HeadToHeadResult = run_head_to_head()
    v = cu_ra5_verdict(result)

    print("\nRA5 / H137 -- the 4-arm head-to-head (the oracle gate vs the field's guardrails)")
    print(f"  {result.n_tasks} Terminal-Bench-shaped tasks, guardrail fidelity phi={result.phi}\n")
    print(f"    {'arm':38s} {'rand':>6s} {'ADV':>6s} {'util':>6s} {'cost':>6s}")
    for a in result.arms:
        print(f"    {_LABELS[a.arm]:38s} {a.random_missed_harm:6.2f} "
              f"{a.adversarial_missed_harm:6.2f} {a.task_success:6.2f} {a.mean_consult_cost:6.2f}")

    print("\n  under an ADAPTIVE adversary:")
    print(f"    learned guardrail (SafePred) breached = {v['learned_guardrail_breached']} "
          f"(adv {v['model_guardrail_adversarial']:.2f})")
    print(f"    permission denylist (Claude-Code) breached = {v['permission_breached']} "
          f"(adv {v['permission_adversarial']:.2f})")
    print(f"    oracle coverage un-gameable = {v['oracle_is_ungameable']} "
          f"(adv {v['oracle_coverage_adversarial']:.2f}) at no utility loss = "
          f"{v['oracle_no_utility_loss']}, {v['oracle_cost']:.1f} oracle calls/task")
    print(f"    a better model is no safer (CU4) = {v['better_model_no_safer']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra5 import plot_ra5

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra5(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
