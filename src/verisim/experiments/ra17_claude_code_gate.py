"""Experiment RA17 -- the oracle coverage gate as a Claude Code permission hook (H149).

The paper's §8 product reading, made concrete and measured: a PreToolUse hook that returns the
coverage gate's verdict as the permission decision (off-surface allow, on-surface ask), compared
on a realistic Claude Code tool-call battery against the pattern-denylist status quo. The finding:
denylist both over-prompts on benign work (a benign `rm` under the work dir) AND leaks the chmod/mv
classes it did not enumerate; the coverage gate has zero benign approval fatigue AND catches every
explicit realizing action (complete mediation), prompting only on the sparse covering surface. The
honest edge (RA4) is reported: a Bash path built by indirection evades the syntactic target.

The runnable hook is `scripts/claude_code_coverage_hook.py`. Hermetic, torch-free, deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.claude_code_gate import (
    cu_ra17_verdict,
    run_gate_eval,
    write_csv,
)

_LABELS = {
    "allow_all": "no gate (allow all)",
    "permission_denylist": "permission denylist (Claude-Code status quo)",
    "oracle_coverage": "oracle coverage hook (verisim)",
}


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA17 -- the coverage gate as a Claude Code hook.")
    parser.add_argument("--out", type=str, default="figures/ra17_claude_code_gate.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    results = run_gate_eval()
    v = cu_ra17_verdict(results)

    print("\nRA17 / H149 -- the oracle coverage gate as a Claude Code permission hook (paper §8)\n")
    print(f"    {'arm':46s} {'benign_ask':>11s} {'miss_expl':>10s} {'ask_all':>8s}")
    for r in results:
        print(f"    {_LABELS[r.arm]:46s} {r.benign_prompt_rate:11.2f} "
              f"{r.missed_harm_explicit:10.2f} {r.overall_prompt_rate:8.2f}")

    print("\n  the §8 operating point, measured:")
    print(f"    coverage gate: no benign approval fatigue = {v['coverage_no_benign_fatigue']}, "
          f"catches every explicit harm = {v['coverage_catches_all_explicit']}")
    print(f"    status-quo denylist leaks the chmod/mv classes = {v['denylist_leaks_explicit']} "
          f"(missed {v['denylist_missed_harm']:.2f} of explicit harms)")
    print(f"    human-approval burden = the surface density {v['coverage_prompt_rate']:.2f} "
          f"(not every action)")
    print(f"    honest edge (RA4): indirection evades the syntactic target = "
          f"{v['coverage_indirection_edge']:.2f}")

    out = write_csv(results, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra17 import plot_ra17

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra17(results, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
