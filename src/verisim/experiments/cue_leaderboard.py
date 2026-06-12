"""Experiment CL1 -- the verisim-cue leaderboard is discriminative (SPEC-21 §8.2, H91).

The artifact-validity result the computer-use benchmark needs before anyone trusts a scorecard: does
verisim-cue *stably rank* computer-use world-models by faithfulness? It is the SPEC-18 H65 parallel
([`bench.leaderboard`](../bench/leaderboard.py)) for the computer-use vertical SPEC-21 §8.2 names --
"packaged on the SPEC-18 verisim-bench line".

The run ([`cue.leaderboard`](../cue/leaderboard.py)): score a controlled fidelity ladder (the
trained arm deferred, the LP7 rule) through the ordered structure->content cue suite, by *recall* of
each
task's keyed set; rank by mean catch; and decide discriminative validity the strict way -- Kendall
tau between disjoint seed-split leaderboards (rank stability, CI-positive) **and** every adjacent
fidelity tier resolved above its paired across-split noise. Writes the committed leaderboard CSV +
the figure. CPU-only, torch-free, seconds; the bankable negative (a non-discriminative scorecard) is
first-class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.cue.leaderboard import (
    CueLeaderboardConfig,
    build_cue_leaderboard,
    write_csv,
)
from verisim.cue.pack import CueManifest


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CL1 -- the verisim-cue leaderboard + its discriminative validity (H91)."
    )
    parser.add_argument("--out", type=str, default="figures/cl1_cue_leaderboard.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CueLeaderboardConfig.smoke() if args.smoke else CueLeaderboardConfig()
    rows, stability = build_cue_leaderboard(config)
    tag = CueManifest().version_tag()

    print(f"verisim-cue leaderboard ({tag}) -- {len(config.seeds)} seeds, "
          f"horizon {config.horizon}, recall over the keyed set:")
    print(f"  {'rank':>4} {'tier':>13} {'alpha':>6} {'mean_catch':>11}   per-task (recall)")
    for i, r in enumerate(rows, start=1):
        cells = "  ".join(
            f"{t.split('-')[0][:5]}={r.per_task[t]:.2f}" for t in sorted(r.per_task)
        )
        print(f"  {i:>4} {r.tier:>13} {r.alpha:>6.2f} {r.mean_catch:>11.4f}   {cells}")

    print("\nH91 -- discriminative validity (does the scorecard stably rank by faithfulness?):")
    print(f"  Kendall tau (disjoint seed splits): {stability.tau_mean:+.3f} "
          f"[{stability.tau_lo:+.3f}, {stability.tau_hi:+.3f}]")
    print(f"  binding adjacent gap {stability.min_adjacent_gap:.4f} vs 2x paired noise "
          f"{2 * stability.max_seed_noise:.4f}")
    verdict = "DISCRIMINATIVE" if stability.discriminative else "NOT discriminative"
    print(f"  verdict: {verdict} -- the cue scorecard "
          f"{'stably ranks' if stability.discriminative else 'does NOT stably rank'} "
          f"computer-use world-models by faithfulness")

    out = write_csv(rows, args.out, tag)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cl1 import plot_cl1

        plot_path = Path(args.plot) if args.plot else out.with_suffix(".png")
        plot_cl1(rows, stability, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
