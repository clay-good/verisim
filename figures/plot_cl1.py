"""Plot the CL1 figure (SPEC-21 §8.2, H91): the verisim-cue leaderboard is discriminative.

Two panels from a :func:`~verisim.cue.leaderboard.build_cue_leaderboard` result:

  - **left -- the leaderboard.** Mean catch (recall over the keyed set) per fidelity tier, ordered,
    with the deep-content task (``content-value``) overlaid: structure tasks saturate at 1.0 for
    every tier (the oracle is never load-bearing there) while content recall climbs with capacity --
    the structure->content gradient *is* what the leaderboard ranks on.
  - **right -- the discrimination.** Each adjacent-tier pair's paired gap vs its 2x across-split
    noise band: every gap clears its noise (the strict H91 test -- the benchmark resolves adjacent
    fidelity tiers, not just top-beats-floor), and the Kendall tau between disjoint seed splits is
    in the title (rank stability).
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from verisim.cue.leaderboard import CueLeaderRow, CueRankStability

_TIER_COLOR = {
    "floor": "#9e9e9e",
    "learned-lo": "#fdae6b",
    "learned-mid": "#fd8d3c",
    "learned-hi": "#e6550d",
    "ceiling": "#1f77b4",
}


def plot_cl1(
    rows: list[CueLeaderRow], stability: CueRankStability, path: str | Path
) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))

    # left: the leaderboard (best at top), with the content-value recall overlay
    ordered = sorted(rows, key=lambda r: r.mean_catch)  # ascending -> best on top in barh
    y = range(len(ordered))
    colors = [_TIER_COLOR.get(r.tier, "#333") for r in ordered]
    ax_l.barh(list(y), [r.mean_catch for r in ordered], color=colors, alpha=0.9)
    content = [r.per_task.get("content-value", 0.0) for r in ordered]
    ax_l.plot(content, list(y), "D", color="#d62728", mfc="none", ms=8,
              label="content-value recall (the discriminating task)")
    ax_l.set_yticks(list(y))
    ax_l.set_yticklabels([f"{r.tier}\nα={r.alpha:g}" for r in ordered], fontsize=8)
    ax_l.set_xlabel("mean catch (recall over the keyed set)")
    ax_l.set_title("the verisim-cue leaderboard (fidelity ladder)")
    ax_l.set_xlim(0, 1.05)
    for i, r in enumerate(ordered):
        ax_l.text(r.mean_catch + 0.01, i, f"{r.mean_catch:.3f}", va="center", fontsize=8)
    ax_l.legend(fontsize=8, loc="lower right")

    # right: each adjacent-tier gap vs its 2x paired-noise band (the strict discrimination test)
    asc = sorted(rows, key=lambda r: r.mean_catch)
    pairs = list(pairwise(asc))
    labels = [f"{b.tier}\n−{a.tier}" for a, b in pairs]
    gaps = [b.mean_catch - a.mean_catch for a, b in pairs]
    x = range(len(pairs))
    ax_r.bar(x, gaps, color="#2ca02c", alpha=0.85, label="adjacent-tier gap")
    ax_r.axhline(2 * stability.max_seed_noise, color="#d62728", ls="--",
                 label=f"2× binding noise = {2 * stability.max_seed_noise:.3f}")
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(labels, fontsize=7)
    ax_r.set_ylabel("paired catch gap between adjacent tiers")
    verdict = "DISCRIMINATIVE" if stability.discriminative else "NOT discriminative"
    ax_r.set_title(
        f"adjacent tiers resolve above noise — {verdict}\n"
        f"Kendall τ = {stability.tau_mean:+.3f} "
        f"[{stability.tau_lo:+.2f}, {stability.tau_hi:+.2f}]"
    )
    ax_r.legend(fontsize=8, loc="upper right")
    ax_r.set_ylim(bottom=0)

    fig.suptitle(
        "CL1 / H91: the verisim-cue scorecard stably ranks computer-use models by faithfulness"
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
