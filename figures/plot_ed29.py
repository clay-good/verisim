"""Plot the ED29 figure (SPEC-7 §3.2, DS0 incr 22): the rolling upgrade.

Two panels from one :class:`~verisim.experiments.ed29.ED29Result`:

  - **left — the safe rolling upgrade.** Rolling every node v0 → v1 one at a time, a propose commits
    at every intermediate step (rate 1.0): the version spread stays inside the compatibility window,
    so a compatible majority always exists.
  - **right — the deploy that breaks the cluster, and why.** An incompatible split (spread 2 > skew
    1, no compatible majority) loses quorum (`no_quorum`, 1.0); the same shape at spread 1 commits
    (1.0), and the same over-spread under a wider window (skew 2) commits (1.0) — it is the spread
    exceeding the window that breaks consensus, not mixed versions per se.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed29 import ED29Result


def plot_ed29(result: ED29Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: the safe rolling upgrade (single bar — commit rate over all upgrade steps)
    bars = ax_l.bar(["rolling upgrade\nv0 → v1"], [result.rolling_commit_rate],
                    color="#2ca02c", alpha=0.85, width=0.5)
    for bar in bars:
        ax_l.text(bar.get_x() + bar.get_width() / 2, result.rolling_commit_rate + 0.02,
                  f"{result.rolling_commit_rate:.2f}\npropose commits\n({result.n_steps} steps)",
                  ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.35)
    ax_l.set_ylabel("propose commit rate over the upgrade")
    ax_l.set_title("Panel A — the safe rolling upgrade\n"
                   "spread stays in the window → quorum maintained throughout",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: the break + the diagnostic contrast
    labels = ["incompatible split\n(spread 2, skew 1)", "within window\n(spread 1, skew 1)",
              "wider window\n(spread 2, skew 2)"]
    vals = [
        1.0 if result.incompatible_split_breaks else 0.0,
        1.0 if result.within_window_commits else 0.0,
        1.0 if result.wider_window_commits else 0.0,
    ]
    colors = ["#d62728", "#2ca02c", "#2ca02c"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    caps = ["no_quorum\n(broke)", "ok\n(safe)", "ok\n(safe)"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_r.set_ylim(0, 1.35)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — the deploy that breaks the cluster, and why\n"
                   "spread > window with no compatible majority loses quorum",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED29 — the rolling upgrade: will this deploy break the cluster?"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
