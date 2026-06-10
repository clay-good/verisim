"""Plot the ED27 figure (SPEC-7 §3.2, DS0 incr 20): membership change.

Two panels from one :class:`~verisim.experiments.ed27.ED27Result`:

  - **left — the quorum threshold tracks the membership.** A lone leader is blocked at full
    membership (1.0); after `remove_replica` makes it the sole member it commits (1.0); an
    `add_replica` raises the threshold and re-blocks it (1.0) — the threshold moves both ways.
  - **right — restore availability after node failure.** One live of three is stuck (`no_quorum`,
    1.0); after `remove_replica` the two dead nodes the survivor commits again (1.0); and the active
    leader cannot be removed (`is_leader`, 1.0) — the safety fence on reconfiguration.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed27 import ED27Result


def plot_ed27(result: ED27Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: the quorum threshold tracks the membership
    labels = ["lone leader\nblocked (full)", "sole member\ncommits", "re-add\nre-blocks"]
    rates = [
        result.alone_blocked_at_full_rate,
        result.sole_member_commits_rate,
        result.regrow_reblocks_rate,
    ]
    colors = ["#d62728", "#2ca02c", "#d62728"]
    bars = ax_l.bar(labels, rates, color=colors, alpha=0.85, width=0.62)
    caps = [f"no_quorum ({result.n_sizes})", "ok (majority of 1)", "no_quorum (threshold↑)"]
    for bar, rate, cap in zip(bars, rates, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_l.set_ylim(0, 1.35)
    ax_l.set_ylabel("rate over the cluster-size sweep")
    ax_l.set_title("Panel A — the quorum threshold tracks the membership\n"
                   "shrink the voting set and a minority becomes a majority",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: restore availability after node failure
    labels = ["1 live of 3\n(before)", "1 of 1\n(after remove)", "remove active\nleader"]
    vals = [
        1.0 if result.stuck_before_removal else 0.0,
        1.0 if result.restored_after_removal else 0.0,
        1.0 if result.active_leader_remove_blocked else 0.0,
    ]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    caps = ["no_quorum\n(stuck)", "ok\n(restored)", "is_leader\n(fenced)"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_r.set_ylim(0, 1.35)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — restore availability after node failure\n"
                   "remove the dead to shrink the quorum; the leader is fenced",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED27 — membership change: the quorum threshold tracks the voting set  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
