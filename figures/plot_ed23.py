"""Plot the ED23 figure (SPEC-7 §3.2, DS0 incr 16): leader election with terms.

Two panels from one :class:`~verisim.experiments.ed23.ED23Result`:

  - **left — no split-brain leadership.** Only a strict-majority partition side can elect: the
    minority side is blocked (rate 1.0) and the majority side succeeds (rate 1.0), and an even split
    leaves the cluster leaderless (neither side elects) — never two leaders.
  - **right — term-fencing, the property a leaderless quorum write lacks.** A deposed leader's
    `propose` is rejected after heal (fenced, 1.0), while a plain `put` by that same stale
    coordinator still commits (1.0) — the stale write the fence exists to stop; the legitimate new
    leader commits (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed23 import ED23Result


def plot_ed23(result: ED23Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: no split-brain — minority blocked, majority elects, even split leaderless
    labels = ["minority\nelect blocked", "majority\nelects", "even split\nleaderless"]
    rates = [
        result.minority_elect_blocked_rate,
        result.majority_elect_rate,
        result.even_split_leaderless_rate,
    ]
    colors = ["#2ca02c", "#1f77b4", "#9467bd"]
    bars = ax_l.bar(labels, rates, color=colors, alpha=0.85, width=0.62)
    caps = [
        f"no_quorum ({result.n_split_scenarios})",
        f"elected ({result.n_split_scenarios})",
        f"no leader ({result.n_even_scenarios})",
    ]
    for bar, rate, cap in zip(bars, rates, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.25)
    ax_l.set_ylabel("rate over the cluster-size sweep")
    ax_l.set_title("Panel A — no split-brain leadership\n"
                   "only a strict-majority side can elect a leader",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: term-fencing — deposed propose fenced vs the unfenced put control
    labels = ["deposed leader\npropose (fenced)", "same stale node\nplain put (control)",
              "new leader\npropose"]
    vals = [
        1.0 if result.deposed_propose_fenced else 0.0,
        1.0 if result.unfenced_put_commits else 0.0,
        1.0 if result.new_leader_commits else 0.0,
    ]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    caps = ["not_leader\nafter heal", "still commits\n(the stale write)", "ok\n(legitimate)"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.35)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — term-fencing (leader completeness)\n"
                   "a deposed leader cannot commit; a leaderless quorum put can",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED23 — leader election with terms: no split-brain + the fence quorum lacks  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
