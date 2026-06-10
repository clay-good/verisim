"""Plot the ED32 figure (SPEC-7 §5.1, DS0 incr 25): the quorum-confirmed linearizable read.

Two panels from one :class:`~verisim.experiments.ed32.ED32Result`:

  - **left — the two linearizable reads, opposite availability.** read_index served at the leader
    (rate 1.0); a non-leader read_index is fenced (1.0); a minority leader is no_quorum (1.0); and
    the contrast — a minority leader with a live lease serves lread where read_index refuses (1.0).
  - **right — linearizable safety + freshness.** read_index returns the committed value (1.0); a
    deposed leader's read_index is fenced even after heal (1.0); and the safety contrast — a plain
    get serves the stale value where read_index refuses, while the new leader's read is fresh (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed32 import ED32Result


def plot_ed32(result: ED32Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: the two linearizable reads, opposite availability
    labels = ["read_index\nserved", "non-leader\nfenced", "minority\nno_quorum",
              "lease serves where\nquorum refuses"]
    vals = [
        result.read_index_ok_rate,
        1.0 if result.nonleader_fenced else 0.0,
        1.0 if result.minority_no_quorum else 0.0,
        1.0 if result.lease_serves_where_quorum_refuses else 0.0,
    ]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    caps = [f"ok ({result.n_sizes})", "not_leader", "no_quorum", "lread ok / RI no_quorum"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — the two linearizable reads, opposite availability\n"
                   "quorum read needs a majority; the lease read trades that for a clock",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: linearizable safety + freshness
    labels = ["reflects\ncommitted", "deposed leader\nfenced", "stale get vs\nsafe read_index"]
    vals = [
        1.0 if result.reflects_committed else 0.0,
        1.0 if result.deposed_read_index_fenced else 0.0,
        1.0 if result.stale_get_vs_safe_read_index else 0.0,
    ]
    colors = ["#2ca02c", "#9467bd", "#d62728"]
    caps = ["committed value", "not_leader after heal", "get stale / RI refuses"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — linearizable safety + freshness\n"
                   "read_index refuses the stale read a plain get would serve",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED32 — read_index: the quorum-confirmed linearizable read (Raft ReadIndex)"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
