"""Plot the ED26 figure (SPEC-7 §5.1, DS0 incr 19): Raft log replication.

Two panels from one :class:`~verisim.experiments.ed26.ED26Result`:

  - **left — commit requires a majority.** A majority-reachable `append` commits (the commit index
    grows, 1.0); a minority-stranded leader's append stays uncommitted (commit index unchanged, 1.0)
    yet is retained on its log (1.0); the commit index never moves backward (1.0).
  - **right — log-matching reconciliation.** While uncommitted, the stale entry is never applied to
    the KV (1.0); after a higher-term leader commits a conflicting entry and the partition heals,
    the deposed leader's entry is overwritten (1.0), all live logs are identical (1.0), the rejoined
    node's KV converges to the committed value (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed26 import ED26Result


def plot_ed26(result: ED26Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: commit requires a majority
    labels = ["majority\nappend commits", "minority\nuncommitted",
              "minority entry\nretained on log", "commit index\nmonotone"]
    rates = [
        result.majority_commit_rate,
        result.minority_uncommitted_rate,
        result.minority_entry_retained_rate,
        result.commit_index_monotone_rate,
    ]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd"]
    bars = ax_l.bar(labels, rates, color=colors, alpha=0.85, width=0.66)
    caps = [f"commit_index+ ({result.n_sizes})", "commit_index=", "on the log", "never backward"]
    for bar, rate, cap in zip(bars, rates, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_l.set_ylim(0, 1.4)
    ax_l.set_ylabel("rate over the cluster-size sweep")
    ax_l.set_title("Panel A — commit requires a majority\n"
                   "the commit index advances only on a reachable majority",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: log-matching reconciliation
    labels = ["uncommitted\nnot applied", "deposed entry\noverwritten",
              "logs identical\nafter heal", "rejoined KV\nconverges"]
    vals = [
        1.0 if result.uncommitted_not_applied else 0.0,
        1.0 if result.deposed_entry_overwritten else 0.0,
        1.0 if result.log_matching_after_heal else 0.0,
        1.0 if result.kv_reflects_committed_log else 0.0,
    ]
    colors = ["#1f77b4", "#2ca02c", "#2ca02c", "#9467bd"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.66)
    caps = ["KV = committed", "stale tail gone", "log-matching", "converges to d"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_r.set_ylim(0, 1.4)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — log-matching reconciliation\n"
                   "a deposed leader's uncommitted tail is overwritten",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED26 — Raft log replication: commit-on-majority + log-matching reconcile  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
