"""Plot the ED31 figure (SPEC-7 §3.2, DS0 incr 24): the config push.

Two panels from one :class:`~verisim.experiments.ed31.ED31Result`:

  - **left — leader-committed rollout + the leader fence.** A leader push reaches every voting
    member (rate 1.0); a non-leader push is fenced (1.0); a no-leader push is rejected (1.0).
  - **right — the partition (will this config push break the cluster?).** A minority-stranded leader
    cannot commit (no_quorum, 1.0); a majority-side push commits but the minority keeps stale config
    (divergence, 1.0); a re-push after heal converges every node (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed31 import ED31Result


def plot_ed31(result: ED31Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: leader-committed rollout + the leader fence
    labels = ["commit reaches\nall voters", "non-leader\nfenced", "no-leader\nfenced"]
    vals = [
        result.commit_full_rate,
        1.0 if result.nonleader_fenced else 0.0,
        1.0 if result.noleader_fenced else 0.0,
    ]
    colors = ["#2ca02c", "#1f77b4", "#9467bd"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    caps = [f"committed ({result.n_sizes})", "not_leader", "not_leader"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — leader-committed rollout + the leader fence\n"
                   "config changes go through consensus, not any node",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: the partition
    labels = ["minority leader\nno_quorum", "majority commits,\nminority stale",
              "re-push\nconverges"]
    vals = [
        1.0 if result.minority_no_quorum else 0.0,
        1.0 if result.minority_stale_under_partition else 0.0,
        1.0 if result.repush_converges else 0.0,
    ]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    caps = ["cannot commit", "config divergence", "all nodes converge"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — the partition: will this config push break the cluster?\n"
                   "majority commits, the partitioned minority keeps stale config",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED31 — the config push: leader-committed, majority-replicated cluster config"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
