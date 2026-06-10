"""Plot the ED34 figure (SPEC-7 §3.2, DS0 incr 27): the atomic counter.

Two panels from one :class:`~verisim.experiments.ed34.ED34Result`:

  - **left — sequential correctness.** incr applied k times counts to exactly k (rate 1.0); the same
    sequence is correct under all three consistency models (1.0).
  - **right — the read-modify-write CAP tradeoff under partition.** eventual loses a concurrent
    increment (two acked, count short by one — the danger, 1.0); quorum makes the minority
    unavailable (no silent loss, 1.0); linearizable rejects the write under any partition (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed34 import ED34Result


def plot_ed34(result: ED34Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: sequential correctness
    labels = [f"counts to k={result.k}", "correct under\nall 3 models"]
    vals = [
        result.seq_correct_rate,
        1.0 if result.seq_correct_all_models else 0.0,
    ]
    colors = ["#2ca02c", "#1f77b4"]
    caps = [f"exact ({result.n_sizes})", "eventual/quorum/lin"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — sequential correctness\n"
                   "with no concurrency, every model counts right",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: the read-modify-write CAP tradeoff
    labels = ["eventual\nlost update", "quorum\nno silent loss", "linearizable\nunavailable"]
    vals = [
        1.0 if result.eventual_lost_update else 0.0,
        1.0 if result.quorum_no_silent_loss else 0.0,
        1.0 if result.linearizable_unavailable else 0.0,
    ]
    colors = ["#d62728", "#2ca02c", "#ff7f0e"]
    caps = ["2 acked, count -1", "minority rejected", "CP rejects"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — the read-modify-write CAP tradeoff\n"
                   "eventual loses a concurrent increment; quorum/lin refuse it",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED34 — the atomic counter: read-modify-write and the lost-update problem"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
