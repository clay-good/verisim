"""Plot the ED28 figure (SPEC-7 §3.2, DS0 incr 21): the distributed FIFO queue.

Two panels from one :class:`~verisim.experiments.ed28.ED28Result`:

  - **left — delivery under partition.** Deliveries of one item when both partition sides dequeue:
    `eventual` = 2 (at-least-once / duplicate), `quorum` = 1 (exactly-once on the majority),
    `linearizable` = 0 (CP, both sides unavailable). The dashed line at 1 marks exactly-once.
  - **right — FIFO + exactly-once on the connected path.** With full connectivity, dequeue order
    equals enqueue order (1.0) and each item is delivered once, then the queue is empty (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed28 import ED28Result


def plot_ed28(result: ED28Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: delivery count of one item under partition, per consistency model
    labels = ["eventual", "quorum", "linearizable"]
    counts = [result.eventual_deliveries, result.quorum_deliveries, result.linearizable_deliveries]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    bars = ax_l.bar(labels, counts, color=colors, alpha=0.85, width=0.6)
    caps = ["at-least-once\n(duplicate)", "exactly-once\n(on majority)", "unavailable\n(CP)"]
    for bar, c, cap in zip(bars, counts, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, c + 0.04,
                  f"{c}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.axhline(1.0, ls="--", color="#555555", lw=1.0, alpha=0.7)
    ax_l.text(2.45, 1.02, "exactly-once", ha="right", va="bottom", fontsize=8, color="#555555")
    ax_l.set_ylim(0, 2.5)
    ax_l.set_ylabel("deliveries of one item (both sides dequeue)")
    ax_l.set_title("Panel A — delivery under partition\n"
                   "the same queue, three delivery semantics by model",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: FIFO + exactly-once on the connected path
    labels = ["FIFO order\npreserved", "exactly-once\nthen empty"]
    vals = [
        1.0 if result.fifo_preserved else 0.0,
        1.0 if result.exactly_once_connected else 0.0,
    ]
    bars = ax_r.bar(labels, vals, color=["#1f77b4", "#2ca02c"], alpha=0.85, width=0.5)
    caps = ["dequeue == enqueue", "each item once"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.35)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — FIFO + exactly-once on the connected path\n"
                   "a correct FIFO queue when the network is whole",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED28 — the distributed FIFO queue: delivery semantics follow the model"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
