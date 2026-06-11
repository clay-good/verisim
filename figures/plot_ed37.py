"""Plot the ED37 figure (SPEC-7 §3.2, DS0 incr 30): the CRDT OR-Set.

Two panels from one :class:`~verisim.experiments.ed37.ED37Result`:

  - **left — the OR-Set's defining wins.** sadd k elements reads back all k (1.0); a removed element
    is re-addable (1.0); a concurrent add survives a concurrent remove — add-wins (1.0); a
    partitioned-minority sadd is acknowledged (AP, 1.0). The two boldface bars (re-addable,
    add-wins)
    are exactly what a naive 2P-Set gets wrong.
  - **right — convergence (the CRDT union join over both halves).** a gossip chain converges every
    node to the same set (1.0); anti_entropy on each node reaches the same set (1.0); the join is
    idempotent (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed37 import ED37Result


def plot_ed37(result: ED37Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: the OR-Set's defining wins
    labels = [f"reads back k={result.k}", "re-addable", "add-wins", "always available\n(AP)"]
    vals = [
        result.adds_read_rate,
        1.0 if result.re_addable else 0.0,
        1.0 if result.add_wins else 0.0,
        1.0 if result.always_available else 0.0,
    ]
    colors = ["#2ca02c", "#d62728", "#d62728", "#1f77b4"]
    caps = [f"exact ({result.n_sizes})", "vs 2P-Set", "vs 2P-Set", "minority adds"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=8.5)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — the OR-Set's defining wins\n"
                   "add-wins + re-addable: what a naive 2P-Set gets wrong",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: convergence (the union join over both halves)
    labels = ["gossip\nconverges", "anti_entropy\nconverges", "idempotent"]
    vals = [
        1.0 if result.gossip_converges else 0.0,
        1.0 if result.anti_entropy_converges else 0.0,
        1.0 if result.idempotent else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#9467bd"]
    caps = ["all nodes same", "all nodes same", "2nd gossip stable"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — convergence (the CRDT union join over both halves)\n"
                   "set union of adds and tombstones: commutative, idempotent, order-free",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED37 — the CRDT OR-Set: add-wins, re-addable, convergent"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
