"""Plot the ED41 figure (SPEC-7 §3.2, DS0 incr 34): the CRDT RGA sequence.

Two panels from one :class:`~verisim.experiments.ed41.ED41Result`:

  - **left — sequence ops + deterministic concurrent insert.** sequential rins builds "abc" (1.0); a
    middle insert and a delete both work (1.0); two nodes inserting different chars at the same
    position concurrently converge to the same string on every node — the RGA property (1.0);
    a partitioned-minority rins is acknowledged (AP, 1.0).
  - **right — convergence (the union join + order function).** a gossip chain converges every node
    to the same sequence (1.0); anti_entropy on each node (1.0); idempotent (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed41 import ED41Result


def plot_ed41(result: ED41Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: sequence ops + deterministic concurrent insert
    labels = ['build "abc"', "insert /\ndelete", "concurrent\nconverges", "always avail\n(AP)"]
    vals = [
        result.build_rate,
        1.0 if result.insert_delete else 0.0,
        1.0 if result.concurrent_converges else 0.0,
        1.0 if result.always_available else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#d62728", "#1f77b4"]
    caps = [f"exact ({result.n_sizes})", "aXbc -> abc", "same order", "minority edits"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=8.5)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — sequence ops + deterministic concurrent insert\n"
                   "concurrent inserts converge to ONE order (collaborative text)",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: convergence
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
    ax_r.set_title("Panel B — convergence (the union join + order function)\n"
                   "set-union of elements; order is a pure function of the set",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED41 — the CRDT RGA: the first ordered CRDT (a sequence / collaborative text)"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
