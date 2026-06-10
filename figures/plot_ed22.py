"""Plot the ED22 figure (SPEC-7 §4, DS0 incr 15): pairwise gossip.

Two panels from one :class:`~verisim.experiments.ed22.ED22Result`:

  - **left — one pairwise gossip reconciles BOTH endpoints; one anti-entropy fixes only one.** With
    `a` and `b` holding complementary holes, a single `gossip a b` fills both (2 endpoints synced)
    while a single `anti_entropy a` fills only `a`'s (1 endpoint) — the bidirectional vs
    one-directional distinction.
  - **right — a chain of pairwise gossips converges the reachable component (epidemic).** A write
    dropped to every peer spreads to the whole reachable chain (rate 1.0); a node partitioned off
    the chain stays stale (0.0) — gossip is bounded by reachability.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed22 import ED22Result


def plot_ed22(result: ED22Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: endpoints reconciled by ONE op — gossip (both) vs anti_entropy (one)
    labels = ["gossip a b\n(pairwise)", "anti_entropy a\n(pull-to-one)"]
    reconciled = [2 if result.gossip_reconciles_both else 0,
                  1 if result.anti_entropy_reconciles_one else 0]
    colors = ["#2ca02c", "#ff7f0e"]
    bars = ax_l.bar(labels, reconciled, color=colors, alpha=0.85, width=0.6)
    captions = ["both endpoints", "only the named node"]
    for bar, n, cap in zip(bars, reconciled, captions, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, n + 0.04,
                  f"{n}/2\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 2.5)
    ax_l.set_ylabel("endpoints reconciled by one op")
    ax_l.set_title("Panel A — pairwise gossip is bidirectional\n"
                   "one gossip syncs both nodes; one anti-entropy syncs one",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: epidemic convergence — reachable chain vs a partitioned node
    labels = ["reachable\nchain", "partitioned\nnode"]
    rates = [result.epidemic_converged_rate, 0.0 if result.partition_blocks_epidemic else 1.0]
    colors = ["#2ca02c", "#d62728"]
    bars = ax_r.bar(labels, rates, color=colors, alpha=0.85, width=0.6)
    captions = [f"converged ({result.n_chain} nodes)", "stays stale (cut off)"]
    for bar, rate, cap in zip(bars, rates, captions, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.2)
    ax_r.set_ylabel("convergence rate via a gossip chain")
    ax_r.set_title("Panel B — a gossip chain converges the component (epidemic)\n"
                   "spreads hop by hop, bounded by reachability",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED22 — pairwise gossip: bidirectional anti-entropy + epidemic convergence  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
