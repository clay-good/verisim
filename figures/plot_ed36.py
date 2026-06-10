"""Plot the ED36 figure (SPEC-7 §3.2, DS0 incr 29): the CRDT PN-counter.

Two panels from one :class:`~verisim.experiments.ed36.ED36Result`:

  - **left — decrement works, loss-free, may go negative (the PN extension).** k cincrs then m
    cdecrs net to k-m (1.0); a fresh cdecr reads -1, the value below zero a G-counter cannot reach
    (1.0); a partitioned-minority cdecr is acknowledged (AP, 1.0); +2 (majority) and -1 (minority)
    across a partition net 1 after heal+gossip — no lost update across both halves (1.0).
  - **right — convergence (the CRDT join over both halves).** a gossip chain converges every node to
    the net total (1.0); anti_entropy on each node reaches the same net (1.0); the join is
    idempotent (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed36 import ED36Result


def plot_ed36(result: ED36Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: decrement works, loss-free, may go negative
    labels = [f"nets to k-m={result.k - result.m}", "goes negative\n(-1)",
              "always available\n(AP)", "no lost update\n(net 1)"]
    vals = [
        result.net_correct_rate,
        1.0 if result.goes_negative else 0.0,
        1.0 if result.always_available else 0.0,
        1.0 if result.no_lost_update else 0.0,
    ]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#2ca02c"]
    caps = [f"exact ({result.n_sizes})", "below zero", "minority counts", "+2 - 1"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=8.5)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — decrement works, loss-free, may go negative\n"
                   "the PN-counter: a decrementable CRDT (extends ED35's G-counter)",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: convergence (the CRDT join over both halves)
    labels = ["gossip\nconverges", "anti_entropy\nconverges", "idempotent"]
    vals = [
        1.0 if result.gossip_converges else 0.0,
        1.0 if result.anti_entropy_converges else 0.0,
        1.0 if result.idempotent else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#9467bd"]
    caps = ["all nodes -> 1", "all nodes -> 1", "2nd gossip stable"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — convergence (the CRDT join over both halves)\n"
                   "per-(key, owner) max of P and N: commutative, idempotent, order-free",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED36 — the CRDT PN-counter: decrementable, loss-free, convergent"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
