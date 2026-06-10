"""Plot the ED35 figure (SPEC-7 §3.2, DS0 incr 28): the CRDT G-counter.

Two panels from one :class:`~verisim.experiments.ed35.ED35Result`:

  - **left — loss-free and always available (the resolution to ED34).** cincr counts to k (1.0);
    a partitioned-minority cincr is acknowledged (AP, 1.0); three concurrent increments across a
    partition total 3 after heal+gossip — no lost update (1.0), where ED34's LWW counter read 2.
  - **right — convergence (the CRDT join).** a gossip chain converges every node to the full total
    (1.0); anti_entropy on each node reaches the same total (1.0); the join is idempotent (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed35 import ED35Result


def plot_ed35(result: ED35Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: loss-free and always available
    labels = [f"counts to k={result.k}", "always available\n(AP)", "no lost update\n(total 3)"]
    vals = [
        result.seq_correct_rate,
        1.0 if result.always_available else 0.0,
        1.0 if result.no_lost_update else 0.0,
    ]
    colors = ["#2ca02c", "#1f77b4", "#2ca02c"]
    caps = [f"exact ({result.n_sizes})", "minority counts", "vs ED34's 2"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — loss-free and always available\n"
                   "the CRDT resolution to ED34's lost update",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: convergence (the CRDT join)
    labels = ["gossip\nconverges", "anti_entropy\nconverges", "idempotent"]
    vals = [
        1.0 if result.gossip_converges else 0.0,
        1.0 if result.anti_entropy_converges else 0.0,
        1.0 if result.idempotent else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#9467bd"]
    caps = ["all nodes -> 3", "all nodes -> 3", "2nd gossip stable"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — convergence (the CRDT join)\n"
                   "per-(key, owner) max: commutative, idempotent, order-free",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED35 — the CRDT G-counter: loss-free, always-available, convergent"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
