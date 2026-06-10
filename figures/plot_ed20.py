"""Plot the ED20 figure (SPEC-7 §3.2/§3.4, DS0 incr 13): message-timing faults.

Two panels from one :class:`~verisim.experiments.ed20.ED20Result`:

  - **left — delay is recoverable; drop is not.** The post-``advance`` convergence rate when a
    write is cut off from a peer, per medium. ``delay`` defers the replication message (the peer is
    stale until the clock passes the deferral, then it arrives — rate 1.0, recoverable); ``drop``
    destroys it (rate 0.0, unrecoverable) — same symptom, two media, completing ED18's contrast.
  - **right — reorder flips the transit, not the converged value.** With two writes staggered so
    the newer is scheduled first, a peer transiently shows the newer value; after ``reorder`` it
    transiently shows the older one (the in-transit observation flips, rate 1.0) — yet both converge
    to the newer write (last-writer-wins is a commutative join, invariance rate 1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed20 import ED20Result

_COLOR = {"delay": "#1f77b4", "drop": "#d62728"}


def plot_ed20(result: ED20Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: convergence rate after advance, per medium (delay vs drop)
    regimes = [r["regime"] for r in result.convergence]
    rates = [r["rate"] for r in result.convergence]
    bars = ax_l.bar(regimes, rates, color=[_COLOR[m] for m in regimes], alpha=0.85, width=0.6)
    for bar, r in zip(bars, result.convergence, strict=True):
        verdict = "converges" if r["rate"] > 0 else "STAYS STALE"
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["rate"] + 0.02,
                  f"{r['rate']:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.2)
    ax_l.set_ylabel("convergence rate after advance\n(peer reaches the written value)")
    ax_l.set_title("Panel A — delay is recoverable; drop is not\n"
                   "delay defers the message (delivered once the clock passes it); drop "
                   "destroys it", fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: reorder — the transit flips, the converged value does not
    labels = ["transit\nobservation", "converged\nvalue"]
    measured = [result.reorder_transit_flip_rate, result.reorder_converged_invariant_rate]
    colors = ["#ff7f0e", "#2ca02c"]
    bars = ax_r.bar(labels, measured, color=colors, alpha=0.85, width=0.6)
    captions = ["flips under reorder", "invariant (LWW)"]
    for bar, rate, cap in zip(bars, measured, captions, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.2)
    ax_r.set_ylabel("rate over peers")
    ax_r.set_title("Panel B — reorder flips the transit, never the converged value\n"
                   "delivery order changes what you catch in flight; LWW lands the same value",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED20 — message timing (delay / reorder): recoverable delay + reorder-invariant "
                 f"convergence  •  Tier-B agrees = {result.tier_b_agrees} "
                 f"({result.tier_b_steps} steps)", y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
