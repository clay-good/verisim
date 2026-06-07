"""Plot the ED19 figure (SPEC-7 §5.1, DS0 incr 12): anti-entropy repairs what message loss broke.

Two panels from one :class:`~verisim.experiments.ed19.ED19Result`:

  - **left — anti-entropy repairs a dropped write; advance cannot.** After a write is dropped to a
    peer and the link heals, the post-repair convergence rate per mechanism. ``advance`` alone never
    recovers the stale replica (rate 0.0 — no in-flight message remains); ``anti_entropy`` pulls the
    latest value directly from the reachable replicas and repairs (rate 1.0). Read-repair converges
    what message loss broke, with no new write.
  - **right — anti-entropy is bounded by reachability.** The same repair op succeeds or fails on
    nothing but connectivity: with the stale peer partitioned away it cannot cross the split (rate
    0.0); once the partition heals it reaches every replica and repairs (rate 1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed19 import ED19Result

_COLOR = {"advance_only": "#d62728", "anti_entropy": "#2ca02c"}


def plot_ed19(result: ED19Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: convergence rate of a dropped-then-healed write, per repair mechanism
    regimes = [r["regime"] for r in result.convergence]
    rates = [r["rate"] for r in result.convergence]
    labels = {"advance_only": "advance only", "anti_entropy": "anti-entropy"}
    bars = ax_l.bar([labels[m] for m in regimes], rates,
                    color=[_COLOR[m] for m in regimes], alpha=0.85, width=0.6)
    for bar, r in zip(bars, result.convergence, strict=True):
        verdict = "repairs" if r["rate"] > 0 else "STAYS STALE"
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["rate"] + 0.02,
                  f"{r['rate']:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.2)
    ax_l.set_ylabel("convergence rate of the dropped write\n(peer reaches the written value)")
    ax_l.set_title("Panel A — anti-entropy repairs a dropped write; advance cannot\n"
                   "read-repair converges what message loss broke, with no new write",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: anti-entropy repair rate, bounded by reachability
    labels_b = ["partitioned\naway", "after\nheal"]
    rates_b = [result.bounded_rate, result.reachable_rate]
    colors_b = ["#d62728", "#2ca02c"]
    bars = ax_r.bar(labels_b, rates_b, color=colors_b, alpha=0.85, width=0.6)
    for bar, rate in zip(bars, rates_b, strict=True):
        verdict = "repairs" if rate > 0 else "cannot cross"
        ax_r.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.2)
    ax_r.set_ylabel("anti-entropy repair rate")
    ax_r.set_title("Panel B — anti-entropy is bounded by reachability\n"
                   "it converges only the reachable set — gossip, not magic", fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED19 — anti-entropy / read-repair: convergence restored after message loss  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
