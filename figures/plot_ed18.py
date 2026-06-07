"""Plot the ED18 figure (SPEC-7 §3.2, DS0 incr 11): message loss breaks convergence.

Two panels from one :class:`~verisim.experiments.ed18.ED18Result`:

  - **left — drop breaks convergence; partition does not.** The post-``heal``+``advance``
    convergence rate when a write is cut off from a peer, per medium. ``partition`` *holds* the
    replication message and delivers it on heal (rate 1.0, recoverable); ``drop`` *destroys* it, so
    the peer stays permanently stale (rate 0.0, unrecoverable) — same symptom, two media.
  - **right — only a newer write heals a dropped write.** After the drop, ``heal``+``advance`` alone
    never repairs the stale replica (rate 0.0); a subsequent write to the same key (a higher MVCC
    version) does (rate 1.0). The dropped value is never observed by the peer — a lost update at the
    network layer, recoverable only by being superseded.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed18 import ED18Result

_COLOR = {"partition": "#1f77b4", "drop": "#d62728"}


def plot_ed18(result: ED18Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: convergence rate after heal+advance, per medium
    regimes = [r["regime"] for r in result.convergence]
    rates = [r["rate"] for r in result.convergence]
    bars = ax_l.bar(regimes, rates, color=[_COLOR[m] for m in regimes], alpha=0.85, width=0.6)
    for bar, r in zip(bars, result.convergence, strict=True):
        verdict = "converges" if r["rate"] > 0 else "STAYS STALE"
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["rate"] + 0.02,
                  f"{r['rate']:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.2)
    ax_l.set_ylabel("convergence rate after heal+advance\n(peer reaches the written value)")
    ax_l.set_title("Panel A — drop breaks convergence; partition does not\n"
                   "partition holds the message (delivered on heal); drop destroys it",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: repairing a dropped write — heal-only vs a newer overwrite
    labels = ["heal only", "newer write"]
    repair = [result.drop_heal_only_rate, result.drop_overwrite_rate]
    colors = ["#d62728", "#2ca02c"]
    bars = ax_r.bar(labels, repair, color=colors, alpha=0.85, width=0.6)
    for bar, rate in zip(bars, repair, strict=True):
        verdict = "repairs" if rate > 0 else "NEVER repairs"
        ax_r.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.2)
    ax_r.set_ylabel("repair rate of the dropped write")
    ax_r.set_title("Panel B — only a newer write heals a dropped write\n"
                   f"lost value never observed by the peer = {result.lost_value_never_observed}",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED18 — message loss (drop): the broken-convergence anomaly  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
