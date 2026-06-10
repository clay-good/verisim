"""Plot the ED21 figure (SPEC-7 §3.2/§3.4, DS0 incr 14): clock skew.

Two panels from one :class:`~verisim.experiments.ed21.ED21Result`:

  - **left — skew shifts the send timing by exactly delta.** The `deliver_after` a node stamps on
    its replication message vs its clock offset: a straight line of slope 1 (every send shifted by
    exactly `delta`). Points below the short-`advance` threshold are delivered within it; a
    positively-skewed (ahead) node's send lands above the line and is deferred.
  - **right — convergence is clock-independent.** The converged-state invariance rate over the skew
    sweep is 1.0: last-writer-wins by `(version, value)` makes the cluster land on the same value
    regardless of any node's clock — the property deterministic-simulation testing injects skew to
    verify.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed21 import ED21Result


def plot_ed21(result: ED21Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: deliver_after vs skew, colored by delivered/deferred after the short advance
    skews = [t["skew"] for t in result.timing]
    deliver = [t["deliver_after"] for t in result.timing]
    colors = ["#2ca02c" if t["converged_after_short"] else "#d62728" for t in result.timing]
    ax_l.plot(skews, deliver, color="#1f77b4", alpha=0.5, zorder=1)
    ax_l.scatter(skews, deliver, c=colors, s=90, zorder=2, edgecolors="black", linewidths=0.5)
    for t in result.timing:
        verdict = "delivered" if t["converged_after_short"] else "deferred"
        ax_l.annotate(verdict, (t["skew"], t["deliver_after"]),
                      textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax_l.set_xlabel("node clock offset (skew δ)")
    ax_l.set_ylabel("deliver_after stamped on the node's send")
    ax_l.set_title("Panel A — skew shifts the send timing by exactly δ\n"
                   "green = lands within a short advance; red = deferred (clock ahead)",
                   fontsize=10)
    ax_l.grid(True, alpha=0.3)
    ax_l.axhline(0, color="gray", lw=0.8)

    # right: converged invariance rate (clock-independence)
    rate = result.converged_invariant_rate
    bar = ax_r.bar(["converged\nstate"], [rate], color="#2ca02c", alpha=0.85, width=0.5)
    verdict = "clock-independent\n(version-LWW)" if rate == 1.0 else "CLOCK-DEPENDENT"
    ax_r.text(bar[0].get_x() + bar[0].get_width() / 2, rate + 0.02,
              f"{rate:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.2)
    ax_r.set_ylabel("invariance rate over the skew sweep")
    ax_r.set_title("Panel B — convergence is clock-independent\n"
                   "skew shifts when a write lands, never which write wins",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED21 — clock skew: a per-node timing shift convergence is immune to  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
