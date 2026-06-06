"""Plot the ED13 figure (SPEC-7 §3.4, DS0 incr 5): causal consistency and its anomaly.

Two panels from one :class:`~verisim.experiments.ed13.ED13Result`:

  - **left -- the anomaly is forbidden.** The effect-before-cause anomaly rate (the observer adopts
    the effect y=b while its cause x=a is still in flight) per model: **eventual** admits it (1.0,
    greedy delivery), **causal** forbids it (0.0 -- the message carries deps={x@1} and is
    held until the cause arrives). The cross-object ordering guarantee, exhibited.
  - **right -- causal does not over-synchronize, and stays live.** Two controls: causal **holds the
    dependent message** (rate 1.0) but **never the independent one** (rate 0.0) -- it orders only
    causally-linked writes, leaving concurrent writes free; and after a heal the eventual and causal
    clusters reach the **identical durable state** (rate 1.0, causal in-flight drained to 0) -- a
    delivery-order refinement, not a different outcome.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed13 import ED13Result

_COLOR = {"eventual": "#d62728", "causal": "#2ca02c"}


def plot_ed13(result: ED13Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: the effect-before-cause anomaly rate per model
    models = [r["model"] for r in result.anomaly]
    rates = [r["anomaly_rate"] for r in result.anomaly]
    bars = ax_l.bar(models, rates, color=[_COLOR[m] for m in models], alpha=0.85, width=0.6)
    for bar, r in zip(bars, result.anomaly, strict=True):
        verdict = "anomaly\nadmitted" if r["anomaly_rate"] > 0 else "anomaly\nforbidden"
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["anomaly_rate"] + 0.02,
                  f"{r['anomaly_rate']:.2f}  ({r['anomalies']}/{r['scenarios']})\n{verdict}",
                  ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.18)
    ax_l.set_ylabel("effect-before-cause anomaly rate\n(observer sees y=b while x still in flight)")
    ax_l.set_title("Panel A — causal forbids the anomaly eventual admits\n"
                   "(the message carries deps={x@1} and is held until the cause arrives)",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: causal blocks only causally-linked writes, and convergence is preserved
    o = result.over_sync
    c = result.convergence
    labels = ["dependent\nmessage", "independent\nmessage", "eventual ≡ causal\nafter heal"]
    vals = [o["dependent_held_rate"], o["independent_held_rate"], c["identical_final_state_rate"]]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    bars2 = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    notes = ["held\n(ordered)", "delivered\n(concurrent)",
             f"identical\n(in-flight→{c['causal_inflight_after_heal']})"]
    for bar, val, note in zip(bars2, vals, notes, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}\n{note}",
                  ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.22)
    ax_r.set_ylabel("rate over the battery")
    ax_r.set_title("Panel B — causal orders only causally-linked writes,\n"
                   "and still converges (a delivery-order refinement)", fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED13 — causal consistency (DS0 incr 5): effect-before-cause anomaly forbidden, "
                 "concurrency + convergence preserved  •  "
                 f"Tier-B reproduces it bit-for-bit = {result.tier_b_agrees} "
                 f"({result.tier_b_steps} steps)", y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
