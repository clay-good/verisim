"""Plot the ED14 figure (SPEC-7 §3.4, DS0 incr 7): the quorum consensus availability/safety trade.

Two panels from one :class:`~verisim.experiments.ed14.ED14Result`:

  - **left — the availability frontier.** Per model, whether a write from a ``k``-node partition
    side commits, swept over ``k``. ``eventual`` is flat at "available" (it never coordinates);
    ``quorum`` steps up at the majority frontier ``k = ⌈n/2⌉`` (available on the majority side, dark
    on the minority); ``linearizable`` is flat at "unavailable" under any partition (needs every
    replica). The dashed line marks the majority threshold — the quorum step.
  - **right — split-brain prevention.** Per model, the fork rate when *both* partition sides write
    the same key: ``eventual`` forks every time (both commit, the object diverges); ``quorum`` and
    ``linearizable`` never fork. Only ``quorum`` is in the bottom-right of the availability×safety
    plane — available on the majority side *and* divergence-free, why real systems use quorums.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed14 import ED14Result

_COLOR = {"eventual": "#d62728", "quorum": "#2ca02c", "linearizable": "#1f77b4"}
_MARKER = {"eventual": "o", "quorum": "s", "linearizable": "^"}


def plot_ed14(result: ED14Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: the availability frontier — commit (1/0) vs partition-side size k, one line per model
    for r in result.availability:
        ks = sorted(r["commits"])
        ys = [r["commits"][k] for k in ks]
        ax_l.plot(ks, ys, marker=_MARKER[r["model"]], markersize=8, linewidth=2,
                  color=_COLOR[r["model"]], label=r["model"], alpha=0.9)
    ax_l.axvline(result.majority - 0.5, color="#555", linestyle="--", linewidth=1, alpha=0.7)
    ax_l.text(result.majority - 0.45, 0.5, f"  majority = {result.majority}",
              fontsize=9, color="#555", rotation=90, va="center")
    ax_l.set_xlabel("size of the writing partition side  (k of n nodes)")
    ax_l.set_ylabel("write commits  (1 = ok, 0 = unavailable)")
    ax_l.set_ylim(-0.1, 1.18)
    ax_l.set_yticks([0, 1])
    ax_l.set_yticklabels(["unavailable", "ok"])
    ax_l.set_title("Panel A — the availability frontier\n"
                   "quorum stays available on the majority side; linearizable goes dark",
                   fontsize=10)
    ax_l.legend(title="consistency model", fontsize=8, loc="center left")
    ax_l.grid(True, alpha=0.3)

    # right: split-brain (fork) rate when both sides write
    models = [r["model"] for r in result.split_brain]
    rates = [r["fork_rate"] for r in result.split_brain]
    bars = ax_r.bar(models, rates, color=[_COLOR[m] for m in models], alpha=0.85, width=0.6)
    for bar, r in zip(bars, result.split_brain, strict=True):
        verdict = "SPLIT-BRAIN" if r["fork_rate"] > 0 else "no fork"
        ax_r.text(bar.get_x() + bar.get_width() / 2, r["fork_rate"] + 0.02,
                  f"{r['fork_rate']:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.2)
    ax_r.set_ylabel("split-brain (fork) rate\n(both sides write → divergent committed value)")
    ax_r.set_title("Panel B — split-brain prevention\n"
                   "only quorum is both available (majority side) and fork-free", fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED14 — the quorum (Raft-subset) consensus model: available on the majority side, "
                 "no split-brain  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
