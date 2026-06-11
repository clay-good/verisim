"""Plot the ED40 figure (SPEC-7 §3.2, DS0 incr 33): the CRDT OR-Map.

Two panels from one :class:`~verisim.experiments.ed40.ED40Result`:

  - **left — map operations + the two composed semantics.** mput/mget/mkeys reads field+value (1.0);
    mdel removes a field (1.0); a concurrent value update resolves by LWW (1.0); a concurrent mput
    survives a concurrent mdel — add-wins field presence (1.0); a partitioned-minority op is
    acknowledged (AP, 1.0).
  - **right — convergence (the composed join).** a gossip chain converges every node to the same
    fields + values (1.0); anti_entropy on each node (1.0); idempotent (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed40 import ED40Result


def plot_ed40(result: ED40Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # left: map ops + the two composed semantics
    labels = ["basic\nread", "delete\nremoves", "value\nLWW", "add-wins\npresence", "avail\n(AP)"]
    vals = [
        result.basic_read_rate,
        1.0 if result.delete_removes_field else 0.0,
        1.0 if result.value_resolves_lww else 0.0,
        1.0 if result.add_wins_presence else 0.0,
        1.0 if result.always_available else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#1f77b4", "#d62728", "#1f77b4"]
    caps = [f"exact ({result.n_sizes})", "field gone", "LWW half", "OR-Set half", "minority op"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — map ops + the two composed semantics\n"
                   "OR-Set field presence (add-wins) + LWW field values",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: convergence (the composed join)
    labels = ["gossip\nconverges", "anti_entropy\nconverges", "idempotent"]
    vals = [
        1.0 if result.gossip_converges else 0.0,
        1.0 if result.anti_entropy_converges else 0.0,
        1.0 if result.idempotent else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#9467bd"]
    caps = ["same fields+vals", "same fields", "2nd gossip stable"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — convergence (the composed join)\n"
                   "field presence by set-union, field value by LWW",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED40 — the CRDT OR-Map: a CRDT of CRDTs (OR-Set presence ∘ LWW values)"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
