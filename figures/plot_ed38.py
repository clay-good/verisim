"""Plot the ED38 figure (SPEC-7 §3.2, DS0 incr 31): the CRDT MV-register.

Two panels from one :class:`~verisim.experiments.ed38.ED38Result`:

  - **left — conflict surfaced, not lost.** mvput/mvget reads back the value (1.0); a sequential
    overwrite resolves to one (1.0); two concurrent writes BOTH survive as siblings (1.0) — the
    boldface bar, exactly where a LWW put silently drops one; a partitioned-minority mvput is
    acknowledged (AP, 1.0).
  - **right — convergence and resolution.** a gossip chain converges every node to the same sibling
    set (1.0); anti_entropy on each node (1.0); idempotent (1.0); a later context-aware mvput
    the siblings to one value cluster-wide (1.0) — the Dynamo read-and-resolve.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed38 import ED38Result


def plot_ed38(result: ED38Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: conflict surfaced, not lost
    labels = ["basic\nread", "sequential\nresolves", "siblings\npreserved", "always avail\n(AP)"]
    vals = [
        result.basic_read_rate,
        1.0 if result.sequential_resolves else 0.0,
        1.0 if result.siblings_preserved else 0.0,
        1.0 if result.always_available else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#d62728", "#1f77b4"]
    caps = [f"exact ({result.n_sizes})", "one value", "vs LWW loss", "minority writes"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=8.5)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — conflict surfaced, not lost\n"
                   "concurrent writes become siblings (vs the KV's silent LWW loss)",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: convergence and resolution
    labels = ["gossip\nconverges", "anti_entropy\nconverges", "idempotent", "resolves\nconflict"]
    vals = [
        1.0 if result.gossip_converges else 0.0,
        1.0 if result.anti_entropy_converges else 0.0,
        1.0 if result.idempotent else 0.0,
        1.0 if result.resolves_conflict else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#9467bd", "#ff7f0e"]
    caps = ["all nodes same", "all nodes same", "2nd gossip stable", "context write -> {c}"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8.5)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — convergence and resolution\n"
                   "set-union join converges; a context-aware write collapses the siblings",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED38 — the CRDT MV-register: conflict-surfacing, convergent, resolvable"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
