"""Plot the ED39 figure (SPEC-7 §3.2, DS0 incr 32): the CRDT LWW-register.

Two panels from one :class:`~verisim.experiments.ed39.ED39Result`:

  - **left — happens-after wins, deterministically.** lwwput/lwwget reads back the value (1.0); a
    causally-later write wins regardless of node id — even a lower-id node's later write beats a
    higher-id earlier one (1.0); concurrent writes resolve to one value on every node (1.0); a
    partitioned-minority lwwput is acknowledged (AP, 1.0).
  - **right — convergence (the max-by-timestamp join).** a gossip chain converges every node to the
    single winner (1.0); anti_entropy on each node (1.0); idempotent (1.0); the concurrent loser is
    dropped — one value, not siblings (1.0, the policy-opposite of the MV-register).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed39 import ED39Result


def plot_ed39(result: ED39Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: happens-after wins, deterministically
    labels = ["basic\nread", "causal LWW\n(happens-after)", "deterministic\nresolve",
              "always avail\n(AP)"]
    vals = [
        result.basic_read_rate,
        1.0 if result.causal_lww else 0.0,
        1.0 if result.deterministic_resolve else 0.0,
        1.0 if result.always_available else 0.0,
    ]
    colors = ["#2ca02c", "#1f77b4", "#2ca02c", "#1f77b4"]
    caps = [f"exact ({result.n_sizes})", "ts beats id", "one winner", "minority writes"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=8.5)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — happens-after wins, deterministically\n"
                   "a Lamport-timestamp total order picks one winner",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: convergence (the max-by-timestamp join)
    labels = ["gossip\nconverges", "anti_entropy\nconverges", "idempotent", "loser\ndropped"]
    vals = [
        1.0 if result.gossip_converges else 0.0,
        1.0 if result.anti_entropy_converges else 0.0,
        1.0 if result.idempotent else 0.0,
        1.0 if result.loser_dropped else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#9467bd", "#d62728"]
    caps = ["all nodes win", "all nodes win", "2nd gossip stable", "one value (vs MV)"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8.5)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — convergence (the max-by-timestamp join)\n"
                   "deterministic resolution: the concurrent loser is dropped",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED39 — the CRDT LWW-register: deterministic, Lamport-ordered, convergent"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
