"""Plot the ED24 figure (SPEC-7 §3.2, DS0 incr 17): voluntary leader step-down.

Two panels from one :class:`~verisim.experiments.ed24.ED24Result`:

  - **left — the voluntary-handoff lifecycle.** After `step_down` the cluster is leaderless at the
    same term, so the same node's `propose` is rejected (rate 1.0); a fresh `elect` of a successor
    lands at a strictly higher term (1.0) and that successor commits (1.0). A clean handoff is
    `step_down` then `elect <successor>`, with no leaderless commit window.
  - **right — authority + partition-independence.** Only the current leader may step down (a
    non-leader is rejected, 1.0; a second step_down is a no-op reject, 1.0). The sharp case: a
    minority-stranded leader can still step down (1.0) where its `propose` is `no_quorum` (the
    control, 1.0) — relinquishing power needs no quorum, exercising it does.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed24 import ED24Result


def plot_ed24(result: ED24Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: the voluntary-handoff lifecycle
    labels = ["step_down →\nleaderless", "successor\nhigher term", "successor\ncommits"]
    rates = [
        result.handoff_leaderless_rate,
        result.reelect_higher_term_rate,
        result.new_leader_commits_rate,
    ]
    colors = ["#9467bd", "#1f77b4", "#2ca02c"]
    bars = ax_l.bar(labels, rates, color=colors, alpha=0.85, width=0.62)
    caps = [
        f"propose not_leader ({result.n_sizes})",
        f"elect term+ ({result.n_sizes})",
        f"propose ok ({result.n_sizes})",
    ]
    for bar, rate, cap in zip(bars, rates, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.25)
    ax_l.set_ylabel("rate over the cluster-size sweep")
    ax_l.set_title("Panel A — voluntary-handoff lifecycle\n"
                   "step_down leaves no leaderless commit window",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: authority + partition-independence
    labels = ["non-leader\nstep_down", "second\nstep_down", "minority leader\nstep_down",
              "minority leader\npropose (control)"]
    vals = [
        1.0 if result.nonleader_stepdown_rejected else 0.0,
        1.0 if result.second_stepdown_rejected else 0.0,
        1.0 if result.minority_leader_steps_down else 0.0,
        1.0 if result.minority_propose_blocked else 0.0,
    ]
    colors = ["#2ca02c", "#2ca02c", "#1f77b4", "#d62728"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.66)
    caps = ["not_leader", "not_leader\n(idempotent)", "stepped_down\n(no quorum needed)",
            "no_quorum\n(commit needs one)"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_r.set_ylim(0, 1.4)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — authority + partition-independence\n"
                   "relinquishing power needs no quorum; committing does",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED24 — voluntary step-down: graceful handoff + relinquish-needs-no-quorum  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
