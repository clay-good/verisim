"""Plot the ED25 figure (SPEC-7 §3.2/§4, DS0 incr 18): the leader lease.

Two panels from one :class:`~verisim.experiments.ed25.ED25Result`:

  - **left — local linearizable reads without a quorum.** A live lease serves `lread` (rate 1.0),
    a minority-stranded leader still reads locally (1.0) where its `propose` is `no_quorum` (the
    control, 1.0), and once the clock passes the deadline the read is `lease_expired` (1.0).
  - **right — the lease/election safety tension.** A fresh `elect` is blocked while the lease is
    live (`lease_held`, 1.0) and unblocked past expiry (1.0); a voluntary `step_down` releases the
    lease, so a successor elects with no wait (1.0) — fast handoff vs the wait-it-out crash path.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed25 import ED25Result


def plot_ed25(result: ED25Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: local reads without a quorum
    labels = ["valid lease\nlread", "minority leader\nlread", "minority leader\npropose (control)",
              "expired lease\nlread"]
    rates = [
        result.valid_lease_read_rate,
        result.minority_lread_rate,
        result.minority_propose_blocked_rate,
        result.expired_lease_read_rate,
    ]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    bars = ax_l.bar(labels, rates, color=colors, alpha=0.85, width=0.66)
    caps = [f"ok ({result.n_sizes})", "ok\n(no quorum needed)", "no_quorum", "lease_expired"]
    for bar, rate, cap in zip(bars, rates, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                  f"{rate:.2f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_l.set_ylim(0, 1.4)
    ax_l.set_ylabel("rate over the cluster-size sweep")
    ax_l.set_title("Panel A — local linearizable reads without a quorum\n"
                   "a live lease lets the leader read locally; expiry rejects",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: the lease/election safety tension
    labels = ["elect under\nlive lease", "elect after\nexpiry", "elect after\nstep_down"]
    vals = [
        1.0 if result.elect_blocked_under_lease else 0.0,
        1.0 if result.elect_after_expiry_ok else 0.0,
        1.0 if result.stepdown_releases_lease else 0.0,
    ]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.62)
    caps = ["lease_held\n(wait it out)", "elected\n(past deadline)", "elected\n(no wait)"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=8)
    ax_r.set_ylim(0, 1.4)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — the lease/election safety tension\n"
                   "a live lease fences elections; step_down releases it",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED25 — leader leases: local reads without a quorum + the lease tension  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
