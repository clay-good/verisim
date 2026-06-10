"""Plot the ED33 figure (SPEC-7 §3.2, DS0 incr 26): the tombstone delete.

Two panels from one :class:`~verisim.experiments.ed33.ED33Result`:

  - **left — delete is a versioned tombstone (LWW).** put+delete leaves every replica reading
    deleted (rate 1.0); the tombstone out-versions the put it deleted (1.0); a genuinely newer put
    legitimately brings the key back (1.0).
  - **right — the resurrection problem under partition + repair.** under a partition the minority
    still reads the old value (the danger, 1.0); after heal, anti_entropy (1.0) and pairwise gossip
    (1.0) converge the minority to deleted — the tombstone's higher version wins, no resurrection.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed33 import ED33Result


def plot_ed33(result: ED33Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: delete is a versioned tombstone
    labels = ["delete removes\n(all replicas)", "tombstone\nout-versions put",
              "newer put\nresurrects"]
    vals = [
        result.delete_removes_rate,
        1.0 if result.tombstone_outversions_put else 0.0,
        1.0 if result.newer_put_resurrects else 0.0,
    ]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    caps = [f"deleted ({result.n_sizes})", "ver+1", "higher ver wins"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — delete is a versioned tombstone (LWW)\n"
                   "a delete is a write of a tombstone, ordered by version",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: the resurrection problem under partition + repair
    labels = ["minority reads\ndeleted item", "anti_entropy\nrepairs", "gossip\nrepairs"]
    vals = [
        1.0 if result.minority_reads_deleted_item else 0.0,
        1.0 if result.anti_entropy_no_resurrection else 0.0,
        1.0 if result.gossip_no_resurrection else 0.0,
    ]
    colors = ["#ff7f0e", "#2ca02c", "#2ca02c"]
    caps = ["the danger", "tombstone wins", "no resurrection"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — the resurrection problem under partition + repair\n"
                   "the tombstone's higher version wins the merge — no resurrection",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED33 — the tombstone delete: versioned removal, resurrection-safe under partition"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
