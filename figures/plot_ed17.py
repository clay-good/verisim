"""Plot the ED17 figure (SPEC-7 §3.2, DS0 incr 10): read-uncommitted, the dirty read + recovery.

Two panels from one :class:`~verisim.experiments.ed17.ED17Result`:

  - **left -- the dirty-read anomaly (oracle side).** The anomaly rate (fraction of scenarios in
    which ``B`` observed the *uncommitted* value an aborted ``A`` wrote) per level: read_uncommitted
    admits it (≈1.0), **read_committed** / **snapshot** / **serializable** forbid it (0.0). The
    textbook dirty-read distinction, exhibited — and the bottom rung of the SQL isolation hierarchy.
  - **right -- Elle recovers the dirty read black-box (reference-free side).** The §5.3 value oracle
    reconstructs the same anomaly from the client-visible history alone (no oracle, no cluster
    state): the ``dirty-read`` recovery rate per level matches the left panel exactly — the cheap
    black-box verifier agrees with the expensive oracle on the question it answers.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed17 import ED17Result

_COLOR = {
    "serializable": "#1f77b4",
    "snapshot": "#d62728",
    "read_committed": "#ff7f0e",
    "read_uncommitted": "#9467bd",
}


def plot_ed17(result: ED17Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # --- left: dirty-read anomaly rate (oracle side) ---------------------------------------------
    levels = [r["isolation"] for r in result.dirty_read]
    anomaly = [r["anomaly_rate"] for r in result.dirty_read]
    bars = ax_l.bar(levels, anomaly, color=[_COLOR[lv] for lv in levels], alpha=0.85)
    ax_l.set_ylabel("dirty-read anomaly rate\n(B observed an aborted txn's uncommitted write)")
    ax_l.set_ylim(0, 1.08)
    ax_l.set_title("ED17: read_uncommitted admits the dirty read, the stronger levels forbid it")
    ax_l.tick_params(axis="x", labelrotation=12)
    for bar, r in zip(bars, result.dirty_read, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["anomaly_rate"] + 0.02,
                  f"{r['anomaly_rate']:.2f}\n({r['dirty_anomalies']}/{r['scenarios']})",
                  ha="center", va="bottom", fontsize=9)

    # --- right: Elle value-oracle recovery rate (black-box side) ---------------------------------
    levels2 = [r["isolation"] for r in result.recovery]
    rate = [r["recovery_rate"] for r in result.recovery]
    bars2 = ax_r.bar(levels2, rate, color=[_COLOR[lv] for lv in levels2], alpha=0.85)
    ax_r.set_ylabel("Elle `dirty-read` recovery rate\n(black-box, from the client history alone)")
    ax_r.set_ylim(0, 1.08)
    ax_r.set_title("ED17: the value oracle recovers the dirty read reference-free (matches oracle)")
    ax_r.tick_params(axis="x", labelrotation=12)
    for bar, r in zip(bars2, result.recovery, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, r["recovery_rate"] + 0.02,
                  f"{r['recovery_rate']:.2f}\n({r['elle_anomalies']}/{r['scenarios']})",
                  ha="center", va="bottom", fontsize=9)

    all_agree = all(r["tier_b_agrees"] for r in result.dirty_read)
    all_match = all(r["matches_oracle"] for r in result.recovery)
    fig.suptitle(
        "Read-uncommitted isolation (DS0 incr 10): the weakest level admits the dirty read, "
        f"recovered black-box by Elle  •  Tier-B agrees = {all_agree}  •  Elle matches oracle = "
        f"{all_match}",
        y=1.00, fontsize=9.5,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
