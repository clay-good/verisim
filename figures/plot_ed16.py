"""Plot the ED16 figure (SPEC-7 §3.2, DS0 incr 9): read-committed — lost update + its price.

Two panels from one :class:`~verisim.experiments.ed16.ED16Result`:

  - **left -- the lost-update anomaly.** The anomaly rate (fraction of same-key read-modify-write
    scenarios in which *both* transactions commit, so the earlier write is lost) per level:
    **read_committed** admits it (≈1.0), **snapshot** and **serializable** forbid it (0.0). The
    textbook lost-update distinction, exhibited.
  - **right -- the price of preventing lost update.** The abort rate under a read-modify-write
    contended workload per level (bootstrap-CI whiskers): read_committed commits everything (abort
    rate 0) — the apparent throughput it buys by selling the correctness of the left panel — while
    snapshot and serializable pay aborts to preserve every update.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed16 import ED16Result

_COLOR = {"serializable": "#1f77b4", "snapshot": "#d62728", "read_committed": "#ff7f0e"}


def plot_ed16(result: ED16Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- left: lost-update anomaly rate -----------------------------------------------------------
    levels = [r["isolation"] for r in result.lost_update]
    anomaly = [r["anomaly_rate"] for r in result.lost_update]
    bars = ax_l.bar(levels, anomaly, color=[_COLOR[lv] for lv in levels], alpha=0.85)
    ax_l.set_ylabel("lost-update anomaly rate\n(both same-key RMW txns commit)")
    ax_l.set_ylim(0, 1.08)
    ax_l.set_title("ED16: read_committed admits lost update, snapshot/serializable forbid it")
    for bar, r in zip(bars, result.lost_update, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["anomaly_rate"] + 0.02,
                  f"{r['anomaly_rate']:.2f}\n({r['anomalies']}/{r['scenarios']})",
                  ha="center", va="bottom", fontsize=9)

    # --- right: abort rate (the price of preventing lost update) --------------------------------
    levels2 = [r["isolation"] for r in result.abort_rate]
    rate = [r["abort_rate"] for r in result.abort_rate]
    lo = [max(0.0, r["abort_rate"] - r["ci_lo"]) for r in result.abort_rate]
    hi = [max(0.0, r["ci_hi"] - r["abort_rate"]) for r in result.abort_rate]
    bars2 = ax_r.bar(levels2, rate, color=[_COLOR[lv] for lv in levels2], alpha=0.85)
    ax_r.errorbar(range(len(levels2)), rate, yerr=[lo, hi], fmt="none", ecolor="#333",
                  capsize=5, lw=1.2)
    ax_r.set_ylabel("abort (conflict) rate under read-modify-write contention")
    ax_r.set_ylim(0, max(rate) * 1.3 + 0.05 if any(rate) else 1.0)
    ax_r.set_title("ED16: read_committed never aborts — the throughput it sells correctness for")
    for bar, val in zip(bars2, rate, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=9)

    all_agree = all(r["tier_b_agrees"] for r in result.lost_update)
    fig.suptitle(
        "Read-committed isolation (DS0 incr 9): the weakest level admits lost update for fewer "
        f"aborts  •  Tier-B agrees on every scenario = {all_agree}",
        y=1.00, fontsize=9.5,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
