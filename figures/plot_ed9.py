"""Plot the ED9 figure (SPEC-7 §3.2, DS0 incr 3): transaction isolation — write skew + its price.

Two panels from one :class:`~verisim.experiments.ed9.ED9Result`:

  - **left -- the write-skew anomaly.** The anomaly rate (fraction of write-skew scenarios in which
    *both* disjoint-write transactions commit) per isolation level: **snapshot** admits it (≈1.0),
    **serializable** forbids it (0.0). The textbook SI/serializable distinction, exhibited.
  - **right -- the price of serializability.** The abort rate under a read-heavy contended workload
    per isolation level (bootstrap-CI whiskers): serializable validates the whole read-set, so it
    aborts strictly more than snapshot (which validates only write-write conflicts) — the extra
    aborts are exactly what buys the stronger guarantee.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed9 import ED9Result

_COLOR = {"serializable": "#1f77b4", "snapshot": "#d62728"}


def plot_ed9(result: ED9Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- left: write-skew anomaly rate ------------------------------------------------------------
    levels = [r["isolation"] for r in result.write_skew]
    anomaly = [r["anomaly_rate"] for r in result.write_skew]
    bars = ax_l.bar(levels, anomaly, color=[_COLOR[lv] for lv in levels], alpha=0.85)
    ax_l.set_ylabel("write-skew anomaly rate\n(both disjoint-write txns commit)")
    ax_l.set_ylim(0, 1.08)
    ax_l.set_title("ED9: snapshot admits write skew, serializable forbids it")
    for bar, r in zip(bars, result.write_skew, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["anomaly_rate"] + 0.02,
                  f"{r['anomaly_rate']:.2f}\n({r['anomalies']}/{r['scenarios']})",
                  ha="center", va="bottom", fontsize=9)

    # --- right: abort rate (the price of serializability) -----------------------------------------
    levels2 = [r["isolation"] for r in result.abort_rate]
    rate = [r["abort_rate"] for r in result.abort_rate]
    lo = [max(0.0, r["abort_rate"] - r["ci_lo"]) for r in result.abort_rate]
    hi = [max(0.0, r["ci_hi"] - r["abort_rate"]) for r in result.abort_rate]
    bars2 = ax_r.bar(levels2, rate, color=[_COLOR[lv] for lv in levels2], alpha=0.85)
    ax_r.errorbar(range(len(levels2)), rate, yerr=[lo, hi], fmt="none", ecolor="#333",
                  capsize=5, lw=1.2)
    ax_r.set_ylabel("abort (conflict) rate under read-heavy contention")
    ax_r.set_ylim(0, max(rate) * 1.3 + 0.05 if rate else 1.0)
    ax_r.set_title("ED9: the price of serializability — more aborts buy the guarantee")
    for bar, val in zip(bars2, rate, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=9)

    all_agree = all(r["tier_b_agrees"] for r in result.write_skew)
    fig.suptitle(
        "Transaction isolation (DS0 incr 3): serializable forbids write skew at the cost of more "
        f"aborts  •  Tier-B agrees on every scenario = {all_agree}",
        y=1.00, fontsize=9.5,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
