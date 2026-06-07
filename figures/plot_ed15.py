"""Plot the ED15 figure (SPEC-7 §3.2, DS0 incr 8): optimistic (OCC) vs pessimistic (2PL) CC.

Two panels from one :class:`~verisim.experiments.ed15.ED15Result`:

  - **left — wasted work (fail-fast).** The mean data operations an *aborted* transaction completed
    before aborting: **OCC** validates at commit, so an aborted txn did *all* its work (maximal
    waste); **2PL** detects the conflict at lock-acquisition, so it fails *earlier* (less waste).
    The classic optimistic/pessimistic tradeoff — OCC wastes more work per abort, 2PL pays upfront.
  - **right — same serializable guarantee.** Both `occ` (serializable) and `2pl` forbid write skew
    (anomaly rate 0.0), reaching serializability by opposite routes (validate-late vs lock-early).
    Both are deterministic and deadlock-free, and Tier-B reproduces both bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed15 import ED15Result

_COLOR = {"occ": "#d62728", "2pl": "#1f77b4"}


def plot_ed15(result: ED15Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: wasted work per aborted txn
    schemes = [r["cc"] for r in result.wasted]
    waste = [r["wasted_ops_per_abort"] for r in result.wasted]
    bars = ax_l.bar(schemes, waste, color=[_COLOR[s] for s in schemes], alpha=0.85, width=0.6)
    for bar, r in zip(bars, result.wasted, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["wasted_ops_per_abort"] + 0.03,
                  f"{r['wasted_ops_per_abort']:.2f} ops\n"
                  f"(commits {r['commits']}, aborts {r['aborts']})",
                  ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, max(waste) * 1.35 + 0.2 if waste else 1.0)
    ax_l.set_ylabel("wasted data ops per aborted txn\n(tget/tput completed before the abort)")
    ax_l.set_xticks(range(len(schemes)))
    ax_l.set_xticklabels(["OCC\n(optimistic,\nvalidate at commit)",
                           "2PL\n(pessimistic,\nlock early)"])
    ax_l.set_title("Panel A — wasted work: OCC aborts late (after all work),\n"
                   "2PL fails fast (at the conflicting lock)", fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: write-skew anomaly rate (both forbid it)
    schemes2 = [r["cc"] for r in result.write_skew]
    rates = [r["anomaly_rate"] for r in result.write_skew]
    bars2 = ax_r.bar(schemes2, rates, color=[_COLOR[s] for s in schemes2], alpha=0.85, width=0.6)
    for bar, r in zip(bars2, result.write_skew, strict=True):
        verdict = "WRITE SKEW" if r["anomaly_rate"] > 0 else "forbidden\n(serializable)"
        ax_r.text(bar.get_x() + bar.get_width() / 2, r["anomaly_rate"] + 0.02,
                  f"{r['anomaly_rate']:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.0)
    ax_r.set_ylabel("write-skew anomaly rate")
    ax_r.set_xticks(range(len(schemes2)))
    ax_r.set_xticklabels(["OCC\n(serializable)", "2PL"])
    ax_r.set_title("Panel B — same serializable guarantee, opposite routes\n"
                   "(OCC validates the read-set late, 2PL locks it early)", fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED15 — optimistic (OCC) vs pessimistic (2PL) concurrency control: the cost of "
                 "aborting  •  "
                 f"Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
