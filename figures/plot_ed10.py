"""Plot the ED10 figure (SPEC-7 §5, §9.1; DS3 incr 2): Elle — black-box serializability checking.

Two panels from one :class:`~verisim.experiments.ed10.ED10Result`:

  - **left -- write skew, recovered black-box.** Elle's G2-cycle rate per isolation level, computed
    from the observable history alone (no oracle): **snapshot** ≈1.0 (the ``A →rw B →rw A`` anti-
    dependency cycle), **serializable** 0.0. This is ED9's oracle-side anomaly rate, recovered by a
    reference-free checker — the agreement is annotated.
  - **right -- Elle certifies the serializable level.** The fraction of contended histories Elle
    flags non-serializable (bootstrap-CI whiskers): **snapshot** positive (the anomalies SI admits),
    **serializable** 0.0 (the guarantee, certified independently of the oracle that enforces it).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed10 import ED10Result

_COLOR = {"serializable": "#1f77b4", "snapshot": "#d62728"}


def plot_ed10(result: ED10Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- left: write-skew G2-cycle rate (black-box) -----------------------------------------------
    levels = [r["isolation"] for r in result.write_skew]
    rate = [r["elle_g2_rate"] for r in result.write_skew]
    bars = ax_l.bar(levels, rate, color=[_COLOR[lv] for lv in levels], alpha=0.85)
    ax_l.set_ylabel("Elle G2 anti-dependency cycle rate\n(write skew, from the history alone)")
    ax_l.set_ylim(0, 1.08)
    ax_l.set_title("ED10: Elle recovers write skew black-box (no oracle)")
    for bar, r in zip(bars, result.write_skew, strict=True):
        agree = "=oracle" if r["elle_matches_oracle"] else "≠oracle"
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["elle_g2_rate"] + 0.02,
                  f"{r['elle_g2_rate']:.2f}\n({r['g2_cycles']}/{r['scenarios']}, {agree})",
                  ha="center", va="bottom", fontsize=9)

    # --- right: non-serializable rate under contention (Elle certifies serializable) --------------
    levels2 = [r["isolation"] for r in result.contention]
    nsr = [r["nonserializable_rate"] for r in result.contention]
    lo = [max(0.0, r["nonserializable_rate"] - r["ci_lo"]) for r in result.contention]
    hi = [max(0.0, r["ci_hi"] - r["nonserializable_rate"]) for r in result.contention]
    bars2 = ax_r.bar(levels2, nsr, color=[_COLOR[lv] for lv in levels2], alpha=0.85)
    ax_r.errorbar(range(len(levels2)), nsr, yerr=[lo, hi], fmt="none", ecolor="#333",
                  capsize=5, lw=1.2)
    ax_r.set_ylabel("fraction of contended histories Elle\nflags non-serializable")
    ax_r.set_ylim(0, 1.08)
    ax_r.set_title("ED10: Elle certifies serializable (0), catches snapshot anomalies")
    for bar, val in zip(bars2, nsr, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                  ha="center", va="bottom", fontsize=9)

    all_match = all(r["elle_matches_oracle"] for r in result.write_skew)
    fig.suptitle(
        "Elle-style black-box serializability checking (DS3 incr 2): a reference-free checker "
        f"recovers the write-skew anomaly the oracle sees  •  agrees every scenario = {all_match}",
        y=1.00, fontsize=9.5,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
