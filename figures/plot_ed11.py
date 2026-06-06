"""Plot the ED11 figure (SPEC-7 §5, §9.1; DS3 incr 3): Elle's version oracle + the split-brain fork.

Two panels from one :class:`~verisim.experiments.ed11.ED11Result`:

  - **left -- the version oracle is sound.** Elle's G2 write-skew rate, now recovered from
    list-append *values* alone (no store-supplied MVCC versions): **snapshot** ≈1.0,
    **serializable** 0.0 — exactly ED10's rates, with the ``sound``/``agrees`` annotations proving
    the version history is bit-identical to the store's. The version oracle removes the last
    store cooperation Elle needed.
  - **right -- the split-brain fork only value-recovery can see.** The ``incompatible-order`` rate
    (bootstrap-CI whisker) on forked histories — 1.0, the anomaly the integer-version mode cannot
    represent — beside the un-forked control at 0. This is the §9.1 split-brain view caught
    reference-free from the client history alone.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed11 import ED11Result

_COLOR = {"serializable": "#1f77b4", "snapshot": "#d62728"}


def plot_ed11(result: ED11Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- left: write-skew G2 rate, recovered from values alone ------------------------------------
    levels = [r["isolation"] for r in result.recovery]
    rate = [r["elle_g2_rate"] for r in result.recovery]
    bars = ax_l.bar(levels, rate, color=[_COLOR[lv] for lv in levels], alpha=0.85)
    ax_l.set_ylabel("Elle G2 write-skew cycle rate\n(recovered from list-append values alone)")
    ax_l.set_ylim(0, 1.12)
    ax_l.set_title("ED11: the version oracle is sound (no store-supplied versions)")
    for bar, r in zip(bars, result.recovery, strict=True):
        tag = "sound=" + ("✓" if r["recovery_sound"] else "✗")
        agree = "=supplied" if r["agrees_supplied"] else "≠supplied"
        ax_l.text(bar.get_x() + bar.get_width() / 2, r["elle_g2_rate"] + 0.02,
                  f"{r['elle_g2_rate']:.2f}\n({tag}, {agree})",
                  ha="center", va="bottom", fontsize=9)

    # --- right: the split-brain fork only value-recovery can represent ----------------------------
    f = result.fork
    cats = ["split-brain\nfork", "un-forked\ncontrol"]
    vals = [f["incompatible_order_rate"], f["clean_control_flag_rate"]]
    lo = [max(0.0, f["incompatible_order_rate"] - f["ci_lo"]), 0.0]
    hi = [max(0.0, f["ci_hi"] - f["incompatible_order_rate"]), 0.0]
    bars2 = ax_r.bar(cats, vals, color=["#9467bd", "#7f7f7f"], alpha=0.85)
    ax_r.errorbar(range(len(cats)), vals, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=5,
                  lw=1.2)
    ax_r.set_ylabel("fraction flagged non-serializable\n(incompatible-order / split-brain)")
    ax_r.set_ylim(0, 1.12)
    ax_r.set_title("ED11: the fork the integer-version mode cannot represent")
    for bar, val in zip(bars2, vals, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                  ha="center", va="bottom", fontsize=9)

    all_sound = all(r["recovery_sound"] and r["agrees_supplied"] for r in result.recovery)
    fig.suptitle(
        "ED11 — Elle's version oracle: serializability from values alone"
        + (" (sound, agrees with the store)" if all_sound else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
