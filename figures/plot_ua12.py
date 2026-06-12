"""Plot the UA12 figure (SPEC-20 §7, H92): the operational detection characteristic.

Two panels from a :class:`~verisim.experiments.ua_host_detection.UA12Result`:

  - **left -- the horizon sweep.** Precision, recall, and F1 of the faithful detector (flat at 1.0)
    vs the free detector (drifting `M_θ`) as the horizon grows. The free detector's **precision
    falls with its recall** -- drift flags untouched files *and* misses real corruptions -- so the
    F1 (solid red) collapses below the recall (dashed red), the part UA8's recall-only metric could
    not see. The shaded F1 gap is the operational cost of drift.
  - **right -- the ρ-knee for F1.** Grounding (re-anchor every round(1/ρ) steps) buys back the
    deployable precision *and* recall at the cheap UA9 knee: F1 rises from the free floor to the
    ceiling, the operating characteristic restored sub-linearly in oracle calls.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ua_host_detection import UA12Result, knee_rho


def plot_ua12(result: UA12Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))

    hs = [r.horizon for r in result.horizon_sweep]
    # left: precision / recall / F1, faithful (blue) vs free (red)
    ax_l.plot(hs, [r.faithful["f1"] for r in result.horizon_sweep], "-o", color="#1f77b4",
              label="faithful F1")
    ax_l.plot(hs, [r.free["precision"] for r in result.horizon_sweep], ":^", color="#d62728",
              label="free precision (false-alarm cost)")
    ax_l.plot(hs, [r.free["recall"] for r in result.horizon_sweep], "--s", color="#ff7f0e",
              label="free recall (UA8's metric)")
    ax_l.plot(hs, [r.free["f1"] for r in result.horizon_sweep], "-D", color="#d62728",
              label="free F1 (deployability)")
    ax_l.fill_between(
        hs,
        [r.free["f1"] for r in result.horizon_sweep],
        [r.faithful["f1"] for r in result.horizon_sweep],
        color="#d62728", alpha=0.08,
    )
    ax_l.set_xlabel("horizon (steps of compounding content drift)")
    ax_l.set_ylabel("score")
    ax_l.set_title("drift costs precision, not just recall (H92)")
    ax_l.set_ylim(-0.02, 1.05)
    ax_l.legend(fontsize=8, loc="center right")

    # right: the F1 knee vs ρ
    rhos = [k.rho for k in result.knee]
    ax_r.plot(rhos, [k.grounded["precision"] for k in result.knee], ":^", color="#2ca02c",
              label="precision")
    ax_r.plot(rhos, [k.grounded["recall"] for k in result.knee], "--s", color="#2ca02c",
              label="recall")
    ax_r.plot(rhos, [k.grounded["f1"] for k in result.knee], "-o", color="#1f77b4",
              label="F1 (deployability)")
    knee = knee_rho(result)
    ax_r.axvline(knee, color="#888", ls=":", label=f"F1 knee ρ = {knee:g}")
    ax_r.set_xlabel("ρ  (oracle-consultation budget)")
    ax_r.set_ylabel("score")
    ax_r.set_title(f"grounding restores the operating point (horizon {result.knee_horizon})")
    ax_r.set_ylim(-0.02, 1.05)
    ax_r.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "UA12 / H92: faithfulness governs detector deployability (precision + recall), "
        "and the cheap knee buys it back"
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
