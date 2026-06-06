"""Plot the ED4 consistency-level learned figure (SPEC-7 §10.2, DS7): absolute H20 on the `M_θ`.

The learned-arm companion to :mod:`figures.plot_ed4_consistency` (the synthetic, gap-only version).
Where the synthetic arm can only report the H19 *gap* per level (the absolute horizon is confounded
by delta composition across levels), the learned arm trains one `M_θ` per level so the **absolute**
free-running horizon is comparable -- the H20 predictability question. Two panels from the
per-level trained models:

  - **left** -- the **absolute-predictability** bars: each level's free-running (ρ=0)
    **bit-faithful** horizon `H_ε` with bootstrap CIs. The H20 prediction is that the strong level
    (`linearizable`, no in-flight medium) is *more* predictable -- the model free-runs further --
    than the weak one (`eventual`), because weaker consistency leaves more hidden state to track.
  - **right** -- the **H19 gap** per level: consistency-faithful minus bit-faithful horizon. The
    synthetic ED4-consistency arm found this gap *exclusively* under `eventual` (the in-flight
    medium). The learned model shows it is **positive at both levels** -- because a real model's
    errors land on consistency-invisible *bookkeeping* (clocks, the causal log, partition
    structure), not only the in-flight medium the dialed synthetic error targets, so the consistency
    oracle forgives more of the real model's errors than the synthetic "weak-consistency-only"
    reading predicts. The honest difference between the dialed and the real error distribution.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed4_consistency_learned import ED4ConsistencyLearnedResult

_COLOR = {"linearizable": "#2ca02c", "eventual": "#d62728"}  # strong green, weak red


def plot_ed4_consistency_learned(
    result: ED4ConsistencyLearnedResult, path: str | Path
) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(result.rows)
    levels = [r["level"] for r in rows]
    xs = list(range(len(rows)))
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: absolute bit-faithful free-running horizon per level (the H20 predictability)
    bit = [r["bit_h"] for r in rows]
    bit_err = [[r["bit_h"] - r["bit_lo"] for r in rows], [r["bit_hi"] - r["bit_h"] for r in rows]]
    ax_l.bar(xs, bit, 0.55, yerr=bit_err, capsize=5,
             color=[_COLOR.get(lv, "#999") for lv in levels])
    for x, r in zip(xs, rows, strict=True):
        ax_l.annotate(f"{r['bit_h']:.1f}", (x, r["bit_hi"]), ha="center", va="bottom", fontsize=9)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(levels, fontsize=10)
    ax_l.set_ylabel("free-running bit-faithful horizon $H_\\epsilon$ (ρ=0)")
    ax_l.set_title("absolute predictability per consistency level (H20)")
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: the H19 gap (consistency-faithful minus bit-faithful) per level
    gap = [r["gap"] for r in rows]
    gap_err = [[r["gap"] - r["gap_lo"] for r in rows], [r["gap_hi"] - r["gap"] for r in rows]]
    ax_r.bar(xs, gap, 0.55, yerr=gap_err, capsize=5,
             color=[_COLOR.get(lv, "#999") for lv in levels])
    for x, r in zip(xs, rows, strict=True):
        ax_r.annotate(f"{r['gap']:.1f}", (x, r["gap_hi"]), ha="center", va="bottom", fontsize=9)
    ax_r.axhline(0.0, color="black", linewidth=0.8)
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels(levels, fontsize=10)
    ax_r.set_ylabel("H19 gap: consistency-faithful − bit-faithful horizon")
    ax_r.set_title("the H19 gap per level (learned $M_\\theta$) — positive at both")
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED4 / H20 (learned $M_\\theta$, SPEC-7) — strong consistency is more "
                 "predictable; the consistency oracle forgives the real model at both levels",
                 fontsize=11)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
