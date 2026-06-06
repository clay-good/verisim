"""Plot the ED4 consistency-level figure (SPEC-7 §12, H20/H19): the gap tracks the in-flight medium.

One panel per error mode (`gross` | `subtle`). Each panel groups the free-running bit-faithful and
consistency-faithful horizon by consistency level (strong `linearizable` → weak `eventual`), with
the per-level **in-flight medium rate** annotated. The `subtle` panel is the headline: under
`eventual` the consistency-faithful horizon outlasts the bit-faithful one (the H19 gap) because a
corrupted in-flight message is bit-visible but consistency-invisible until delivery; under
`linearizable` the in-flight medium is structurally absent (rate 0), so the subtle error class is
empty and the gap collapses. The `gross` panel is the control: a durable-replica error is
consistency-visible at once, so bit and consistency horizons coincide at every level.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed4_consistency import ED4ConsistencyResult

_COLORS = {"bit": "#d62728", "cons": "#2ca02c"}


def plot_ed4_consistency(  # pragma: no cover - local plotting
    result: ED4ConsistencyResult, path: str | Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(r["mode"] for r in result.rows))
    levels = list(dict.fromkeys(r["level"] for r in result.rows))
    by = {(r["level"], r["mode"]): r for r in result.rows}

    fig, axes = plt.subplots(1, len(modes), figsize=(6.0 * len(modes), 4.8), squeeze=False)
    width = 0.38
    x = range(len(levels))
    for ax, mode in zip(axes[0], modes, strict=False):
        bit = [by[(lv, mode)]["bit_h"] for lv in levels]
        cons = [by[(lv, mode)]["cons_h"] for lv in levels]
        ax.bar([i - width / 2 for i in x], bit, width, color=_COLORS["bit"],
               label="bit-faithful  H_ε")
        ax.bar([i + width / 2 for i in x], cons, width, color=_COLORS["cons"],
               label="consistency-faithful  H_ε")
        for i, lv in enumerate(levels):
            r = by[(lv, mode)]
            if r["consistency_outlasts"]:
                ax.annotate(f"gap +{r['gap']:.1f}", (i, max(bit[i], cons[i])),
                            textcoords="offset points", xytext=(0, 8), ha="center",
                            fontsize=9, color="#2ca02c", fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels([
            f"{lv}\n({'strong' if i == 0 else 'weak'})\nin-flight/step: "
            f"{by[(lv, mode)]['inflight_rate']:.1f}"
            for i, lv in enumerate(levels)
        ])
        ax.set_ylabel("free-running faithful horizon  H_ε  (steps)")
        ax.set_title(f"{mode}-error proposer", fontsize=10)
        ax.legend(fontsize=8, loc="upper center")
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(y=0.18)

    fig.suptitle("ED4 (consistency level) — the H19 gap is a weak-consistency phenomenon: "
                 "it tracks the in-flight medium (H20, SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
