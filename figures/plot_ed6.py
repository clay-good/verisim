"""Plot the ED6 figure (SPEC-7 §10.1, DS8): H5 -- distributed counterfactual grounding.

Two panels from one :class:`~verisim.experiments.ed6.run_ed6` stat list:

  - **left -- intervention exact** (the robust headline): full next-cluster-state bit-exact
    prediction of held-out fault interventions, one bar per training arm. ``+counterfactual``
    beating *both*
    ``trajectory`` arms is the H5 signal -- the lift being structure (free oracle fault branches),
    not data volume (the matched-count ``trajectory-more`` control).
  - **right -- medium recall**: of the interventions whose true effect *changes the medium* (a new
    partition split or a crashed node -- the split-brain precondition, §17 Q7), does the model
    predict the resulting partition/down structure exactly? The operationally-decisive readout.

Bars carry bootstrap-CI whiskers over the held-out eval seeds.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed6 import ARMS, METRICS, ArmStat

_TITLE = {
    "intervention_exact": "intervention exact\n(full next-cluster-state, bit-for-bit)",
    "medium_recall": "medium recall\n(predicts the partition/crash split-brain)",
}
_COLOR = {"trajectory": "#9bbcd6", "trajectory-more": "#c44e52", "+counterfactual": "#1f77b4"}


def plot_ed6(stats: list[ArmStat], path: str | Path) -> Path:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.4 * len(METRICS), 4.4), squeeze=False)
    for ax, metric in zip(axes[0], METRICS, strict=True):
        cells = [s for s in stats if s.metric == metric]
        order = {a: i for i, a in enumerate(ARMS)}
        cells.sort(key=lambda s: order.get(s.arm, 99))
        xs = range(len(cells))
        ax.bar(
            list(xs), [c.mean for c in cells],
            yerr=[[c.mean - c.ci_lo for c in cells], [c.ci_hi - c.mean for c in cells]],
            color=[_COLOR.get(c.arm, "#999") for c in cells], capsize=4,
        )
        for i, c in zip(xs, cells, strict=True):
            ax.annotate(f"{c.mean:.2f}", (i, c.mean), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([c.arm for c in cells], rotation=15, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(_TITLE[metric])
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "ED6 / H5 — does free oracle counterfactual fault replay train distributed intervention "
        "fidelity? (SPEC-7)"
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
