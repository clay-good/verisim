"""Plot the ED3 figure (SPEC-7 §8.3, §10 ED3; DS7): correction-operator comparison.

One panel per proposer error mode (`gross` | `subtle`). Each panel bars the faithful horizon per
correction operator (`hard_reset` / `residual` / `projection` / `replicas_only`) with bootstrap-CI
error bars. The three full-correction operators snap the coupled state to truth and so are identical
on `H_ε` (the v0 identity, SPEC-2 §6.2); the partial `replicas_only` operator -- which snaps the
durable replicas but trusts the model's in-flight -- is the test of whether the distributed world
breaks that identity.

The figure makes the mode-dependent, structural verdict visible: in the **gross** panel (a corrupted
replica write) all four bars match -- `replicas_only` fixes the replica, so the identity holds. In
the **subtle** panel (a corrupted in-flight message) the three full-correction bars match but
`replicas_only` collapses -- it trusts the corrupted in-flight and the coupled state keeps drifting.
The in-flight medium is the distributed world's hidden state a partial correction cannot see.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed3 import ED3Result

_COLORS = {
    "hard_reset": "#d62728",
    "residual": "#ff7f0e",
    "projection": "#9467bd",
    "replicas_only": "#1f77b4",
}


def plot_ed3(result: ED3Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(c["mode"] for c in result.cells))
    operators = list(dict.fromkeys(c["operator"] for c in result.cells))
    by = {(c["mode"], c["operator"]): c for c in result.cells}
    verdict_by_mode = {v["mode"]: v for v in result.verdict}

    fig, axes = plt.subplots(1, len(modes), figsize=(6.0 * len(modes), 4.8), squeeze=False)
    for ax, mode in zip(axes[0], modes, strict=False):
        xs = range(len(operators))
        ys = [by[(mode, op)]["h_eps"] for op in operators]
        lo = [by[(mode, op)]["h_eps"] - by[(mode, op)]["ci_lo"] for op in operators]
        hi = [by[(mode, op)]["ci_hi"] - by[(mode, op)]["h_eps"] for op in operators]
        ax.bar(list(xs), ys, 0.64, color=[_COLORS.get(op, "#888") for op in operators],
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})
        ax.set_xticks(list(xs))
        ax.set_xticklabels(operators, rotation=15, fontsize=8)
        ax.set_ylabel("faithful horizon  H_ε  (steps)")
        v = verdict_by_mode.get(mode)
        if v is not None:
            verdict = "identity BROKEN" if v["partial_costs_horizon"] else "identity holds"
            ax.set_title(f"{mode}-error proposer\nfull H={v['full_h']:.1f}  "
                         f"partial(replicas_only) H={v['partial_h']:.1f}  → {verdict} "
                         f"(gap {v['horizon_gap']:.1f})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED3 — correction operators: does the distributed world break v0's operator "
                 "identity? (SPEC-7 §8.3)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
