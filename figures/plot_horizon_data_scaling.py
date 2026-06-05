"""Plot the HS1.2 data cross-axis (SPEC-10 §5) -- does feeding a fixed model recover its horizon?

Capacity is held fixed; the x-axis is the number of transitions in the shared coverage set. A
*rising* `H_free` with data means the HS1.1 frontier decline was data starvation (Chinchilla), not a
capacity wall; a flat/low `H_free` means the wall is real at this capacity. Two panels, figures from
the reduced :class:`ScaleStat` records (SPEC-2 §7.3).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.horizon_scaling import ModelScale, ScaleStat


def plot_horizon_data_scaling(  # pragma: no cover - local plotting
    stats: list[ScaleStat],
    scale: ModelScale,
    path: str | Path,
    *,
    suptitle: str = "HS1.2 — the data cross-axis: capacity wall or data starvation? (SPEC-10)",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(metric: str) -> tuple[list[int], list[float], list[float], list[float]]:
        cells = sorted((s for s in stats if s.metric == metric), key=lambda s: s.params)
        return (
            [c.params for c in cells],
            [c.mean for c in cells],
            [c.ci_lo for c in cells],
            [c.ci_hi for c in cells],
        )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6))

    for metric, label, color in (
        ("h_free_id", "H_free  in-distribution", "#1f77b4"),
        ("h_free_ood", "H_free  adversarial (ood)", "#d62728"),
        ("h_indep_id", "independence p/(1−p)  [id]", "#999999"),
    ):
        xs, ys, lo, hi = series(metric)
        if not xs:
            continue
        ls = "--" if metric.startswith("h_indep") else "-"
        ax_l.plot(xs, ys, marker="o", ls=ls, color=color, label=label)
        if ls == "-":
            ax_l.fill_between(xs, lo, hi, alpha=0.15, color=color)
    ax_l.set_xscale("log")
    ax_l.set_xlabel("coverage-set size (transitions)")
    ax_l.set_ylabel("free-running faithful horizon (steps)")
    ax_l.set_title(f"Does data recover horizon? (fixed {scale.label}, {scale.params:,} params)")
    ax_l.legend(loc="upper left", fontsize=8)

    for metric, label, color in (
        ("one_step_acc_id", "p  in-distribution", "#1f77b4"),
        ("one_step_acc_ood", "p  adversarial (ood)", "#d62728"),
    ):
        xs, ys, lo, hi = series(metric)
        if not xs:
            continue
        ax_r.plot(xs, ys, marker="s", color=color, label=label)
        ax_r.fill_between(xs, lo, hi, alpha=0.15, color=color)
    ax_r.set_xscale("log")
    ax_r.set_ylim(0.0, 1.02)
    ax_r.set_xlabel("coverage-set size (transitions)")
    ax_r.set_ylabel("one-step acceptance p")
    ax_r.set_title("Per-step accuracy vs data")
    ax_r.legend(loc="lower right", fontsize=8)

    fig.suptitle(suptitle)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
