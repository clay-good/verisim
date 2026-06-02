"""Shared scale-up harness for EN8/EN9 (SPEC-8 §7.1, milestone OG5).

The OG3/OG4 figures are single-seed *smoke* instances (SPEC-8 §7.1). This module is the reusable
machinery that turns them into a result that cannot be dismissed: sweep **world size x model size x
seed**, reduce each cell to a *gap* (the difference the oracle buys), and aggregate over seeds into
a mean with a **bootstrap confidence interval** (reusing
:func:`verisim.metrics.aggregate.bootstrap_ci`, the same CI machinery EN1 uses). The deliverable is
a scaling curve with CIs; the bar is *disjoint* CIs that are stable or growing with scale (§7.1).

Pure orchestration: no torch here, so it imports cheaply and is property-testable without a model.
The two runners (:mod:`verisim.experiments.en8_scale`, :mod:`verisim.experiments.en9_scale`) own the
training; this owns the grid, the reduction, the CSV, and the plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class ModelSize:
    """One point on the model-size axis (SPEC-8 §7.1). ``label`` names it in the figure/CSV."""

    label: str
    d_model: int
    mp_rounds: int = 3
    n_layer: int = 2
    n_head: int = 2


@dataclass(frozen=True)
class GapStat:
    """A single (world_size, model, metric) cell reduced over seeds: mean + bootstrap CI."""

    world_size: int
    model_label: str
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world_size},{self.model_label},{self.metric},"
            f"{self.mean:.6f},{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


CSV_HEADER = "world_size,model_label,metric,mean,ci_lo,ci_hi,n"


def summarize(
    world_size: int, model_label: str, metric: str, values: list[float], *, seed: int = 0
) -> GapStat:
    """Reduce per-seed ``values`` for a cell to a :class:`GapStat` (mean + percentile bootstrap CI).

    Deterministic in ``seed`` (the bootstrap resampling seed). A single value collapses the CI to it
    (so a smoke, one-seed cell is still well-formed); the empty case is ``nan`` mean, matching
    :func:`bootstrap_ci`.
    """
    mean = fmean(values) if values else float("nan")
    lo, hi = bootstrap_ci(values, seed=seed)
    return GapStat(world_size, model_label, metric, mean, lo, hi, len(values))


def disjoint_from_zero(stat: GapStat) -> bool:
    """``True`` iff the CI excludes 0 - the "cannot be dismissed" test for a *gap* metric (§7.1)."""
    return (stat.ci_lo > 0.0) or (stat.ci_hi < 0.0)


def write_csv(stats: list[GapStat], path: Path) -> None:
    """Write the long-format scaling CSV (one row per cell), creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER, *(s.csv_row() for s in stats)]
    path.write_text("\n".join(lines) + "\n")


def plot_scaling_curves(
    stats: list[GapStat], metrics: list[str], path: Path, *, title: str, gap_metrics: set[str]
) -> None:  # pragma: no cover - plotting
    """One panel per metric: x = world size, a line per model size, shaded bootstrap-CI band.

    ``gap_metrics`` get a dashed ``y = 0`` reference line (the "no advantage" mark): a curve whose
    CI band sits clear of it is the undismissable result (§7.1).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    labels = sorted({s.model_label for s in stats})
    for ax, metric in zip(axes[0], metrics, strict=True):
        for label in labels:
            cells = sorted(
                (s for s in stats if s.metric == metric and s.model_label == label),
                key=lambda s: s.world_size,
            )
            if not cells:
                continue
            xs = [s.world_size for s in cells]
            ys = [s.mean for s in cells]
            lo = [s.ci_lo for s in cells]
            hi = [s.ci_hi for s in cells]
            (line,) = ax.plot(xs, ys, marker="o", label=label)
            ax.fill_between(xs, lo, hi, alpha=0.18, color=line.get_color())
        if metric in gap_metrics:
            ax.axhline(0.0, ls="--", lw=1, color="#555")
        ax.set_xlabel("world size (hosts)")
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
