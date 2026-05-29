"""Plot the E1 ``H_ε(ρ)`` curve from run-records (SPEC-2 §7.3, §12, milestone M6).

Figures are produced *only* from run-records: this script reads the JSONL records
written by ``verisim.experiments.e1``, aggregates them into curve points with
bootstrap CIs (``verisim.metrics.aggregate``), writes the curve table as CSV (the
numbers behind the figure), and renders the figure with matplotlib's Agg backend.

Usage:
    python figures/plot_e1.py --records runs/e1/records.jsonl \\
        --out figures/e1_curve.png --csv figures/e1_curve.csv

Both the plotting script and the record ids are committed next to the figure, so
the figure is reproducible (SPEC-2 §12).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from verisim.metrics.aggregate import CurvePoint, aggregate_curve
from verisim.metrics.record import read_records


def write_csv(points: list[CurvePoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["difficulty", "epsilon", "rho", "mean_h", "ci_low", "ci_high", "n"])
        for p in points:
            writer.writerow([p.difficulty, p.epsilon, p.rho, p.mean_h, p.ci_low, p.ci_high, p.n])
    return out


def plot_curve(
    points: list[CurvePoint], out_path: str | Path, *, title: str = "Verisim E1"
) -> Path:
    """Render H_ε(ρ) with one line per (difficulty, ε) and shaded bootstrap CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[tuple[str, float], list[CurvePoint]] = {}
    for p in points:
        series.setdefault((p.difficulty, p.epsilon), []).append(p)

    fig, ax = plt.subplots(figsize=(7, 5))
    for (difficulty, epsilon), pts in sorted(series.items()):
        pts = sorted(pts, key=lambda p: p.rho)
        xs = [p.rho for p in pts]
        ys = [p.mean_h for p in pts]
        lo = [p.ci_low for p in pts]
        hi = [p.ci_high for p in pts]
        line = ax.plot(xs, ys, marker="o", label=f"{difficulty}, ε={epsilon}")[0]
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())

    ax.set_xlabel("oracle-consultation budget  ρ")
    ax.set_ylabel("faithful horizon  H_ε  (steps)")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the E1 H_eps(rho) curve.")
    parser.add_argument("--records", type=str, default="runs/e1/records.jsonl")
    parser.add_argument("--out", type=str, default="figures/e1_curve.png")
    parser.add_argument("--csv", type=str, default="figures/e1_curve.csv")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    records = read_records(args.records)
    points = aggregate_curve(records, n_resamples=args.resamples, seed=0)
    csv_path = write_csv(points, args.csv)
    fig_path = plot_curve(points, args.out)
    print(f"{len(records)} records -> {len(points)} curve points")
    print(f"wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
