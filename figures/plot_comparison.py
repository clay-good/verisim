"""Plot an equal-budget comparison (E2 policies / E3 operators) from run-records.

Mirrors ``plot_e1.py``: reads the JSONL records written by
``verisim.experiments.e2`` / ``.e3``, aggregates them into one bar per compared
variant with a bootstrap CI (``verisim.metrics.aggregate.aggregate_comparison``),
writes the table as CSV (the numbers behind the figure), and renders a bar chart
with matplotlib's Agg backend. Figures are produced *only* from records (SPEC-2
§7.3, §12).

Usage:
    python figures/plot_comparison.py --records runs/e2/records.jsonl --key policy \\
        --out figures/e2_policies.png --csv figures/e2_policies.csv
    python figures/plot_comparison.py --records runs/e3/records.jsonl --key operator \\
        --out figures/e3_operators.png --csv figures/e3_operators.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from verisim.metrics.aggregate import ComparisonPoint, aggregate_comparison
from verisim.metrics.record import read_records


def write_csv(points: list[ComparisonPoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "epsilon", "mean_h", "ci_low", "ci_high", "mean_calls", "n"])
        for p in points:
            writer.writerow(
                [p.label, p.epsilon, p.mean_h, p.ci_low, p.ci_high, p.mean_calls, p.n]
            )
    return out


def plot_comparison(
    points: list[ComparisonPoint], out_path: str | Path, *, key: str, title: str
) -> Path:
    """Render H_ε by variant as grouped bars (one group per ε) with CI error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted({p.label for p in points})
    epsilons = sorted({p.epsilon for p in points})
    by_key = {(p.label, p.epsilon): p for p in points}

    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(labels):
        xs = [j + i * width for j in range(len(epsilons))]
        ys = [by_key[(label, e)].mean_h if (label, e) in by_key else 0.0 for e in epsilons]
        lo = [by_key[(label, e)].mean_h - by_key[(label, e)].ci_low for e in epsilons]
        hi = [by_key[(label, e)].ci_high - by_key[(label, e)].mean_h for e in epsilons]
        ax.bar(xs, ys, width=width, label=label, yerr=[lo, hi], capsize=3)

    ax.set_xticks([j + width * (len(labels) - 1) / 2 for j in range(len(epsilons))])
    ax.set_xticklabels([f"ε={e}" for e in epsilons])
    ax.set_ylabel("faithful horizon  H_ε  (steps)")
    ax.set_title(title)
    ax.legend(title=key, fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an equal-budget comparison (E2/E3).")
    parser.add_argument("--records", type=str, required=True)
    parser.add_argument("--key", type=str, choices=["policy", "operator"], required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    records = read_records(args.records)
    points = aggregate_comparison(records, key=args.key, n_resamples=args.resamples, seed=0)
    csv_path = write_csv(points, args.csv)
    fig_path = plot_comparison(
        points, args.out, key=args.key, title=f"Verisim comparison by {args.key}"
    )
    print(f"{len(records)} records -> {len(points)} {args.key} points")
    print(f"wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
