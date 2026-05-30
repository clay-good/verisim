"""Plot the uncertainty-calibration reliability curve from pairs (SPEC-2 §7.2, §17.2).

Reads the ``(signal, divergence)`` pairs JSONL written by
``verisim.experiments.calibration``, computes the calibration report
(``verisim.metrics.calibration_report``), writes the reliability table as CSV, and
renders the reliability curve (binned mean signal vs. mean divergence) with the
Pearson/Spearman correlations in the title. Figures are produced *only* from the
committed pairs (SPEC-2 §7.3, §12). Torch-free: reads the JSONL directly.

Usage:
    python figures/plot_calibration.py --pairs runs/calibration/pairs.jsonl \\
        --out figures/calibration.png --csv figures/calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from verisim.metrics.calibration import CalibrationReport, Pair, calibration_report


def read_pairs(path: str | Path) -> list[Pair]:
    text = Path(path).read_text().strip()
    pairs: list[Pair] = []
    for line in text.splitlines():
        if line:
            r = json.loads(line)
            pairs.append((float(r["signal"]), float(r["divergence"])))
    return pairs


def write_csv(report: CalibrationReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lo", "hi", "mean_signal", "mean_divergence", "n"])
        for b in report.bins:
            writer.writerow([b.lo, b.hi, b.mean_signal, b.mean_divergence, b.n])
        writer.writerow([])
        writer.writerow(["pearson", "spearman", "n", "mean_signal", "mean_divergence"])
        writer.writerow(
            [report.pearson, report.spearman, report.n, report.mean_signal, report.mean_divergence]
        )
    return out


def plot_calibration(report: CalibrationReport, out_path: str | Path) -> Path:
    """Reliability curve: binned mean uncertainty signal vs. mean divergence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [b.mean_signal for b in report.bins]
    ys = [b.mean_divergence for b in report.bins]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("model uncertainty signal  (mean decode entropy, nats)")
    ax.set_ylabel("actual per-step divergence  d")
    ax.set_title(
        f"Verisim uncertainty calibration "
        f"(pearson={report.pearson:.2f}, spearman={report.spearman:.2f}, n={report.n})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the uncertainty-calibration curve.")
    parser.add_argument("--pairs", type=str, default="runs/calibration/pairs.jsonl")
    parser.add_argument("--out", type=str, default="figures/calibration.png")
    parser.add_argument("--csv", type=str, default="figures/calibration.csv")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    report = calibration_report(read_pairs(args.pairs), n_bins=args.bins)
    csv_path = write_csv(report, args.csv)
    fig_path = plot_calibration(report, args.out)
    print(f"{report.n} pairs -> {len(report.bins)} bins; wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
