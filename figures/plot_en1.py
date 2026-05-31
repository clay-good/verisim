"""Plot the EN1 network ``H_ε(ρ)`` curve from run-records (SPEC-5 §12, NW6).

The network analogue of ``plot_e1.py`` and **SPEC-5's headline figure** (§0). Figures are
produced *only* from run-records: this reads the JSONL written by ``verisim.experiments.en1``,
aggregates into curve points with bootstrap CIs (``verisim.metrics.aggregate``, reused from
v0), writes the curve table as CSV, and renders the figure with matplotlib's Agg backend.

Usage:
    python figures/plot_en1.py --records runs/en1/records.jsonl \\
        --out figures/en1_curve.png --csv figures/en1_curve.csv
"""

from __future__ import annotations

import argparse

# Reuse v0's CSV writer and renderer verbatim -- the record schema is identical.
from plot_e1 import plot_curve, write_csv

from verisim.metrics.aggregate import aggregate_curve
from verisim.metrics.record import read_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the EN1 network H_eps(rho) curve.")
    parser.add_argument("--records", type=str, default="runs/en1/records.jsonl")
    parser.add_argument("--out", type=str, default="figures/en1_curve.png")
    parser.add_argument("--csv", type=str, default="figures/en1_curve.csv")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    records = read_records(args.records)
    points = aggregate_curve(records, n_resamples=args.resamples, seed=0)
    csv_path = write_csv(points, args.csv)
    fig_path = plot_curve(points, args.out, title="Verisim EN1 — network H_ε(ρ)")
    print(f"{len(records)} records -> {len(points)} curve points")
    print(f"wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
