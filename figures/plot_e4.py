"""Plot the E4 size/difficulty ablation from run-records (SPEC-2 §9, §17.5).

Reads the records written by ``verisim.experiments.e4``, aggregates clean per-step
accuracy by ``(size, difficulty)`` with bootstrap CIs
(``verisim.metrics.aggregate_values``), writes the table as CSV, and renders accuracy
vs. model size with one line per difficulty. Figures are produced *only* from
records (SPEC-2 §7.3, §12). Torch-free.

Usage:
    python figures/plot_e4.py --records runs/e4/records.jsonl \\
        --out figures/e4_ablation.png --csv figures/e4_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from verisim.metrics.aggregate import GroupStat, aggregate_values
from verisim.metrics.record import RunRecord, read_records

# Model sizes in increasing-capacity order for the x-axis.
_SIZE_ORDER = ["tiny", "small", "medium", "large"]


def _size_rank(label: str) -> int:
    return _SIZE_ORDER.index(label) if label in _SIZE_ORDER else len(_SIZE_ORDER)


def write_csv(stats: list[GroupStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["size", "difficulty", "mean_accuracy", "ci_low", "ci_high", "n"])
        for s in stats:
            writer.writerow([s.key[0], s.key[1], s.mean, s.ci_low, s.ci_high, s.n])
    return out


def plot_ablation(records: list[RunRecord], stats: list[GroupStat], out_path: str | Path) -> Path:
    """Clean per-step accuracy vs. model size, one line per difficulty, CI bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({s.key[0] for s in stats}, key=_size_rank)
    difficulties = sorted({s.key[1] for s in stats})
    by_key = {(s.key[0], s.key[1]): s for s in stats}

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = list(range(len(sizes)))
    for difficulty in difficulties:
        pts = [by_key.get((size, difficulty)) for size in sizes]
        ys = [p.mean if p else float("nan") for p in pts]
        lo = [(p.mean - p.ci_low) if p else 0.0 for p in pts]
        hi = [(p.ci_high - p.mean) if p else 0.0 for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=difficulty)

    ax.set_xticks(xs)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("model size")
    ax.set_ylabel("clean per-step accuracy  (ρ=0, teacher-forced)")
    ax.set_title("Verisim E4 — capacity vs. clean faithfulness")
    ax.legend(title="difficulty", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the E4 size/difficulty ablation.")
    parser.add_argument("--records", type=str, default="runs/e4/records.jsonl")
    parser.add_argument("--out", type=str, default="figures/e4_ablation.png")
    parser.add_argument("--csv", type=str, default="figures/e4_ablation.csv")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    records = read_records(args.records)
    stats = aggregate_values(
        records,
        group_keys=["size", "difficulty"],
        value="step_accuracy",
        n_resamples=args.resamples,
        seed=0,
    )
    csv_path = write_csv(stats, args.csv)
    fig_path = plot_ablation(records, stats, args.out)
    print(f"{len(records)} records -> {len(stats)} (size,difficulty) cells")
    print(f"wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
