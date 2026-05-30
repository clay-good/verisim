"""Plot the E4 objective axis (supervised vs. +RLVR) from run-records (SPEC-2 §9, §17.4).

Reads the records written by ``verisim.experiments.objective``, aggregates clean
per-step accuracy and clean horizon by ``(objective, difficulty)`` with bootstrap CIs
(``verisim.metrics.aggregate_values``), writes the table as CSV, and renders clean
per-step accuracy as grouped bars (one group per difficulty, one bar per objective).
Figures are produced *only* from records (SPEC-2 §7.3, §12). Torch-free.

Usage:
    python figures/plot_objective.py --records runs/objective/records.jsonl \\
        --out figures/objective.png --csv figures/objective.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from verisim.metrics.aggregate import GroupStat, aggregate_values
from verisim.metrics.record import read_records

# Training arms in pipeline order (Stage 1 -> Stage 2) for a stable bar order.
_OBJECTIVE_ORDER = ["supervised", "rlvr"]


def _objective_rank(label: str) -> int:
    return _OBJECTIVE_ORDER.index(label) if label in _OBJECTIVE_ORDER else len(_OBJECTIVE_ORDER)


def write_csv(
    accuracy: list[GroupStat], horizon: list[GroupStat], path: str | Path
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    horizon_by_key = {h.key: h for h in horizon}
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "objective",
                "difficulty",
                "mean_accuracy",
                "acc_ci_low",
                "acc_ci_high",
                "mean_horizon",
                "horizon_ci_low",
                "horizon_ci_high",
                "n",
            ]
        )
        for a in accuracy:
            h = horizon_by_key[a.key]
            writer.writerow(
                [a.key[0], a.key[1], a.mean, a.ci_low, a.ci_high, h.mean, h.ci_low, h.ci_high, a.n]
            )
    return out


def plot_objective(stats: list[GroupStat], out_path: str | Path) -> Path:
    """Clean per-step accuracy as grouped bars: one group per difficulty, one bar per arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objectives = sorted({s.key[0] for s in stats}, key=_objective_rank)
    difficulties = sorted({s.key[1] for s in stats})
    by_key = {(s.key[0], s.key[1]): s for s in stats}

    fig, ax = plt.subplots(figsize=(7, 5))
    n_arms = len(objectives)
    width = 0.8 / max(1, n_arms)
    xs = list(range(len(difficulties)))
    for i, objective in enumerate(objectives):
        pts = [by_key.get((objective, difficulty)) for difficulty in difficulties]
        ys = [p.mean if p else float("nan") for p in pts]
        lo = [(p.mean - p.ci_low) if p else 0.0 for p in pts]
        hi = [(p.ci_high - p.mean) if p else 0.0 for p in pts]
        offsets = [x + (i - (n_arms - 1) / 2) * width for x in xs]
        ax.bar(offsets, ys, width=width, yerr=[lo, hi], capsize=3, label=objective)

    ax.set_xticks(xs)
    ax.set_xticklabels(difficulties)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("difficulty")
    ax.set_ylabel("clean per-step accuracy  (ρ=0, teacher-forced)")
    ax.set_title("Verisim E4 (objective) — supervised vs. +RLVR")
    ax.legend(title="objective", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the E4 objective axis (supervised/+RLVR).")
    parser.add_argument("--records", type=str, default="runs/objective/records.jsonl")
    parser.add_argument("--out", type=str, default="figures/objective.png")
    parser.add_argument("--csv", type=str, default="figures/objective.csv")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    records = read_records(args.records)
    accuracy = aggregate_values(
        records, group_keys=["objective", "difficulty"], value="step_accuracy",
        n_resamples=args.resamples, seed=0,
    )
    horizon = aggregate_values(
        records, group_keys=["objective", "difficulty"], value="clean_horizon",
        n_resamples=args.resamples, seed=0,
    )
    csv_path = write_csv(accuracy, horizon, args.csv)
    fig_path = plot_objective(accuracy, args.out)
    print(f"{len(records)} records -> {len(accuracy)} (objective,difficulty) cells")
    print(f"wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
