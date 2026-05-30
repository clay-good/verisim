"""Plot the E4 representation axis (delta vs. full-state) from run-records (SPEC-2 §9, §10).

Reads the records written by ``verisim.experiments.representation``, aggregates clean
per-step accuracy and clean horizon by ``(representation, difficulty)`` with bootstrap
CIs (``verisim.metrics.aggregate_values``), writes the table as CSV, and renders clean
per-step accuracy as grouped bars (one group per difficulty, one bar per representation).
Figures are produced *only* from records (SPEC-2 §7.3, §12). Torch-free.

Usage:
    python figures/plot_representation.py --records runs/representation/records.jsonl \\
        --out figures/representation.png --csv figures/representation.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from verisim.metrics.aggregate import GroupStat, aggregate_values
from verisim.metrics.record import read_records

# Representations in spec order (primary target first) for a stable bar order.
_REPRESENTATION_ORDER = ["delta", "full_state"]


def _representation_rank(label: str) -> int:
    if label in _REPRESENTATION_ORDER:
        return _REPRESENTATION_ORDER.index(label)
    return len(_REPRESENTATION_ORDER)


def write_csv(accuracy: list[GroupStat], horizon: list[GroupStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    horizon_by_key = {h.key: h for h in horizon}
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "representation",
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


def plot_representation(stats: list[GroupStat], out_path: str | Path) -> Path:
    """Clean per-step accuracy as grouped bars: one group per difficulty, one bar per arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    representations = sorted({s.key[0] for s in stats}, key=_representation_rank)
    difficulties = sorted({s.key[1] for s in stats})
    by_key = {(s.key[0], s.key[1]): s for s in stats}

    fig, ax = plt.subplots(figsize=(7, 5))
    n_arms = len(representations)
    width = 0.8 / max(1, n_arms)
    xs = list(range(len(difficulties)))
    for i, representation in enumerate(representations):
        pts = [by_key.get((representation, difficulty)) for difficulty in difficulties]
        ys = [p.mean if p else float("nan") for p in pts]
        lo = [(p.mean - p.ci_low) if p else 0.0 for p in pts]
        hi = [(p.ci_high - p.mean) if p else 0.0 for p in pts]
        offsets = [x + (i - (n_arms - 1) / 2) * width for x in xs]
        ax.bar(offsets, ys, width=width, yerr=[lo, hi], capsize=3, label=representation)

    ax.set_xticks(xs)
    ax.set_xticklabels(difficulties)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("difficulty")
    ax.set_ylabel("clean per-step accuracy  (ρ=0, teacher-forced)")
    ax.set_title("Verisim E4 (representation) — delta vs. full-state")
    ax.legend(title="representation", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the E4 representation axis (delta vs full-state)."
    )
    parser.add_argument("--records", type=str, default="runs/representation/records.jsonl")
    parser.add_argument("--out", type=str, default="figures/representation.png")
    parser.add_argument("--csv", type=str, default="figures/representation.csv")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    records = read_records(args.records)
    accuracy = aggregate_values(
        records, group_keys=["representation", "difficulty"], value="step_accuracy",
        n_resamples=args.resamples, seed=0,
    )
    horizon = aggregate_values(
        records, group_keys=["representation", "difficulty"], value="clean_horizon",
        n_resamples=args.resamples, seed=0,
    )
    csv_path = write_csv(accuracy, horizon, args.csv)
    fig_path = plot_representation(accuracy, args.out)
    print(f"{len(records)} records -> {len(accuracy)} (representation,difficulty) cells")
    print(f"wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
