"""Plot the K1+K2 result from run-records (SPEC-2.1 §5-6).

Two panels, from the records written by ``verisim.experiments.k2``:

  - **left** — clean (ρ=0) per-step faithfulness on the held-out non-trivial difficulty
    (exact / acceptance@ε / graded), with the **0.5 acceptance-floor gate** line. Clearing it
    on a non-trivial world is what makes a knee possible (K3/K4).
  - **right** — the K1 coverage of the transition space: the create-depth histogram (the
    copy-distribution axis K0 flagged), annotated with the number of failure cells covered.

Figures are produced *only* from records (SPEC-2 §7.3, §12). Torch-free.

Usage:
    python figures/plot_k2.py --records runs/k2/records.jsonl \\
        --out figures/k2_faithfulness.png --csv figures/k2_faithfulness.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from verisim.metrics.record import read_records


def _split(records: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    coverage: dict[str, Any] = {}
    faith: dict[str, Any] = {}
    for r in records:
        if r.config.get("part") == "coverage":
            coverage = r.config
        elif r.config.get("part") == "faithfulness":
            faith = r.config
    return coverage, faith


def write_csv(coverage: dict[str, Any], faith: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ("eval_driver", "exact", "acceptance", "graded", "epsilon", "gate",
                    "gate_passed", "n_train_transitions", "train_steps", "hard_negative_rounds"):
            writer.writerow([key, faith.get(key)])
        writer.writerow(["n_failure_cells", coverage.get("n_failures")])
        writer.writerow(["depth", "create_count"])
        depths = coverage.get("create_depths", {})
        for depth in sorted(depths, key=int):
            writer.writerow([depth, depths[depth]])
    return out


def plot_k2(coverage: dict[str, Any], faith: dict[str, Any], out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate = float(faith.get("gate", 0.5))
    eps = faith.get("epsilon", 0.05)
    labels = ["exact", f"acc@{eps}", "graded"]
    vals = [float(faith.get("exact", 0.0)), float(faith.get("acceptance", 0.0)),
            float(faith.get("graded", 0.0))]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2a7" if v > gate else "#c55" for v in vals]
    ax_l.bar(labels, vals, color=colors)
    ax_l.axhline(gate, color="crimson", linestyle="--", label=f"gate = {gate:g}")
    ax_l.set_ylim(0.0, 1.0)
    ax_l.set_ylabel("clean per-step faithfulness (ρ=0)")
    ax_l.set_title(f"K2 — faithfulness on '{faith.get('eval_driver', '?')}' (non-trivial)")
    ax_l.legend(fontsize="small")
    for i, v in enumerate(vals):
        ax_l.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize="small")
    ax_l.grid(True, axis="y", alpha=0.3)

    depths = coverage.get("create_depths", {})
    if depths:
        ks = sorted(depths, key=int)
        ax_r.bar([str(k) for k in ks], [depths[k] for k in ks], color="#69c")
        ax_r.set_xlabel("created-path depth (segments)")
        ax_r.set_ylabel("count")
        ax_r.set_title(
            f"K1 coverage — create-depth distribution "
            f"({coverage.get('n_failures', 0)} failure cells)"
        )
        ax_r.grid(True, axis="y", alpha=0.3)
    else:
        ax_r.set_visible(False)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the K1+K2 coverage + faithfulness result.")
    parser.add_argument("--records", type=str, default="runs/k2/records.jsonl")
    parser.add_argument("--out", type=str, default="figures/k2_faithfulness.png")
    parser.add_argument("--csv", type=str, default="figures/k2_faithfulness.csv")
    args = parser.parse_args()

    records = read_records(args.records)
    coverage, faith = _split(records)
    csv_path = write_csv(coverage, faith, args.csv)
    fig_path = plot_k2(coverage, faith, args.out)
    print(f"{len(records)} records -> {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
