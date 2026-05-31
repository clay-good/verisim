"""Plot the K0 diagnostics + control from run-records (SPEC-2.1 §4).

Two panels, from the records written by ``verisim.experiments.k0``:

  - **left** — the headline contrast: the baseline (under-trained) model's clean held-out
    accuracy vs. the *trivial-world control* trained properly, with the 0.95 gate line. The
    control clearing the gate while the baseline sits on the floor is the K0 evidence that
    the floor is a data/training problem, not a capacity or world-difficulty one.
  - **right** — per-command accuracy of the baseline (where it fails): which commands the
    under-trained model gets right vs. wrong.

Figures are produced *only* from records (SPEC-2 §7.3, §12). Torch-free.

Usage:
    python figures/plot_k0.py --records runs/k0/records.jsonl \\
        --out figures/k0_control.png --csv figures/k0_control.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from verisim.metrics.record import read_records


def _split(records: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    control: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for r in records:
        if r.config.get("part") == "control":
            control = r.config
        elif r.config.get("part") == "diagnostics":
            diagnostics = r.config
    return control, diagnostics


def _per_command_accuracy(diagnostics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cmd, pair in diagnostics.get("per_command", {}).items():
        correct, total = pair
        if total:
            out[cmd] = correct / total
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def write_csv(control: dict[str, Any], diagnostics: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["trivial_control_exact", control.get("clean_faithfulness_exact")])
        writer.writerow(["trivial_control_graded", control.get("clean_faithfulness_graded")])
        writer.writerow(["gate", control.get("gate")])
        writer.writerow(["gate_passed", control.get("gate_passed")])
        writer.writerow(["n_train_transitions", control.get("n_train_transitions")])
        writer.writerow(["baseline_val_accuracy", diagnostics.get("baseline_val_accuracy")])
        writer.writerow(["baseline_train_accuracy", diagnostics.get("baseline_train_accuracy")])
        writer.writerow(
            ["baseline_mean_bits_to_correct", diagnostics.get("baseline_mean_bits_to_correct")]
        )
        writer.writerow(["command", "accuracy"])
        for cmd, acc in _per_command_accuracy(diagnostics).items():
            writer.writerow([cmd, acc])
    return out


def plot_k0(control: dict[str, Any], diagnostics: dict[str, Any], out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate = float(control.get("gate", 0.95))
    baseline_val = float(diagnostics.get("baseline_val_accuracy", 0.0))
    baseline_train = float(diagnostics.get("baseline_train_accuracy", 0.0))
    control_exact = float(control.get("clean_faithfulness_exact", 0.0))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ["baseline\n(val)", "baseline\n(train)", "trivial control\n(held-out)"]
    vals = [baseline_val, baseline_train, control_exact]
    colors = ["#bbbbbb", "#888888", "#2a7"]
    ax_l.bar(bars, vals, color=colors)
    ax_l.axhline(gate, color="crimson", linestyle="--", label=f"gate = {gate:g}")
    ax_l.set_ylim(0.0, 1.0)
    ax_l.set_ylabel("clean per-step faithfulness (ρ=0, exact)")
    ax_l.set_title("K0 control — can the pipeline learn?")
    ax_l.legend(fontsize="small")
    for i, v in enumerate(vals):
        ax_l.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize="small")
    ax_l.grid(True, axis="y", alpha=0.3)

    per_cmd = _per_command_accuracy(diagnostics)
    if per_cmd:
        cmds = list(per_cmd)
        ax_r.barh(cmds, [per_cmd[c] for c in cmds], color="#69c")
        ax_r.set_xlim(0.0, 1.0)
        ax_r.set_xlabel("accuracy")
        ax_r.set_title("K0 diagnostics — baseline per-command accuracy")
        ax_r.grid(True, axis="x", alpha=0.3)
    else:
        ax_r.set_visible(False)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the K0 control + diagnostics.")
    parser.add_argument("--records", type=str, default="runs/k0/records.jsonl")
    parser.add_argument("--out", type=str, default="figures/k0_control.png")
    parser.add_argument("--csv", type=str, default="figures/k0_control.csv")
    args = parser.parse_args()

    records = read_records(args.records)
    control, diagnostics = _split(records)
    csv_path = write_csv(control, diagnostics, args.csv)
    fig_path = plot_k0(control, diagnostics, args.out)
    print(f"{len(records)} records -> {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
