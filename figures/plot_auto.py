"""Plot the autoresearch ratchet from its trial log (SPEC-2 §17.5 automation).

Reads the per-trial records written by ``verisim.auto.search``, writes a CSV of the
trial table, and renders the ratchet: each trial's oracle-gated score (kept trials
filled, rejected hollow) with the monotone best-so-far envelope over the top.
Figures are produced *only* from records (SPEC-2 §7.3, §12). Torch-free.

Usage:
    python figures/plot_auto.py --records runs/auto/log.jsonl \\
        --out figures/auto_search.png --csv figures/auto_search.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from verisim.metrics.record import RunRecord, read_records


def write_csv(records: list[RunRecord], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: int(r.config["trial"]))
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["trial", "score", "best_score", "kept", "n_layer", "n_embd", "train_iters",
             "train_steps_per_traj", "lr"]
        )
        for r in rows:
            c = r.config
            writer.writerow(
                [c["trial"], c["score"], c["best_score"], c["kept"], c["n_layer"], c["n_embd"],
                 c["train_iters"], c["train_steps_per_traj"], c["lr"]]
            )
    return out


def plot_auto(records: list[RunRecord], out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(records, key=lambda r: int(r.config["trial"]))
    trials = [int(r.config["trial"]) for r in rows]
    scores = [float(r.config["score"]) for r in rows]
    best = [float(r.config["best_score"]) for r in rows]
    kept = [bool(r.config["kept"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(trials, best, where="post", color="C0", label="best so far (ratchet)")
    keep_x = [t for t, k in zip(trials, kept, strict=True) if k]
    keep_y = [s for s, k in zip(scores, kept, strict=True) if k]
    rej_x = [t for t, k in zip(trials, kept, strict=True) if not k]
    rej_y = [s for s, k in zip(scores, kept, strict=True) if not k]
    ax.scatter(keep_x, keep_y, color="C2", zorder=3, label="trial kept")
    ax.scatter(rej_x, rej_y, facecolors="none", edgecolors="C3", zorder=3, label="trial rejected")

    ax.set_xlabel("trial")
    ax.set_ylabel("clean per-step accuracy  (ρ=0, oracle-gated)")
    ax.set_title("Verisim autoresearch ratchet — keep-if-better vs. the oracle")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the autoresearch config ratchet.")
    parser.add_argument("--records", type=str, default="runs/auto/log.jsonl")
    parser.add_argument("--out", type=str, default="figures/auto_search.png")
    parser.add_argument("--csv", type=str, default="figures/auto_search.csv")
    args = parser.parse_args()

    records = read_records(args.records)
    csv_path = write_csv(records, args.csv)
    fig_path = plot_auto(records, args.out)
    best = max(float(r.config["score"]) for r in records)
    print(f"{len(records)} trials -> best score {best:.4f}")
    print(f"wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
