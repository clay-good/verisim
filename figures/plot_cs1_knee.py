"""Plot the SPEC-21 knee trajectory -- the *cost* dimension of the scale law -- from the CS1 CSV.

The load-bearing frontier (`cs1_loadbearing_frontier.png`) says *where* faithfulness is load-bearing
as the model scales. This companion figure says *how expensive it is to buy back*: the useful-knee ρ
(the smallest consultation budget that recovers the faithful catch, UA9/H81) per load-bearing
task vs model capacity. The read: does the irreducible content residue stay cheaply buyable as
the model scales (a flat/low knee), or get more expensive (a rising knee -- the residue *doubly*
hard, load-bearing AND costly)? Reads the committed CS1 CSV (knee_rho; -1 = not load-bearing).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def plot_cs1_knee(csv_path: str | Path, out_path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    tasks: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"params": [], "knee": [], "order": []}
    )
    for r in rows:
        knee = float(r["knee_rho"])
        if knee < 0:  # not load-bearing at this rung -> no knee
            continue
        t = tasks[r["task"]]
        t["params"].append(float(r["params"]))
        t["knee"].append(knee)
        t["order"].append(float(r["order"]))
    ordered = sorted(tasks.items(), key=lambda kv: kv[1]["order"][0])
    cmap = {"process": "tab:green", "fd-con": "tab:olive", "file-i": "tab:orange",
            "conten": "tab:red"}

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for name, t in ordered:
        pairs = sorted(zip(t["params"], t["knee"], strict=True))
        xs = [p for p, _ in pairs]
        ys = [k for _, k in pairs]
        color = cmap.get(name[:6], "tab:blue")
        ax.plot(xs, ys, marker="o", color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("model capacity (params, log scale)")
    ax.set_ylabel("useful-knee ρ  (consultation budget to buy back faithfulness)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "SPEC-21 — the cost dimension of the scale law:\n"
        "how expensive it is to buy back faithfulness, per load-bearing task vs scale",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left", title="load-bearing task")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
