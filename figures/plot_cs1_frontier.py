"""Plot the SPEC-21 CS1 load-bearing frontier (the scale law) from its committed CSV.

Two panels:

  - **left -- the load-bearing frontier**: the per-task faithful-vs-free gap vs model scale (params,
    log x), one line per task, ordered structure->content. The frontier is the contour where the gap
    crosses the load-bearing threshold; the claim (H87) is that it **recedes structural-first** with
    scale, with the deep-content task staying above threshold at every rung (H88, the irreducible
    residue).
  - **right -- the forecast**: the cheap per-task keyed drift vs the (expensive) gap, every (rung,
    task) cell -- they should fall on a line (H89: the cheap drift forecasts the load-bearing gap).

This reads the CPU-proven CSV; the committed *headline* frontier is the GPU run of the same pipeline
(SPEC-21 §5), but the apparatus and the figure are identical -- only the ladder widens.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def plot_cs1_frontier(  # pragma: no cover
    csv_path: str | Path, out_path: str | Path, *, threshold: float = 0.05,
    title: str = "SPEC-21 CS1 -- the faithfulness-for-control scale law (CPU-proven apparatus)",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    tasks: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"params": [], "gap": [], "drift": [], "order": []}
    )
    for r in rows:
        t = tasks[r["task"]]
        t["params"].append(float(r["params"]))
        t["gap"].append(float(r["gap"]))
        t["drift"].append(float(r["keyed_drift"]))
        t["order"].append(float(r["order"]))
    ordered = sorted(tasks.items(), key=lambda kv: kv[1]["order"][0])
    cmap = ["tab:green", "tab:olive", "tab:orange", "tab:red"]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))

    for (name, t), color in zip(ordered, cmap, strict=False):
        pairs = sorted(zip(t["params"], t["gap"], strict=True))
        xs = [p for p, _ in pairs]
        ys = [g for _, g in pairs]
        axl.plot(xs, ys, marker="o", color=color, label=name)
    axl.axhline(threshold, color="0.5", ls="--", alpha=0.7,
                label=f"load-bearing threshold ({threshold})")
    axl.set_xscale("log")
    axl.set_xlabel("model capacity (params, log scale)")
    axl.set_ylabel("faithful-vs-free gap (load-bearing signal)")
    axl.set_ylim(-0.05, 1.05)
    axl.set_title(
        "the load-bearing frontier (H87/H88):\ngap per task vs scale (structure->content)",
        fontsize=8,
    )
    axl.legend(fontsize=8, loc="upper right")
    axl.grid(alpha=0.3)

    for (name, t), color in zip(ordered, cmap, strict=False):
        axr.scatter(t["drift"], t["gap"], color=color, label=name, s=40)
    lim = max([*[g for _, t in ordered for g in t["gap"]], 0.1]) * 1.1
    axr.plot([0, lim], [0, lim], color="0.6", ls=":", alpha=0.7, label="y = x")
    axr.set_xlabel("cheap per-task keyed drift (the forecast)")
    axr.set_ylabel("faithful-vs-free gap (the expensive ablation)")
    axr.set_title("the forecast (H89):\ncheap drift predicts the load-bearing gap", fontsize=9)
    axr.legend(fontsize=8, loc="upper left")
    axr.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
