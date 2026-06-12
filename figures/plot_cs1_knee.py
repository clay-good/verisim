"""Plot the SPEC-21 cost dimension of the scale law -- the knee trajectory + the cost forecast.

The load-bearing frontier (`cs1_loadbearing_frontier.png`) says *where* faithfulness is load-bearing
as the model scales. This companion figure is the *cost* dimension -- *how expensive it is to buy
back* -- in two panels:

  - **left -- the knee trajectory**: the useful-knee ρ (the smallest consultation budget recovering
    the faithful catch, UA9/H81) per load-bearing task vs model capacity. The read: does the
    irreducible content residue stay cheaply buyable as the model scales? (On the fine ρ grid
    it is flat at ρ≈0.25 -- cheaply *and stably* buyable, the cost not growing with scale.)
  - **right -- the cost forecast (H89 extended)**: the cheap per-task keyed drift vs the knee, on
    the load-bearing cells. H89 showed the cheap drift forecasts the *gap* (Spearman +0.965); this
    asks whether it also forecasts the *cost* (the knee) -- it does, at +0.717.

Reads the committed CS1 CSV (knee_rho; -1 = not load-bearing, omitted).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def plot_cs1_knee(csv_path: str | Path, out_path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from verisim.metrics.calibration import spearman

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    tasks: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"params": [], "knee": [], "order": []}
    )
    drifts: list[float] = []
    knees: list[float] = []
    for r in rows:
        knee = float(r["knee_rho"])
        if knee < 0:  # not load-bearing at this rung -> no knee
            continue
        t = tasks[r["task"]]
        t["params"].append(float(r["params"]))
        t["knee"].append(knee)
        t["order"].append(float(r["order"]))
        drifts.append(float(r["keyed_drift"]))
        knees.append(knee)
    ordered = sorted(tasks.items(), key=lambda kv: kv[1]["order"][0])
    cmap = {"process": "tab:green", "fd-con": "tab:olive", "file-i": "tab:orange",
            "conten": "tab:red"}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: the knee trajectory (cost vs scale) ---
    for name, t in ordered:
        pairs = sorted(zip(t["params"], t["knee"], strict=True))
        xs = [p for p, _ in pairs]
        ys = [k for _, k in pairs]
        axl.plot(xs, ys, marker="o", color=cmap.get(name[:6], "tab:blue"), label=name)
    axl.set_xscale("log")
    axl.set_xlabel("model capacity (params, log scale)")
    axl.set_ylabel("useful-knee ρ  (budget to buy back faithfulness)")
    axl.set_ylim(0.0, 1.0)
    axl.set_title("the knee trajectory: cost to buy back faithfulness vs scale\n"
                  "(deep residue flat at ρ≈0.25 — cheaply & stably buyable)", fontsize=9)
    axl.legend(fontsize=8, loc="upper left", title="load-bearing task")
    axl.grid(alpha=0.3)

    # --- right: the cost forecast (cheap drift -> knee, H89 extended) ---
    for name, _t in ordered:
        idx = [i for i, r in enumerate(rows)
               if r["task"] == name and float(r["knee_rho"]) >= 0]
        axr.scatter([float(rows[i]["keyed_drift"]) for i in idx],
                    [float(rows[i]["knee_rho"]) for i in idx],
                    color=cmap.get(name[:6], "tab:blue"), label=name, s=40)
    rho = spearman(drifts, knees) if len(drifts) >= 2 else 0.0
    axr.set_xlabel("cheap per-task keyed drift (the forecast)")
    axr.set_ylabel("useful-knee ρ  (the cost, from the ρ-sweep)")
    axr.set_title(f"the cost forecast (H89 extended): cheap drift -> knee\n"
                  f"Spearman = {rho:+.3f} (the gap forecast is +0.965)", fontsize=9)
    axr.legend(fontsize=8, loc="upper left", title="load-bearing task")
    axr.grid(alpha=0.3)

    fig.suptitle("SPEC-21 — the cost dimension of the scale law", fontsize=11)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
