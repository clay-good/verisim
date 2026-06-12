"""Plot the SPEC-20/21 distributed recession — is the structural-first recession universal?

A refinement of the SPEC-21 scale law (H87). On host/network the load-bearing frontier recedes
*structural-first* — but there the structure gap was already ~0 at every scale, so that was free.
The distributed world is the test: here even the partition *structure* is hard to learn (a non-zero
gap at small scale), so we can watch whether the structure gap recedes to ~0 (structural-first) or
*persists*. This figure plots both gaps vs scale from the committed CSV; the robust finding
is that the structure gap **persists** (~0.2 at the top rung, not ~0 like host/network) — the
partition structure under the in-flight/fault medium is *itself* hard to learn. So the structural-
first recession is *not universal*: it needs a world where structure is trivially learnable.
"""

from __future__ import annotations

import csv
from pathlib import Path


def plot_ua11_dist_recession(  # pragma: no cover
    csv_path: str | Path, out_path: str | Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r["params"]))
    params = [float(r["params"]) for r in rows]
    structure = [float(r["structure_gap"]) for r in rows]
    content = [float(r["content_gap"]) for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(params, structure, marker="o", color="tab:green", label="partition-control (structure)")
    ax.plot(params, content, marker="o", color="tab:red", label="value-integrity (content)")
    ax.set_xscale("log")
    ax.set_xlabel("distributed model capacity (params, log scale)")
    ax.set_ylabel("faithful-vs-free gap (load-bearing signal)")
    ax.set_ylim(0.0, max(content + structure) + 0.1)
    ax.set_title(
        "SPEC-20/21 — is the structural-first recession universal? (the distributed test)\n"
        "NO: the structure gap persists (~0.2, not ~0 like host/net) — partition structure is hard",
        fontsize=8,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
