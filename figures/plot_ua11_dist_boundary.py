"""Plot the SPEC-20 UA11 distributed boundary — the structure/content law on the third world.

One panel from the committed CSV: the faithful-vs-free gap for the two distributed tasks —
`partition-control` (structure) and `value-integrity` (content) — with bootstrap CIs. The boundary
holds iff the content gap materially exceeds the structure gap: faithfulness is more load-bearing
where the model drifts (the content / replicated values), not on the structure it learns (the
partition topology). This completes the cross-world boundary law on host + network + distributed.
"""

from __future__ import annotations

import csv
from pathlib import Path


def plot_ua11_dist_boundary(csv_path: str | Path, out_path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: 0 if r["band"] == "structure" else 1)
    labels = [f"{r['task']}\n({r['band']})" for r in rows]
    gaps = [float(r["gap"]) for r in rows]
    lo = [float(r["gap"]) - float(r["ci_lo"]) for r in rows]
    hi = [float(r["ci_hi"]) - float(r["gap"]) for r in rows]
    colors = ["tab:green" if r["band"] == "structure" else "tab:red" for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.bar(labels, gaps, yerr=[lo, hi], color=colors, capsize=5, alpha=0.85, width=0.55)
    ax.axhline(0.0, color="0.5", lw=0.8)
    for i, g in enumerate(gaps):
        ax.text(i, g + 0.02, f"{g:+.2f}", ha="center", fontsize=10)
    ax.set_ylabel("faithful-vs-free gap (load-bearing signal)")
    ax.set_ylim(min(0.0, *gaps) - 0.05, max(gaps) + 0.15)
    ax.set_title(
        "SPEC-20 UA11 — the structure/content boundary on the DISTRIBUTED world (3rd world):\n"
        "faithfulness is load-bearing on the content (values) the model drifts on, not structure",
        fontsize=9,
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
