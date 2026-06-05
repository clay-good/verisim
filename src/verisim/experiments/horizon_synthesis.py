"""SPEC-10 cross-proposer synthesis -- the floor is *proposer-dependent* (the HS arc's capstone).

The whole SPEC-10 arc reduces to one contrast the free oracle made measurable, and this module
draws it in one figure. Sweeping the *same* capacity axis, the **flat** transformer's free-running
faithful horizon `H_free = H_ε(ρ=0)` **lifts ~9× with capacity** (HS1) and its floor dissolves into
a resourcing story across capacity, data, and world size (HS1/HS1.2/HS2). The **structured**
GNN+RSSM graph arm -- the proposer that *beats* the flat arm on one-step delta-exact (EN4) -- shows
the opposite: its `H_free` is **pinned at the floor (≈ 0, η < 1)** and moves with *neither* capacity
(HS3 incr 1) *nor* data (incr 2) *nor* world size (incr 3). So the program's standing question --
*"is the floor+cliff a resourcing artifact?"* -- has **no single answer: it depends on the
proposer.** For the flat arm it is under-resourcing; for the structured arm it is a genuine
compounding ceiling at this world's exact tolerance.

Like [`synthesis`](./synthesis.py) (the cross-*world* floor+cliff), this is a
**figures-from-records** synthesis (SPEC-2 §7.3): it re-reads the two committed capacity-sweep CSVs
(the flat `horizon_scaling` and the structured `horizon_graph_scaling`) and overlays them on the
shared `params` axis. It re-runs nothing; it is pure and dependency-free.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: The committed capacity-sweep CSVs, by proposer (the flat HS1 arm, the structured HS3 arm).
DEFAULT_PROPOSERS: dict[str, str] = {
    "flat transformer (HS1)": "figures/horizon_scaling.csv",
    "graph GNN+RSSM (HS3)": "figures/horizon_graph_scaling.csv",
}


@dataclass(frozen=True)
class ProposerPoint:
    """One (proposer, params) cell: the free-running horizon (id + ood) and one-step accuracy."""

    proposer: str
    params: int
    h_free_id: float
    h_free_ood: float
    p_id: float

    def csv_row(self) -> str:
        return (
            f"{self.proposer},{self.params},"
            f"{self.h_free_id:.4f},{self.h_free_ood:.4f},{self.p_id:.4f}"
        )


CSV_HEADER = "proposer,params,h_free_id,h_free_ood,p_id"


def _by_params(csv_path: str | Path) -> dict[int, dict[str, float]]:
    """Read a capacity-sweep CSV into ``{params: {metric: mean}}`` (the HS1/HS3 schema)."""
    out: dict[int, dict[str, float]] = {}
    with Path(csv_path).open() as handle:
        for row in csv.DictReader(handle):
            out.setdefault(int(row["params"]), {})[row["metric"]] = float(row["mean"])
    return out


def cross_proposer_synthesis(
    proposers: Mapping[str, str | Path] | None = None,
) -> list[ProposerPoint]:
    """Overlay free-running horizon vs capacity for each proposer -- the proposer-dependence figure.

    Reads each committed capacity-sweep CSV and emits one :class:`ProposerPoint` per (proposer,
    params) cell, carrying the id/ood free-running horizon and the in-distribution one-step `p`.
    """
    proposers = proposers or DEFAULT_PROPOSERS
    points: list[ProposerPoint] = []
    for proposer, path in proposers.items():
        for params, metrics in sorted(_by_params(path).items()):
            points.append(
                ProposerPoint(
                    proposer=proposer,
                    params=params,
                    h_free_id=metrics.get("h_free_id", 0.0),
                    h_free_ood=metrics.get("h_free_ood", 0.0),
                    p_id=metrics.get("one_step_acc_id", 0.0),
                )
            )
    return points


def write_csv(points: list[ProposerPoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="SPEC-10 cross-proposer synthesis (the floor is proposer-dependent)."
    )
    parser.add_argument("--out", type=str, default="figures/horizon_synthesis.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_synthesis.png")
    args = parser.parse_args()
    points = cross_proposer_synthesis()
    path = write_csv(points, args.out)
    print(f"wrote {path}")
    for proposer in dict.fromkeys(p.proposer for p in points):
        cells = sorted((p for p in points if p.proposer == proposer), key=lambda p: p.params)
        lo, hi = cells[0], cells[-1]
        print(
            f"  {proposer:<24} H_free(id) {lo.h_free_id:.2f} -> {hi.h_free_id:.2f}  "
            f"(p {lo.p_id:.2f} -> {hi.p_id:.2f})"
        )
    try:
        _plot(points, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


def _plot(points: list[ProposerPoint], path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"flat": "#1f77b4", "graph": "#d62728"}
    for proposer in dict.fromkeys(p.proposer for p in points):
        cells = sorted((p for p in points if p.proposer == proposer), key=lambda p: p.params)
        color = colors["graph"] if "graph" in proposer.lower() else colors["flat"]
        ax.plot(
            [c.params for c in cells], [c.h_free_id for c in cells],
            marker="o", color=color, label=f"{proposer}  [id]",
        )
        ax.plot(
            [c.params for c in cells], [c.h_free_ood for c in cells],
            marker="s", ls="--", color=color, alpha=0.55, label=f"{proposer}  [ood]",
        )
    ax.set_xscale("log")
    ax.set_xlabel("model params (≈ n_layer · n_embd²)")
    ax.set_ylabel("free-running faithful horizon  H_ε(ρ=0)  (steps)")
    ax.set_title(
        "SPEC-10 — the floor is proposer-dependent:\n"
        "capacity buys horizon for the flat arm, not the structured arm"
    )
    ax.legend(fontsize="small", loc="center left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
