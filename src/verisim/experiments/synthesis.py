"""Cross-world synthesis -- the floor+cliff ``H_ε(ρ)`` across the three worlds (the headline).

The program's most general empirical statement: the faithful-horizon-vs-consultation curve is the
**same floor+cliff shape in every world** -- the filesystem (SPEC-2, a tree), the network (SPEC-5, a
graph), and the host (SPEC-6, a coupled bundle) -- despite entirely different state types, oracles,
and models. This module reads the three worlds' committed ``H_ε(ρ)`` curve CSVs (the artifacts E1 /
EN1 / EH1 emit), normalizes each by its own horizon ceiling ``T`` (so worlds with different rollout
lengths are comparable), averages over difficulty at a fixed tolerance ``ε``, and returns one
normalized curve per world. Figures are produced *only* from committed records (the SPEC-2 §7.3
discipline) -- this re-reads them, it does not re-run anything. Pure and dependency-free.

The synthesis makes one figure carry the claim: if the three normalized curves overlay -- a floor
across the ``ρ`` interior, then a cliff to ``H_ε/T = 1`` at ``ρ=1`` -- then "a little consultation
does not buy a lot of horizon; you pay near-linearly for faithfulness" is a property of the
*oracle-loop method*, not of any one world or model (the SPEC.md §0 model-agnostic-primitive claim,
now also world-agnostic).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

#: The committed curve CSVs, by world (filesystem E1, network EN1, host EH1).
DEFAULT_WORLDS: dict[str, str] = {
    "filesystem (E1)": "figures/e1_curve.csv",
    "network (EN1)": "figures/en1_curve.csv",
    "host (EH1)": "figures/eh1_curve.csv",
}


@dataclass(frozen=True)
class SynthPoint:
    """One (world, ρ) cell: the difficulty-averaged horizon, the world's ceiling, the fraction."""

    world: str
    rho: float
    mean_h: float
    ceiling: float
    frac_h: float  # mean_h / ceiling -- the normalized faithful horizon, comparable across worlds

    def csv_row(self) -> str:
        return f"{self.world},{self.rho},{self.mean_h:.4f},{self.ceiling:.4f},{self.frac_h:.4f}"


CSV_HEADER = "world,rho,mean_h,ceiling,frac_h"


def world_curve(csv_path: str | Path, epsilon: float) -> list[tuple[float, float]]:
    """Read one world's curve CSV; return ``[(ρ, difficulty-averaged mean H_ε)]`` at tolerance ε."""
    by_rho: dict[float, list[float]] = {}
    with Path(csv_path).open() as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["epsilon"]) - epsilon) < 1e-9:
                by_rho.setdefault(float(row["rho"]), []).append(float(row["mean_h"]))
    return [(rho, fmean(vals)) for rho, vals in sorted(by_rho.items())]


def cross_world_curve(
    worlds: Mapping[str, str | Path] | None = None, *, epsilon: float = 0.0
) -> list[SynthPoint]:
    """Normalized ``H_ε/T`` vs ``ρ`` for each world -- the cross-world floor+cliff overlay.

    Each world is normalized by its own ceiling ``T`` (the max difficulty-averaged horizon over
    ``ρ``, i.e. the fully-consulted ``ρ=1`` value), so different-length worlds share one axis.
    """
    worlds = worlds or DEFAULT_WORLDS
    points: list[SynthPoint] = []
    for world, path in worlds.items():
        curve = world_curve(path, epsilon)
        ceiling = max((h for _, h in curve), default=0.0) or 1.0
        for rho, mean_h in curve:
            points.append(
                SynthPoint(world, rho, mean_h, ceiling, mean_h / ceiling)
            )
    return points


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Cross-world H_eps(rho) synthesis (floor+cliff).")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="figures/synthesis_floor_cliff.csv")
    args = parser.parse_args()
    points = cross_world_curve(epsilon=args.epsilon)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    print(f"wrote {out}")
    for world in dict.fromkeys(p.world for p in points):
        cells = [p for p in points if p.world == world]
        floor = next((p.frac_h for p in cells if p.rho == 0.0), 0.0)
        print(f"  {world:<18} ρ=0 floor {floor:.2f}·T  ->  ρ=1 ceiling 1.00·T")
    _plot(points, out.with_suffix(".png"), args.epsilon)


def _plot(points: list[SynthPoint], path: Path, epsilon: float) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for world in dict.fromkeys(p.world for p in points):
        cells = sorted((p for p in points if p.world == world), key=lambda p: p.rho)
        ax.plot([p.rho for p in cells], [p.frac_h for p in cells], marker="o", label=world)
    ax.set_xlabel("oracle-consultation budget  ρ")
    ax.set_ylabel("normalized faithful horizon  H_ε / T")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Verisim — the floor+cliff is the same shape in every world (ε={epsilon})")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
