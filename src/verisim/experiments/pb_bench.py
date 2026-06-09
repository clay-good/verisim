"""Experiment PB-bench: the discriminative-validity leaderboard (SPEC-18 §6, H65). *Gates the rest.*

A leaderboard that does not discriminate is not worth packaging, so H65 gates the whole product
line.
PB-bench scores the reference fidelity ladder (:data:`~verisim.bench.manifest.REFERENCE_PROPOSERS`)
on
the frozen battery across seeds, per world, and reports:

  - the **leaderboard** — mean faithful fraction `H_ε/T` per (world, proposer), attributed to the
    battery's manifest hash;
  - **rank stability** — Kendall's τ between disjoint seed-split leaderboards (target τ ≥ 0.8, CI
    excluding 0), and the strict **adjacent-tier resolution** check: the binding (worst-margin)
    adjacent
    pair's gap exceeds twice its *paired* seed noise (common-mode seed noise does not reorder the
    ranking, so the gap that matters is between adjacent proposers, and the noise that matters is
    the
    noise of that gap).

H65 is supported when both hold at every world. The committed core scores controlled-stand-in
proposers
(a per-step-accuracy ladder); the trained flat-transformer / GNN+RSSM arms are the deferred real
entries
(`skipif`-guarded, never scored without a checkpoint — the LP7 rule). The drift and divergence are
real
(the SPEC-13 world bundles against the real oracle). CPU-only, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.bench.leaderboard import (
    CSV_HEADER,
    STABILITY_HEADER,
    LeaderRow,
    RankStability,
    build_leaderboard,
)
from verisim.bench.manifest import REFERENCE_PROPOSERS, BatteryManifest


@dataclass(frozen=True)
class PBBenchConfig:
    """A small, fast discriminative-validity instance (the dependency-free core)."""

    seeds: int = 24
    n_steps: int = 80
    epsilon_g: float = 1.0

    def manifest(self) -> BatteryManifest:
        return BatteryManifest(
            seeds=tuple(range(self.seeds)), n_steps=self.n_steps, epsilon_g=self.epsilon_g,
            proposers=REFERENCE_PROPOSERS,
        )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PBBenchConfig:
        b = PBBenchConfig()
        return PBBenchConfig(
            seeds=d.get("seeds", b.seeds),
            n_steps=d.get("n_steps", b.n_steps),
            epsilon_g=d.get("epsilon_g", b.epsilon_g),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> PBBenchConfig:
        return PBBenchConfig.from_dict(json.loads(Path(path).read_text()))


def run_pb_bench(
    config: PBBenchConfig | None = None,
) -> tuple[BatteryManifest, list[LeaderRow], list[RankStability]]:
    """Score the ladder on the battery; return the manifest, leaderboard, and rank stability (H65).
    """
    config = config or PBBenchConfig()
    manifest = config.manifest()
    rows, stability = build_leaderboard(manifest)
    return manifest, rows, stability


def _print_summary(
    manifest: BatteryManifest, rows: list[LeaderRow], stability: list[RankStability]
) -> None:
    print("PB-bench / H65 - the faithfulness benchmark is discriminative (stable rankings):")
    print("  [trained transformer/GNN arms DEFERRED -- core is the fidelity-ladder stand-in]")
    print(f"  battery: {manifest.version_tag()}")
    for world in manifest.worlds:
        print(f"  -- {world} leaderboard (H_ε/T) --")
        cells = sorted((r for r in rows if r.world == world),
                       key=lambda r: r.mean_faithful, reverse=True)
        for r in cells:
            print(f"    {r.proposer:14s} {r.tier:12s} {r.mean_faithful:.3f}")
        s = next(s for s in stability if s.world == world)
        print(f"    rank stability: τ={s.tau_mean:.3f} [{s.tau_lo:.2f},{s.tau_hi:.2f}]  "
              f"adj gap {s.min_adjacent_gap:.3f} vs 2×noise {2 * s.max_seed_noise:.3f}  "
              f"-> {'DISCRIMINATIVE' if s.discriminative else 'not discriminative'}")
    all_discr = all(s.discriminative for s in stability)
    mean_tau = sum(s.tau_mean for s in stability) / len(stability)
    verdict = (
        f"every world's leaderboard is rank-stable (mean τ={mean_tau:.3f}) and resolves adjacent "
        "fidelity tiers above paired seed noise - H65 supported: the benchmark discriminates, "
        "worth packaging (PB-transfer, PB-pack)"
        if all_discr
        else "some world's leaderboard does not clear the noise band - H65 a fixable measurement "
        "finding (add seeds / tighten ε / harden the battery)"
    )
    print(f"  verdict: {verdict}")


def _plot(rows: list[LeaderRow], stability: list[RankStability], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worlds = sorted({r.world for r in rows})
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.4 * len(worlds), 4.4), sharey=True)
    if len(worlds) == 1:
        axes = [axes]
    palette = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    for ax, world in zip(axes, worlds, strict=True):
        cells = sorted((r for r in rows if r.world == world), key=lambda r: r.mean_faithful)
        names = [r.proposer for r in cells]
        ys = [r.mean_faithful for r in cells]
        ax.barh(range(len(names)), ys, color=palette[: len(names)])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        s = next(s for s in stability if s.world == world)
        verdict = "discriminative" if s.discriminative else "noisy"
        ax.set_title(f"{world}\nτ={s.tau_mean:.2f}, {verdict}", fontsize=9)
        ax.set_xlim(0, 1.05)
    axes[0].set_xlabel("faithful fraction H_ε / T")
    fig.suptitle("PB-bench / H65: the faithfulness leaderboard stably orders the fidelity ladder")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="PB-bench discriminative leaderboard (H65).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/pb_bench_leaderboard.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = PBBenchConfig.from_json_file(args.config) if args.config else PBBenchConfig()
    manifest, rows, stability = run_pb_bench(cfg)
    _print_summary(manifest, rows, stability)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mh = manifest.manifest_hash()
    lines = [CSV_HEADER, *(r.csv_row(mh) for r in rows), "", STABILITY_HEADER,
             *(f"{s.world},{s.tau_mean:.6f},{s.tau_lo:.6f},{s.tau_hi:.6f},{s.min_adjacent_gap:.6f},"
               f"{s.max_seed_noise:.6f},{s.discriminative}" for s in stability)]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(rows, stability, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
