"""Experiment SR3: multi-draft (tree) verification (H42, SPEC-13 §6).

The SpecInfer move: sample several candidate rollouts from the drafter and verify them all against
**one** oracle trajectory, keeping the longest-faithful branch. The oracle step costs the same
whether
one draft or many are checked against it, so a tree of drafts is a near-free way to raise the
accepted
prefix -- *if* the drafter's divergence is **variance** (its errors are stochastic, so different
drafts stall in different places and at least one threads the faithful needle) rather than **bias**
(its errors are systematic, so every draft stalls in the same places and the tree cannot help).

SR3 measures exactly that fork. The controlled drafter
(:class:`~verisim.experiments.sr_common.StallDrafter`)
has two modes seeded identically except for one bit: ``systematic=False`` stalls independently per
draft variant (variance) and ``systematic=True`` stalls in variant-invariant places (bias). The
committed result shows best-of-``m`` accepted prefix rising with ``m`` under variance and flat under
bias -- the H42 prediction made visible, and the diagnostic that says whether the real ``M_θ``'s
divergence is correctable by a wider tree or needs a debias/correction step instead (SPEC-13 §5
H42).

CPU-only, torch-free, deterministic, seeded (the controlled stand-in; the trained-``M_θ`` arm is the
deferred/``skipif``-guarded one, never counted -- SPEC-13 §9, the LP7 discipline).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.experiments.sr2 import SR2Config, granularity
from verisim.experiments.sr_common import StallDrafter, all_worlds, mean
from verisim.loop.speculative import speculative_rollout
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class SR3Config:
    """A small, fast tree-draft instance (the dependency-free core)."""

    alpha: float = 0.7
    k: int = 12
    g: float = 1.0  # ε = g·δ
    m_values: tuple[int, ...] = (1, 2, 4, 8)  # number of draft variants in the tree
    n_steps: int = 80
    n_seeds: int = 24
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SR3Config:
        b = SR3Config()
        return SR3Config(
            alpha=d.get("alpha", b.alpha),
            k=d.get("k", b.k),
            g=d.get("g", b.g),
            m_values=tuple(d.get("m_values", b.m_values)),
            n_steps=d.get("n_steps", b.n_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> SR3Config:
        return SR3Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SR3Stat:
    """One (world, error-mode, m) cell: best-of-m accepted prefix, mean + bootstrap CI."""

    world: str
    mode: str  # "variance" | "bias"
    m: int
    mean_prefix: float
    pref_lo: float
    pref_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world},{self.mode},{self.m},{self.mean_prefix:.6f},"
            f"{self.pref_lo:.6f},{self.pref_hi:.6f},{self.n}"
        )


CSV_HEADER = "world,mode,m,mean_prefix,pref_lo,pref_hi,n"


def run_sr3(config: SR3Config | None = None) -> list[SR3Stat]:
    """Per (world, mode, m): best-of-m accepted prefix -- the variance/bias fork (H42)."""
    config = config or SR3Config()
    gran_cfg = SR2Config(n_steps=config.n_steps)
    stats: list[SR3Stat] = []
    for world in all_worlds():
        epsilon = config.g * granularity(world, gran_cfg)
        for mode, systematic in (("variance", False), ("bias", True)):
            for m in config.m_values:
                prefixes: list[float] = []
                for s in range(config.n_seeds):
                    seed = config.base_seed + s
                    s0, actions = world.make_actions(3000 + seed, config.n_steps)
                    drafter = StallDrafter(
                        world.oracle_step, config.alpha, seed=seed, systematic=systematic
                    )
                    rec = speculative_rollout(
                        s0, actions, drafter, world.oracle_step, world.diverge,
                        k=config.k, epsilon=epsilon, n_drafts=m,
                    )
                    prefixes.extend(float(a) for a in rec.accepted_prefixes)
                lo, hi = bootstrap_ci(prefixes, seed=0)
                stats.append(
                    SR3Stat(world.name, mode, m, mean(prefixes), lo, hi, len(prefixes))
                )
    return stats


def _print_summary(stats: list[SR3Stat]) -> None:
    print("SR3 / H42 - multi-draft (tree) verification: helps under variance, null under bias:")
    print("  [trained-M_θ arm DEFERRED -- committed core is the stand-in drafter (§9, LP7 rule)]")
    for w in sorted({s.world for s in stats}):
        print(f"  -- {w} --")
        for mode in ("variance", "bias"):
            cells = sorted((s for s in stats if s.world == w and s.mode == mode), key=lambda s: s.m)
            row = "  ".join(f"m={s.m}:{s.mean_prefix:.2f}" for s in cells)
            print(f"  {mode:>9}: {row}")
    # The fork: under variance best-of-m grows; under bias it is flat.
    var_lift: list[float] = []
    bias_lift: list[float] = []
    for w in sorted({s.world for s in stats}):
        for mode, bucket in (("variance", var_lift), ("bias", bias_lift)):
            cells = sorted((s for s in stats if s.world == w and s.mode == mode), key=lambda s: s.m)
            if len(cells) >= 2 and cells[0].mean_prefix > 0:
                bucket.append(cells[-1].mean_prefix / cells[0].mean_prefix)
    verdict = (
        f"best-of-m lifts the accepted prefix {mean(var_lift):.2f}× under variance but "
        f"{mean(bias_lift):.2f}× under bias - H42 supported (a tree helps iff divergence is "
        "stochastic; systematic error needs debiasing, not more drafts)"
        if mean(var_lift) > mean(bias_lift) + 0.1
        else "the tree gives no variance/bias separation - H42 unsupported on this drafter"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[SR3Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worlds = sorted({s.world for s in stats})
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.4 * len(worlds), 4.2), sharey=True)
    if len(worlds) == 1:
        axes = [axes]
    styles = {"variance": ("#1f77b4", "-o"), "bias": ("#d62728", "--s")}
    for ax, w in zip(axes, worlds, strict=True):
        for mode in ("variance", "bias"):
            cells = sorted((s for s in stats if s.world == w and s.mode == mode), key=lambda s: s.m)
            color, fmt = styles[mode]
            xs = [s.m for s in cells]
            ys = [s.mean_prefix for s in cells]
            ax.plot(xs, ys, fmt, color=color, label=mode)
            ax.fill_between(xs, [s.pref_lo for s in cells], [s.pref_hi for s in cells],
                            color=color, alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("draft-tree width m")
        ax.set_title(w)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("best-of-m accepted prefix (steps)")
    fig.suptitle("SR3 / H42: a draft tree raises the prefix under variance, not under bias")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="SR3 multi-draft tree verification (H42).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/sr3_tree.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = SR3Config.from_json_file(args.config) if args.config else SR3Config()
    stats = run_sr3(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
