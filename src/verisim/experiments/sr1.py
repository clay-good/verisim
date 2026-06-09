"""Experiment SR1: speculative vs fixed-``ρ`` at equal budget (H39, the headline; SPEC-13 §6).

The program's central negative was a *no favorable knee*: the fixed/uniform consultation policies
produced a floor + cliff with no regime where a smarter schedule earned more faithful horizon per
oracle dollar (EN7/H22). SPEC-13's bet is that the one shape those policies are not -- **consult at
the break, not on a clock** (accept the longest faithful prefix, re-anchor at the first divergence)
--
is the escape. SR1 measures it at *equal expensive budget*: give both policies the same number of
full oracle corrections ``B = ρ·T`` and compare the faithful fraction ``H_ε / T``.

The honest result is a **budget crossover**, not a clean win (the program's epistemic style: report
the
split). Accept-longest-prefix is *budget-greedy* -- it consults reactively at breaks, so under a
scarce
budget it spends its corrections early and free-runs (drifts) the tail, while fixed's uniform clock
spreads the same corrections across the whole rollout. So:

  - **at scarce budget (ρ < ρ\\*)** fixed wins -- uniform coverage beats greedy early spend (H39
    *refuted* in this regime, banked as a real property of reactive scheduling);
  - **at sufficient budget (ρ ≥ ρ\\*)** speculative wins and reaches **full faithfulness**
    -- every correction lands exactly on a step that was about to break, none wasted on a clock tick
    over a still-faithful step (H39 *supported*).

The crossover ``ρ\\*`` is the measured quantity. A second panel reports the spec's stated figure of
merit -- *faithful steps advanced per oracle call* (the speculative speedup ``E[a]``) -- on which
speculative is above fixed throughout (its calls adapt to the accepted prefix). The two panels make
the distinction precise: per-call, consult-at-break is always more efficient; in *total*
faithfulness
under a hard call budget, it must also be allowed to ration, which pure accept-longest-prefix does
not.

CPU-only, torch-free, deterministic, seeded (the controlled stand-in drafter; the trained-``M_θ``
magnitudes are the deferred/``skipif``-guarded arm, never counted -- SPEC-13 §9, the LP7
discipline).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.experiments.sr2 import granularity
from verisim.experiments.sr_common import SRWorld, StallDrafter, all_worlds, mean
from verisim.loop.speculative import fixed_interval_rollout, speculative_rollout
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class SR1Config:
    """A small, fast speculative-vs-fixed instance at equal expensive budget (the dependency-free
    core).
    """

    alpha: float = 0.72  # the drafter's raw per-step accuracy (low enough that drift bites)
    k: int = 16  # speculative draft-window length
    g: float = 1.5  # ε = g·δ (gradual-but-drifting: free-run drifts past ε, so budget matters)
    n_steps: int = 120
    budgets: tuple[int, ...] = (2, 4, 6, 8, 12, 16, 24, 32)  # expensive oracle corrections B
    n_seeds: int = 24
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SR1Config:
        b = SR1Config()
        return SR1Config(
            alpha=d.get("alpha", b.alpha),
            k=d.get("k", b.k),
            g=d.get("g", b.g),
            n_steps=d.get("n_steps", b.n_steps),
            budgets=tuple(d.get("budgets", b.budgets)),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> SR1Config:
        return SR1Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SR1Stat:
    """One (world, B) cell: faithful fraction and per-call speedup, speculative vs fixed."""

    world: str
    budget: int
    rho: float
    spec_faithful: float  # H_ε / T for speculative at this budget
    spec_lo: float
    spec_hi: float
    fix_faithful: float  # H_ε / T for fixed-ρ at the same expensive budget
    fix_lo: float
    fix_hi: float
    spec_steps_per_call: float  # E[a] -- accepted free-run steps per oracle call
    fix_steps_per_call: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world},{self.budget},{self.rho:.6f},{self.spec_faithful:.6f},{self.spec_lo:.6f},"
            f"{self.spec_hi:.6f},{self.fix_faithful:.6f},{self.fix_lo:.6f},{self.fix_hi:.6f},"
            f"{self.spec_steps_per_call:.6f},{self.fix_steps_per_call:.6f},{self.n}"
        )


CSV_HEADER = (
    "world,budget,rho,spec_faithful,spec_lo,spec_hi,fix_faithful,fix_lo,fix_hi,"
    "spec_steps_per_call,fix_steps_per_call,n"
)


def _cell(world: SRWorld[Any, Any], config: SR1Config, epsilon: float, budget: int) -> SR1Stat:
    spec_frac: list[float] = []
    fix_frac: list[float] = []
    spec_spc: list[float] = []
    fix_spc: list[float] = []
    interval = max(1, round(config.n_steps / budget))
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        s0, actions = world.make_actions(2000 + seed, config.n_steps)
        drafter = StallDrafter(world.oracle_step, config.alpha, seed=seed)
        spec = speculative_rollout(
            s0, actions, drafter, world.oracle_step, world.diverge,
            k=config.k, epsilon=epsilon, max_corrections=budget,
        )
        fix = fixed_interval_rollout(
            s0, actions, drafter, world.oracle_step, world.diverge,
            interval=interval, epsilon=epsilon,
        )
        spec_frac.append(spec.faithful_steps / spec.total_steps)
        fix_frac.append(fix.faithful_steps / fix.total_steps)
        spec_spc.append(spec.steps_per_call)
        fix_spc.append(fix.steps_per_call)
    s_lo, s_hi = bootstrap_ci(spec_frac, seed=0)
    f_lo, f_hi = bootstrap_ci(fix_frac, seed=0)
    return SR1Stat(
        world=world.name,
        budget=budget,
        rho=budget / config.n_steps,
        spec_faithful=mean(spec_frac),
        spec_lo=s_lo,
        spec_hi=s_hi,
        fix_faithful=mean(fix_frac),
        fix_lo=f_lo,
        fix_hi=f_hi,
        spec_steps_per_call=mean(spec_spc),
        fix_steps_per_call=mean(fix_spc),
        n=len(spec_frac),
    )


def run_sr1(config: SR1Config | None = None) -> list[SR1Stat]:
    """Per (world, B): faithful fraction and per-call speedup, speculative vs fixed-ρ (H39)."""
    config = config or SR1Config()
    from verisim.experiments.sr2 import SR2Config

    gran_cfg = SR2Config(n_steps=config.n_steps)
    stats: list[SR1Stat] = []
    for world in all_worlds():
        epsilon = config.g * granularity(world, gran_cfg)
        for budget in config.budgets:
            stats.append(_cell(world, config, epsilon, budget))
    return stats


def _crossover(cells: list[SR1Stat]) -> float | None:
    """The smallest ρ at which speculative's faithful fraction first meets/exceeds fixed's (ρ\\*).
    """
    for s in sorted(cells, key=lambda s: s.budget):
        if s.spec_faithful >= s.fix_faithful:
            return s.rho
    return None


def _print_summary(stats: list[SR1Stat]) -> None:
    print("SR1 / H39 - speculative (consult-at-break) vs fixed-ρ (clock) at equal budget:")
    print("  [trained-M_θ arm DEFERRED -- committed core is the stand-in drafter (§9, LP7 rule)]")
    for w in sorted({s.world for s in stats}):
        cells = sorted((s for s in stats if s.world == w), key=lambda s: s.budget)
        print(f"  -- {w} --")
        # spec E[a] = accepted prefix per cheap verify; fix run/call = faithful steps per expensive
        # consult -- different cost types, shown descriptively, not as a head-to-head (SPEC-13 §8).
        print(f"  {'ρ':>6} {'spec H_ε/T':>11} {'fix H_ε/T':>10} {'gap':>7} "
              f"{'spec E[a]':>10} {'fix run/c':>9}")
        for s in cells:
            print(
                f"  {s.rho:>6.3f} {s.spec_faithful:>11.3f} {s.fix_faithful:>10.3f} "
                f"{s.spec_faithful - s.fix_faithful:>+7.3f} {s.spec_steps_per_call:>10.2f} "
                f"{s.fix_steps_per_call:>9.2f}"
            )
        star = _crossover(cells)
        print(f"  crossover ρ* = {star if star is None else f'{star:.3f}'} "
              "(below: fixed wins; above: speculative wins, reaches full faithfulness)")
    high = [s for s in stats if s.rho >= 0.15]
    low = [s for s in stats if s.rho <= 0.05]
    sh, fh = mean([s.spec_faithful for s in high]), mean([s.fix_faithful for s in high])
    fl, sl = mean([s.fix_faithful for s in low]), mean([s.spec_faithful for s in low])
    verdict = (
        f"at sufficient budget (ρ≥0.15) speculative reaches full faithfulness "
        f"({sh:.2f} vs {fh:.2f}); "
        f"at scarce budget (ρ≤0.05) fixed's uniform spread wins ({fl:.2f} vs {sl:.2f}) "
        "- H39 supported above ρ*, refuted below (budget-greedy: it spends early "
        "and free-runs the tail)"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[SR1Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worlds = sorted({s.world for s in stats})
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.4 * len(worlds), 4.2), sharey=True)
    if len(worlds) == 1:
        axes = [axes]
    for ax, w in zip(axes, worlds, strict=True):
        cells = sorted((s for s in stats if s.world == w), key=lambda s: s.rho)
        xs = [s.rho for s in cells]
        ax.plot(xs, [s.spec_faithful for s in cells], "-o", color="#1f77b4", label="speculative")
        ax.fill_between(xs, [s.spec_lo for s in cells], [s.spec_hi for s in cells],
                        color="#1f77b4", alpha=0.12)
        ax.plot(xs, [s.fix_faithful for s in cells], "-s", color="#d62728", label="fixed-ρ")
        ax.fill_between(xs, [s.fix_lo for s in cells], [s.fix_hi for s in cells],
                        color="#d62728", alpha=0.12)
        star = _crossover(cells)
        if star is not None:
            ax.axvline(star, color="#888", ls=":", lw=1)
            ax.text(star, 0.05, f" ρ*={star:.2f}", fontsize=7, color="#444")
        ax.set_xlabel("oracle budget ρ = B / T")
        ax.set_title(w)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("faithful fraction H_ε / T")
    fig.suptitle("SR1 / H39: consult-at-break vs consult-on-clock — the budget crossover ρ*")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="SR1 speculative vs fixed at equal budget (H39).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/sr1_knee.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = SR1Config.from_json_file(args.config) if args.config else SR1Config()
    stats = run_sr1(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
