"""Experiment CX1: the counterfactual effect is hidden-state-dependent (H61, SPEC-17 §6).

The do-calculus reading of H5 as an *effect-size*, measured purely on the oracle. For each world,
sweep
interventions across rollout depths and seeds and measure two effects of each ``do(a')``:

  - **immediate** -- the rung-2 one-step divergence the intervention causes;
  - **downstream** -- the rung-3 terminal divergence after re-running ``F`` forward with the factual
    future held fixed (abduction-action-prediction).

Where the world has a *persistent exogenous medium* (the distributed partition/crash state) an
intervention's effect **amplifies** as ``F`` re-runs forward (downstream ≫ immediate) -- abduction
(rung 3) reveals consequences the one-step branch (rung 2) cannot, and the on-policy distribution
underrepresents that off-policy region. Where the world re-converges (the network's reachability
heals)
the effect washes out (downstream ≲ immediate). So the *structure* a counterfactual carries is
hidden-state-dependent -- large on the distributed world (ED6), small on the on-policy-complete
network/host worlds (EN6/EH6) -- exactly the mixed H5 result, now with a mechanism.

This is the pure-oracle effect-size. The **learned** lift -- whether *training* a parametric ``M_θ``
on
rung-3 targets improves held-out counterfactual prediction, and the matched-coverage cut that
separates
*branching* from *coverage* (CX2-CX4, H62) -- needs a contrastive trained arm to exploit the paired
structure and is deferred (the LP7 rule, SPEC-17 §7); a non-parametric stand-in sees only the
coverage
channel and is never counted as the structure verdict. CPU-only, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.causal.scm import Intervention, abduct_and_replay, downstream_amplification
from verisim.experiments.cx_common import CXWorld, all_cx_worlds
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class CX1Config:
    """A small, fast counterfactual-effect-size instance (the dependency-free core)."""

    n_steps: int = 40
    n_seeds: int = 16
    depths: tuple[float, ...] = (0.25, 0.5, 0.75)  # intervene at these fractions of rollout depth
    interventions_per_cell: int = 3  # alt actions sampled per (seed, depth)
    consequential_ratio: float = 2.0  # downstream/immediate above this => "consequential"
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CX1Config:
        b = CX1Config()
        return CX1Config(
            n_steps=d.get("n_steps", b.n_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            depths=tuple(d.get("depths", b.depths)),
            interventions_per_cell=d.get("interventions_per_cell", b.interventions_per_cell),
            consequential_ratio=d.get("consequential_ratio", b.consequential_ratio),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CX1Config:
        return CX1Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CX1Stat:
    """One world's counterfactual-effect cell: amplification + consequential-intervention rate."""

    world: str
    immediate: float  # mean rung-2 one-step effect
    downstream: float  # mean rung-3 terminal effect
    amplification: float  # mean downstream/immediate over interventions with a non-zero immediate
    amp_lo: float
    amp_hi: float
    consequential_rate: float  # fraction with amplification >= consequential_ratio
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world},{self.immediate:.6f},{self.downstream:.6f},{self.amplification:.6f},"
            f"{self.amp_lo:.6f},{self.amp_hi:.6f},{self.consequential_rate:.6f},{self.n}"
        )


CSV_HEADER = "world,immediate,downstream,amplification,amp_lo,amp_hi,consequential_rate,n"


def _world_stat(world: CXWorld[Any, Any], config: CX1Config) -> CX1Stat:
    immediates: list[float] = []
    downstreams: list[float] = []
    amps: list[float] = []
    consequential = 0
    total = 0
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        s0, actions, states = abduct_and_replay(
            world.make_actions, world.oracle_step, seed, config.n_steps
        )
        for frac in config.depths:
            t = min(len(actions) - 1, max(0, int(frac * config.n_steps)))
            for j in range(config.interventions_per_cell):
                alt = world.alt_action(states[t], 7000 + seed * 97 + int(frac * 100) * 13 + j)
                immediate, downstream = downstream_amplification(
                    world.oracle_step, world.diverge, s0, actions, states, Intervention(t, alt)
                )
                immediates.append(immediate)
                downstreams.append(downstream)
                if immediate > 0:
                    amp = downstream / immediate
                    amps.append(amp)
                    total += 1
                    if amp >= config.consequential_ratio:
                        consequential += 1
    lo, hi = bootstrap_ci(amps, seed=0)
    return CX1Stat(
        world.name, fmean(immediates), fmean(downstreams), fmean(amps) if amps else 0.0,
        lo, hi, consequential / total if total else 0.0, len(amps)
    )


def run_cx1(config: CX1Config | None = None) -> list[CX1Stat]:
    """Per world: the rung-2/rung-3 amplification and consequential-intervention rate (H61)."""
    config = config or CX1Config()
    return [_world_stat(world, config) for world in all_cx_worlds()]


def _print_summary(stats: list[CX1Stat]) -> None:
    print("CX1 / H61 - the counterfactual effect is hidden-state-dependent (rung-3 amplifies):")
    print("  [the LEARNED lift (training M_θ on rung-3 targets) + the matched-coverage cut are the")
    print("   deferred trained/contrastive-arm bets (CX2-CX4); CX1 is the pure-oracle effect-size]")
    print(f"  {'world':>11} {'immed':>8} {'downstr':>8} {'amplify':>8} {'95% CI':>13} {'cq':>5}")
    for s in sorted(stats, key=lambda s: s.amplification, reverse=True):
        print(
            f"  {s.world:>11} {s.immediate:>10.4f} {s.downstream:>11.4f} {s.amplification:>8.2f}× "
            f"{f'[{s.amp_lo:.2f}, {s.amp_hi:.2f}]':>16} {s.consequential_rate:>8.2f}"
        )
    dist = next((s for s in stats if s.world == "distributed"), None)
    on_policy = [s for s in stats if s.world in ("network", "host")]
    amp_dist = dist.amplification if dist else 0.0
    amp_op = fmean([s.amplification for s in on_policy]) if on_policy else 0.0
    verdict = (
        f"the distributed world's counterfactual effect amplifies {amp_dist:.1f}× downstream (its "
        f"persistent partition/crash medium carries the intervention forward) while the "
        f"on-policy-complete network/host worlds amplify only {amp_op:.1f}× (washes out) - "
        "H61 effect-size supported: counterfactual structure is hidden-state-dependent "
        "reading of the mixed H5); the learned lift is deferred to the trained arm"
        if dist and amp_dist > amp_op + 0.3
        else "no clear hidden-state ordering in the effect size - H61 effect-size unsupported here"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[CX1Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ordered = sorted(stats, key=lambda s: s.amplification)
    worlds = [s.world for s in ordered]
    x = range(len(worlds))
    ax1.barh(x, [s.amplification for s in ordered], color="#d62728")
    ax1.errorbar([s.amplification for s in ordered], list(x),
                 xerr=[[s.amplification - s.amp_lo for s in ordered],
                       [s.amp_hi - s.amplification for s in ordered]],
                 fmt="none", ecolor="#333", capsize=3)
    ax1.axvline(1.0, color="#888", ls=":", lw=1, label="no amplification (rung-3 = rung-2)")
    ax1.set_yticks(list(x))
    ax1.set_yticklabels(worlds, fontsize=9)
    ax1.set_xlabel("downstream / immediate amplification")
    ax1.set_title("rung-3 amplifies where there is persistent hidden state")
    ax1.legend(fontsize=8)
    width = 0.38
    xs = range(len(stats))
    ax2.bar([i - width / 2 for i in xs], [s.immediate for s in stats], width,
            color="#9467bd", label="rung-2 immediate")
    ax2.bar([i + width / 2 for i in xs], [s.downstream for s in stats], width,
            color="#d62728", label="rung-3 downstream")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([s.world for s in stats], fontsize=8, rotation=15)
    ax2.set_ylabel("intervention effect (divergence)")
    ax2.set_title("the counterfactual effect, immediate vs downstream")
    ax2.legend(fontsize=8)
    fig.suptitle("CX1 / H61: the counterfactual effect is hidden-state-dependent (do-calculus H5)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CX1 hidden-state-dependent cf effect (H61).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cx1_counterfactual_effect.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CX1Config.from_json_file(args.config) if args.config else CX1Config()
    stats = run_cx1(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
