"""Experiment SR5: the two-tier self-speculative cascade (H43, SPEC-13 §6).

Self-speculative decoding's idea: a cheap drafter pre-filters for an expensive one, so the expensive
model only runs on what the cheap one could not handle. SR5 lifts that to the verisim loop -- a tiny
``M_θ`` drafts, a larger ``M_θ`` re-drafts only the rejected suffix, and the oracle verifies last --
and asks the only question that matters here: does the cheap pre-filter reduce **oracle** calls per
faithful step? (Not GPU time -- the oracle is the expensive resource, the inversion of LLM
speculative decoding, SPEC-13 §8.)

The pre-registered answer is a banked **negative** (H43), and SR5 confirms it: the cascade does
*not*
beat simply running the larger drafter directly. Only the oracle adjudicates faithfulness, and its
verify already stops at the first divergence -- so the cheap tier adds an extra verify round at each
break without removing any. The cheapness the self-speculative line exploits lives on the GPU
(drafting time), which is free here; it does not live in the oracle, which is the only thing SR5
pays
for. The clean lesson: when an exact oracle is the verifier, **use the best drafter directly** --
model-vs-model speculation buys nothing.

CPU-only, torch-free, deterministic, seeded (the controlled stand-in drafters at two accuracies; the
trained tiny/large ``M_θ`` arm is the deferred/``skipif``-guarded one, never counted -- SPEC-13 §9,
the LP7 discipline).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.experiments.sr2 import SR2Config, granularity
from verisim.experiments.sr_common import SRWorld, StallDrafter, all_worlds, mean
from verisim.loop.speculative import free_run_divergences, speculative_rollout
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon


@dataclass(frozen=True)
class SR5Config:
    """A small, fast cascade instance (the dependency-free core)."""

    alpha_tiny: float = 0.6  # the cheap pre-filter's raw accuracy
    alpha_large: float = 0.9  # the larger re-drafter's raw accuracy
    k: int = 16
    g: float = 1.5  # ε = g·δ
    n_steps: int = 80
    n_seeds: int = 24
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SR5Config:
        b = SR5Config()
        return SR5Config(
            alpha_tiny=d.get("alpha_tiny", b.alpha_tiny),
            alpha_large=d.get("alpha_large", b.alpha_large),
            k=d.get("k", b.k),
            g=d.get("g", b.g),
            n_steps=d.get("n_steps", b.n_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> SR5Config:
        return SR5Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SR5Stat:
    """One (world, arm) cell: oracle calls per faithful step (lower is better)."""

    world: str
    arm: str  # "tiny-only" | "large-only" | "cascade"
    calls_per_faithful: float
    cpf_lo: float
    cpf_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world},{self.arm},{self.calls_per_faithful:.6f},"
            f"{self.cpf_lo:.6f},{self.cpf_hi:.6f},{self.n}"
        )


CSV_HEADER = "world,arm,calls_per_faithful,cpf_lo,cpf_hi,n"


def _cascade(
    world: SRWorld[Any, Any],
    s0: Any,
    actions: Sequence[Any],
    tiny: StallDrafter[Any, Any],
    large: StallDrafter[Any, Any],
    *,
    k: int,
    epsilon: float,
) -> tuple[int, int]:
    """Run the two-tier cascade over non-overlapping windows; return (oracle_calls, faithful_steps).

    Per window: the tiny drafter drafts, the oracle verifies (round 1); on a rejected suffix the
    larger drafter re-drafts from the true break state, the oracle verifies again (round 2). Each
    verify round is one oracle call -- the cascade's extra round is exactly the cost the negative is
    about.
    """
    anchor = s0
    i = 0
    oracle_calls = 0
    faithful = 0
    n = len(actions)
    while i + 1 < n:
        window = actions[i : i + k]
        w = len(window)
        d_tiny = free_run_divergences(
            anchor, window, tiny, world.oracle_step, world.diverge, start=i
        )
        a_tiny = faithful_horizon(d_tiny, epsilon)
        oracle_calls += 1  # tiny verify round
        a_total = a_tiny
        if a_tiny < w:  # larger re-drafts the rejected suffix from the true break state
            mid = anchor
            for action in window[:a_tiny]:
                mid = world.oracle_step(mid, action)
            suffix = window[a_tiny:]
            d_large = free_run_divergences(
                mid, suffix, large, world.oracle_step, world.diverge, start=i + a_tiny
            )
            a_total = a_tiny + faithful_horizon(d_large, epsilon)
            oracle_calls += 1  # larger verify round
        faithful += a_total
        advance = a_total + 1 if a_total < w else w
        state = anchor
        for action in window[:advance]:
            state = world.oracle_step(state, action)
        anchor = state
        i += advance
    return oracle_calls, faithful


def run_sr5(config: SR5Config | None = None) -> list[SR5Stat]:
    """Per (world, arm): oracle calls per faithful step for tiny / large / cascade (H43)."""
    config = config or SR5Config()
    gran_cfg = SR2Config(n_steps=config.n_steps)
    stats: list[SR5Stat] = []
    for world in all_worlds():
        epsilon = config.g * granularity(world, gran_cfg)
        ratios: dict[str, list[float]] = {"tiny-only": [], "large-only": [], "cascade": []}
        for s in range(config.n_seeds):
            seed = config.base_seed + s
            s0, actions = world.make_actions(5000 + seed, config.n_steps)
            tiny = StallDrafter(world.oracle_step, config.alpha_tiny, seed=seed)
            large = StallDrafter(world.oracle_step, config.alpha_large, seed=1000 + seed)
            rt = speculative_rollout(
                s0, actions, tiny, world.oracle_step, world.diverge, k=config.k, epsilon=epsilon
            )
            rl = speculative_rollout(
                s0, actions, large, world.oracle_step, world.diverge, k=config.k, epsilon=epsilon
            )
            oc, fa = _cascade(world, s0, actions, tiny, large, k=config.k, epsilon=epsilon)
            if rt.faithful_free_steps:
                ratios["tiny-only"].append(rt.oracle_calls / rt.faithful_free_steps)
            if rl.faithful_free_steps:
                ratios["large-only"].append(rl.oracle_calls / rl.faithful_free_steps)
            if fa:
                ratios["cascade"].append(oc / fa)
        for arm, vals in ratios.items():
            lo, hi = bootstrap_ci(vals, seed=0)
            stats.append(SR5Stat(world.name, arm, mean(vals), lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[SR5Stat]) -> None:
    print("SR5 / H43 - the two-tier cascade: does a cheap pre-filter cut ORACLE calls? (no):")
    print("  [trained tiny/large M_θ arm DEFERRED -- the committed core is the stand-in (§9)]")
    for w in sorted({s.world for s in stats}):
        print(f"  -- {w} (oracle calls per faithful step, lower better) --")
        for arm in ("tiny-only", "large-only", "cascade"):
            s = next(s for s in stats if s.world == w and s.arm == arm)
            print(f"  {arm:>11}: {s.calls_per_faithful:.3f}  [{s.cpf_lo:.3f}, {s.cpf_hi:.3f}]")
    large = mean([s.calls_per_faithful for s in stats if s.arm == "large-only"])
    casc = mean([s.calls_per_faithful for s in stats if s.arm == "cascade"])
    verdict = (
        f"the cascade ({casc:.3f}) does not beat the larger drafter directly ({large:.3f}) - "
        "H43 refuted (model-vs-model speculation saves no ORACLE calls; the cheap tier adds verify "
        "rounds — the cheapness lives on the GPU, which is free here, not in the oracle, §8)"
        if casc >= large - 0.005
        else f"the cascade ({casc:.3f}) beats large-only ({large:.3f}) - H43 supported (unexpected)"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[SR5Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    worlds = sorted({s.world for s in stats})
    arms = ("tiny-only", "large-only", "cascade")
    colors = {"tiny-only": "#d62728", "large-only": "#1f77b4", "cascade": "#ff7f0e"}
    width = 0.25
    x = range(len(worlds))
    for j, arm in enumerate(arms):
        vals = [next(s.calls_per_faithful for s in stats if s.world == w and s.arm == arm)
                for w in worlds]
        los = [next(s.cpf_lo for s in stats if s.world == w and s.arm == arm) for w in worlds]
        his = [next(s.cpf_hi for s in stats if s.world == w and s.arm == arm) for w in worlds]
        errs = [[v - lo for v, lo in zip(vals, los, strict=True)],
                [hi - v for v, hi in zip(vals, his, strict=True)]]
        ax.bar([i + (j - 1) * width for i in x], vals, width, yerr=errs, capsize=3,
               color=colors[arm], label=arm)
    ax.set_xticks(list(x))
    ax.set_xticklabels(worlds)
    ax.set_ylabel("oracle calls per faithful step (lower is better)")
    ax.set_title("SR5 / H43: a cheap pre-filter adds oracle rounds — use the best drafter directly")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="SR5 two-tier self-speculative cascade (H43).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/sr5_cascade.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = SR5Config.from_json_file(args.config) if args.config else SR5Config()
    stats = run_sr5(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
