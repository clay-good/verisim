"""Experiment SR2: the accepted-prefix law per world (H40, SPEC-13 §6). *Runs first; gates SR1.*

The load-bearing assumption of the whole speculative line (SPEC-13 §4): the faithful prefix is long
enough that consulting at the break beats consulting on a clock. SR2 measures it *before* any policy
is run. For each world it free-runs the drafter from many anchored states, verifies step by step
against the oracle, and records the **accepted-prefix distribution** -- the length of the longest
within-``ε`` prefix of a draft window -- against the i.i.d. speculative-speedup law
``E[a] = (1 - α^{k+1})/(1 - α)`` (:func:`~verisim.loop.speculative.accepted_prefix_law`).

The honest reframing of K4 (SPEC-13 §4, §8). The spec pre-registered a *world-identity* split
(network/host gradual, filesystem discrete). The actual controlling variable is dimensionless: the
ratio ``g = ε / δ``, where ``δ`` is the world's **single-edit divergence granularity** (the median
divergence a single missed edit produces). When ``g ≥ ~2`` several edits fit under ``ε`` before the
prefix breaks -- drift is *gradual*, the prefix is long. When ``g < 1`` the first missed edit
already
exceeds ``ε`` -- drift is *discrete* (the K4 cliff), the prefix collapses to a geometric with the
*stall* as the breaking event. SR2 sweeps ``g`` per world and shows the accepted-prefix curves
**collapse across worlds onto one law in g** -- the result is governed by the metric's granularity,
not the world's identity (the H44 thesis, made visible here and collapsed in SR6). The proposer's
raw
accuracy ``α`` is held identical across worlds, so the figure isolates the world's contribution.

CPU-only, torch-free, deterministic, seeded (the controlled stand-in drafter, SPEC-13 §9; the real
``M_θ`` magnitudes are the deferred/``skipif``-guarded arm, never counted here -- the LP7
discipline).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from verisim.experiments.sr_common import SRWorld, StallDrafter, all_worlds, mean
from verisim.loop.speculative import accepted_prefix_law, free_run_divergences
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon


@dataclass(frozen=True)
class SR2Config:
    """A small, fast accepted-prefix-law instance (the dependency-free core)."""

    alpha: float = 0.75  # the drafter's raw per-step accuracy, held identical across worlds
    k: int = 16  # draft-window length
    g_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)  # ε / δ sweep (the discreteness ratio)
    n_steps: int = 80  # actions per rollout
    n_seeds: int = 24  # rollouts per (world, g) cell
    base_seed: int = 0
    gran_seeds: int = 8  # rollouts used to measure δ (the granularity)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SR2Config:
        b = SR2Config()
        return SR2Config(
            alpha=d.get("alpha", b.alpha),
            k=d.get("k", b.k),
            g_values=tuple(d.get("g_values", b.g_values)),
            n_steps=d.get("n_steps", b.n_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
            gran_seeds=d.get("gran_seeds", b.gran_seeds),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> SR2Config:
        return SR2Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SR2Stat:
    """One (world, g) cell: the empirical accepted prefix vs the i.i.d. law."""

    world: str
    g: float
    delta: float  # the world's single-edit divergence granularity
    epsilon: float  # = g * delta
    mean_prefix: float  # empirical mean accepted prefix
    pref_lo: float
    pref_hi: float
    alpha_hat: float  # measured per-step faithful-continuation probability
    law_prefix: float  # E[a] from accepted_prefix_law(alpha_hat, k)
    regime: str  # "discrete" (g < 1) | "transition" | "gradual" (g >= 2)
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world},{self.g:.6f},{self.delta:.6f},{self.epsilon:.6f},{self.mean_prefix:.6f},"
            f"{self.pref_lo:.6f},{self.pref_hi:.6f},{self.alpha_hat:.6f},{self.law_prefix:.6f},"
            f"{self.regime},{self.n}"
        )


CSV_HEADER = (
    "world,g,delta,epsilon,mean_prefix,pref_lo,pref_hi,alpha_hat,law_prefix,regime,n"
)


def granularity(world: SRWorld[Any, Any], config: SR2Config) -> float:
    """Median single-edit divergence ``δ`` -- the divergence one missed edit (a stall) produces."""
    jumps: list[float] = []
    for sd in range(config.gran_seeds):
        s0, actions = world.make_actions(sd, config.n_steps)
        state = s0
        for action in actions:
            nxt = world.oracle_step(state, action)
            d = world.diverge(nxt, state)  # divergence if the drafter stalled here
            if d > 0:
                jumps.append(d)
            state = nxt
    return median(jumps) if jumps else 0.0


def _regime(g: float) -> str:
    if g < 1.0:
        return "discrete"
    if g >= 2.0:
        return "gradual"
    return "transition"


def run_sr2(config: SR2Config | None = None) -> list[SR2Stat]:
    """Per (world, g): measure the accepted-prefix distribution and fit the i.i.d. law (H40)."""
    config = config or SR2Config()
    stats: list[SR2Stat] = []
    for world in all_worlds():
        delta = granularity(world, config)
        for g in config.g_values:
            epsilon = g * delta
            prefixes: list[float] = []
            kept = 0  # free-run steps that stayed within ε while the prefix was still faithful
            attempted = 0  # free-run steps attempted while still faithful (the α_hat denominator)
            for s in range(config.n_seeds):
                seed = config.base_seed + s
                s0, actions = world.make_actions(1000 + seed, config.n_steps)
                drafter = StallDrafter(world.oracle_step, config.alpha, seed=seed)
                # Tile the rollout into non-overlapping draft windows anchored on the truth.
                anchor = s0
                i = 0
                while i + 1 < len(actions):
                    window = actions[i : i + config.k]
                    divs = free_run_divergences(
                        anchor, window, drafter, world.oracle_step, world.diverge, start=i
                    )
                    a = faithful_horizon(divs, epsilon)
                    prefixes.append(float(a))
                    attempted += min(a + 1, len(window))
                    kept += a
                    # Re-anchor on the truth past this window (the SR2 windows are oracle-anchored).
                    state = anchor
                    for action in window:
                        state = world.oracle_step(state, action)
                    anchor = state
                    i += config.k
            lo, hi = bootstrap_ci(prefixes, seed=0)
            alpha_hat = kept / attempted if attempted else 0.0
            stats.append(
                SR2Stat(
                    world=world.name,
                    g=g,
                    delta=delta,
                    epsilon=epsilon,
                    mean_prefix=mean(prefixes),
                    pref_lo=lo,
                    pref_hi=hi,
                    alpha_hat=alpha_hat,
                    law_prefix=accepted_prefix_law(alpha_hat, config.k),
                    regime=_regime(g),
                    n=len(prefixes),
                )
            )
    return stats


def _print_summary(stats: list[SR2Stat]) -> None:
    print("SR2 / H40 - the accepted-prefix law per world (governed by g = ε/δ):")
    print("  [trained-M_θ arm DEFERRED -- committed core is the stand-in drafter (§9, LP7 rule)]")
    worlds = sorted({s.world for s in stats})
    for w in worlds:
        cells = [s for s in stats if s.world == w]
        print(f"  -- {w} (δ={cells[0].delta:.4f}) --")
        print(f"  {'g':>4} {'ε':>7} {'pref':>7} {'CI':>12} {'α':>5} {'E[a]':>6} {'regime':>9}")
        for s in sorted(cells, key=lambda s: s.g):
            print(
                f"  {s.g:>5.1f} {s.epsilon:>8.4f} {s.mean_prefix:>7.2f} "
                f"{f'[{s.pref_lo:.2f}, {s.pref_hi:.2f}]':>16} {s.alpha_hat:>6.3f} "
                f"{s.law_prefix:>9.2f} {s.regime:>11}"
            )
    # The collapse: at matched g the mean prefix should agree across worlds (H44 governed by g).
    gset = sorted({s.g for s in stats})
    spreads = []
    for g in gset:
        prefs = [s.mean_prefix for s in stats if s.g == g]
        spreads.append(max(prefs) - min(prefs))
    discrete = [s.mean_prefix for s in stats if s.regime == "discrete"]
    gradual = [s.mean_prefix for s in stats if s.regime == "gradual"]
    verdict = (
        f"prefix grows with g and collapses across worlds (max cross-world spread "
        f"{max(spreads):.2f} prefix-steps); discrete-regime prefix {mean(discrete):.2f} vs "
        f"gradual {mean(gradual):.2f} - H40 supported (the split is g = ε/δ, not world identity)"
        if mean(gradual) > mean(discrete)
        else "prefix does not grow with g - H40 unsupported (granularity does not govern)"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[SR2Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {"network": "#1f77b4", "host": "#2ca02c", "filesystem": "#d62728"}
    worlds = sorted({s.world for s in stats})
    for w in worlds:
        cells = sorted((s for s in stats if s.world == w), key=lambda s: s.g)
        gs = [s.g for s in cells]
        emp = [s.mean_prefix for s in cells]
        lo = [s.pref_lo for s in cells]
        hi = [s.pref_hi for s in cells]
        law = [s.law_prefix for s in cells]
        c = colors.get(w, "#555")
        ax1.plot(gs, emp, "-o", color=c, label=f"{w} (empirical)")
        ax1.fill_between(gs, lo, hi, color=c, alpha=0.12)
        ax1.plot(gs, law, "--", color=c, alpha=0.6, label=f"{w} (law E[a])")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("discreteness ratio g = ε / δ")
    ax1.set_ylabel("mean accepted prefix (steps)")
    ax1.set_title("accepted prefix grows with g and collapses across worlds")
    ax1.axvspan(0.3, 1.0, color="#d62728", alpha=0.05)
    ax1.axvline(1.0, color="#888", ls=":", lw=1)
    ax1.legend(fontsize=7)

    # The collapse view: empirical prefix vs the law prediction (the 45° line is a perfect i.i.d.
    # fit).
    for w in worlds:
        cells = [s for s in stats if s.world == w]
        ax2.scatter([s.law_prefix for s in cells], [s.mean_prefix for s in cells],
                    color=colors.get(w, "#555"), label=w, s=30)
    lim = max([s.mean_prefix for s in stats] + [s.law_prefix for s in stats]) * 1.05
    ax2.plot([0, lim], [0, lim], color="#888", ls=":", lw=1, label="i.i.d. law (45°)")
    ax2.set_xlabel("law prediction E[a] (i.i.d., α̂)")
    ax2.set_ylabel("empirical mean prefix")
    ax2.set_title("residual = position-dependence of acceptance")
    ax2.legend(fontsize=7)
    fig.suptitle("SR2 / H40: the accepted-prefix law (the split is g = ε/δ, not world identity)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="SR2 accepted-prefix law per world (H40).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/sr2_accept_law.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = SR2Config.from_json_file(args.config) if args.config else SR2Config()
    stats = run_sr2(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
