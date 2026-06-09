"""Experiment CF4: conformalizability — belief_var vs decode-entropy (SPEC-15 §6, H53).

The program's unexplained split: the host's calibrated belief-variance trigger beat fixed ~2.2×
(EH2/H9), while the flat arm's decode-entropy trigger lost to fixed and lost badly (ED2-smart). CF4
gives it one mechanism: run the *identical* conformal calibration on both signals and show that

  - a **calibrated** signal (the ``belief_var`` stand-in, ``corr ≈ 0.9``) conformalizes -- its
    score-conditional divergence rises monotonically (high score ⇒ high divergence), so the
    threshold
    certifies a target ``α`` at *far lower* ρ than fixed;
  - an **uncalibrated** signal (the decode-entropy stand-in, ``corr ≈ 0``) does not -- its
    score-conditional divergence is flat (the score does not separate within-ε from breach steps),
    so
    its conformal threshold buys *no* ρ over fixed: the guarantee still holds (conformal is valid
    for
    any signal) but only by consulting nearly everywhere.

This is the measured, mechanistic statement of "entropy is a decode artifact, not a calibrated
belief": conformal validity is signal-agnostic, conformal *efficiency* is not.

CPU-only, torch-free, deterministic, seeded (controlled signals; the trained-``M_θ`` arm is the
deferred/``skipif``-guarded one, the LP7 rule). Drift and divergence are real (the SPEC-13 drafter).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.conformal.calibrate import calibrate_threshold, undetected_rate
from verisim.experiments.cf_common import breaches, default_world, exchangeable_pool, mean, quantile
from verisim.metrics.aggregate import bootstrap_ci

ARMS: tuple[tuple[str, float], ...] = (("calibrated", 0.9), ("uncalibrated", 0.0))


@dataclass(frozen=True)
class CF4Config:
    """A small, fast conformalizability instance (the dependency-free core)."""

    drafter_alpha: float = 0.8
    breach_quantile: float = 0.6
    alpha: float = 0.05  # the target undetected-breach rate the two signals must both certify
    n_score_bins: int = 5
    n_rollouts: int = 24
    steps: int = 60
    n_seeds: int = 12
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CF4Config:
        b = CF4Config()
        return CF4Config(
            drafter_alpha=d.get("drafter_alpha", b.drafter_alpha),
            breach_quantile=d.get("breach_quantile", b.breach_quantile),
            alpha=d.get("alpha", b.alpha),
            n_score_bins=d.get("n_score_bins", b.n_score_bins),
            n_rollouts=d.get("n_rollouts", b.n_rollouts),
            steps=d.get("steps", b.steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CF4Config:
        return CF4Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CF4Stat:
    """One signal arm: coverage, ρ to certify, ρ saved over fixed, and the calibration slope."""

    arm: str
    test_undetected: float
    conformal_rho: float
    cf_lo: float
    cf_hi: float
    fixed_rho: float
    score_div_slope: float  # div(high-score bin) − div(low-score bin): the conformalizability link
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.test_undetected:.6f},{self.conformal_rho:.6f},{self.cf_lo:.6f},"
            f"{self.cf_hi:.6f},{self.fixed_rho:.6f},{self.score_div_slope:.6f},{self.n}"
        )


CSV_HEADER = "arm,test_undetected,conformal_rho,cf_lo,cf_hi,fixed_rho,score_div_slope,n"


@dataclass(frozen=True)
class ScoreBin:
    """One score-conditional divergence point (the reliability curve)."""

    arm: str
    score_bin: float
    mean_divergence: float
    div_lo: float
    div_hi: float
    n: int


def _score_divergence_curve(pool: list[Any], n_bins: int) -> list[tuple[float, float]]:
    """Mean divergence per score bin -- does a higher score mean higher divergence? (the link)."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    out: list[tuple[float, float]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        cell = [p.divergence for p in pool
                if (lo <= p.score < hi) or (b == n_bins - 1 and p.score == hi)]
        if cell:
            out.append(((lo + hi) / 2, mean(cell)))
    return out


def calibration_curves(config: CF4Config | None = None) -> list[ScoreBin]:
    """The score-conditional divergence curve per arm -- the figure's reliability panel."""
    config = config or CF4Config()
    world = default_world()
    points: list[ScoreBin] = []
    for arm, corr in ARMS:
        edges = [i / config.n_score_bins for i in range(config.n_score_bins + 1)]
        per_bin: dict[int, list[float]] = {b: [] for b in range(config.n_score_bins)}
        for s in range(config.n_seeds):
            seed = config.base_seed + s
            pool = exchangeable_pool(
                world, drafter_alpha=config.drafter_alpha, corr=corr,
                n_rollouts=config.n_rollouts, steps=config.steps, seed=seed,
            )
            for p in pool:
                for b in range(config.n_score_bins):
                    lo, hi = edges[b], edges[b + 1]
                    if (lo <= p.score < hi) or (b == config.n_score_bins - 1 and p.score == hi):
                        per_bin[b].append(p.divergence)
                        break
        for b in range(config.n_score_bins):
            if per_bin[b]:
                lo, hi = bootstrap_ci(per_bin[b], seed=0)
                points.append(
                    ScoreBin(arm, (edges[b] + edges[b + 1]) / 2, mean(per_bin[b]), lo, hi,
                             len(per_bin[b]))
                )
    return points


def run_cf4(config: CF4Config | None = None) -> list[CF4Stat]:
    """Per arm: conformal coverage, ρ-to-certify vs fixed, and the score↔divergence slope (H53)."""
    config = config or CF4Config()
    world = default_world()
    stats: list[CF4Stat] = []
    for arm, corr in ARMS:
        undet: list[float] = []
        rho: list[float] = []
        fixed: list[float] = []
        slopes: list[float] = []
        for s in range(config.n_seeds):
            seed = config.base_seed + s
            pool = exchangeable_pool(
                world, drafter_alpha=config.drafter_alpha, corr=corr,
                n_rollouts=config.n_rollouts, steps=config.steps, seed=seed,
            )
            eps = quantile([p.divergence for p in pool], config.breach_quantile)
            half = len(pool) // 2
            cal, test = pool[:half], pool[half:]
            th = calibrate_threshold([p.score for p in cal], breaches(cal, eps), config.alpha)
            ts = [p.score for p in test]
            tb = breaches(test, eps)
            breach_rate = sum(tb) / len(tb) if tb else 0.0
            undet.append(undetected_rate(ts, tb, th.tau))
            rho.append(sum(1 for x in ts if x > th.tau) / len(ts))
            fixed.append(max(0.0, 1.0 - config.alpha / breach_rate) if breach_rate > 0 else 0.0)
            curve = _score_divergence_curve(pool, config.n_score_bins)
            slopes.append(curve[-1][1] - curve[0][1] if len(curve) >= 2 else 0.0)
        clo, chi = bootstrap_ci(rho, seed=0)
        stats.append(
            CF4Stat(arm, mean(undet), mean(rho), clo, chi, mean(fixed), mean(slopes), len(rho))
        )
    return stats


def _print_summary(stats: list[CF4Stat]) -> None:
    print("CF4 / H53 - conformalizability: belief_var conformalizes, decode-entropy does not:")
    print("  [trained-M_θ arm DEFERRED -- core is the calibrated/uncalibrated stand-in (§9)]")
    for s in stats:
        print(
            f"  {s.arm:>12}: test_undet={s.test_undetected:.3f} conf ρ={s.conformal_rho:.3f} "
            f"fixed ρ={s.fixed_rho:.3f} (saved {s.fixed_rho - s.conformal_rho:+.3f})  "
            f"score↔div slope={s.score_div_slope:+.4f}"
        )
    cal = next(s for s in stats if s.arm == "calibrated")
    unc = next(s for s in stats if s.arm == "uncalibrated")
    cal_saves = cal.fixed_rho - cal.conformal_rho
    unc_saves = unc.fixed_rho - unc.conformal_rho
    verdict = (
        f"both signals hit coverage (conformal is valid for any signal), but the calibrated signal "
        f"saves {cal_saves:+.2f} ρ over fixed while the uncalibrated saves only {unc_saves:+.2f} "
        f"(slope {unc.score_div_slope:+.3f} vs calibrated {cal.score_div_slope:+.3f}) "
        "- H53 supported: conformal *validity* is signal-agnostic, conformal *efficiency* is not "
        "(the EH2-yes / ED2-smart-no mechanism)"
        if cal_saves > unc_saves + 0.1
        else "the two signals conformalize alike - H53 unsupported on this drafter"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[CF4Stat], curves: list[ScoreBin], path: Path) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {"calibrated": "#1f77b4", "uncalibrated": "#d62728"}
    # Panel 1: ρ to certify the same α, per signal, vs the fixed baseline.
    arms = [s.arm for s in stats]
    x = range(len(arms))
    ax1.bar([i - 0.2 for i in x], [s.conformal_rho for s in stats], 0.4,
            color=[colors[a] for a in arms], label="conformal")
    ax1.bar([i + 0.2 for i in x], [s.fixed_rho for s in stats], 0.4, color="#bbb", label="fixed")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(arms)
    ax1.set_ylabel("oracle budget ρ to certify α")
    ax1.set_title("calibrated conformalizes (saves ρ); uncalibrated ≈ fixed")
    ax1.legend(fontsize=8)
    # Panel 2: the reliability curve -- score-conditional divergence (monotone iff conformalizable).
    for arm in ("calibrated", "uncalibrated"):
        pts = sorted((p for p in curves if p.arm == arm), key=lambda p: p.score_bin)
        xs = [p.score_bin for p in pts]
        ys = [p.mean_divergence for p in pts]
        ax2.plot(xs, ys, "-o", color=colors[arm], label=arm)
        ax2.fill_between(xs, [p.div_lo for p in pts], [p.div_hi for p in pts],
                         color=colors[arm], alpha=0.12)
    ax2.set_xlabel("score bin")
    ax2.set_ylabel("mean oracle divergence")
    ax2.set_title("the link: monotone (calibrated) vs flat (uncalibrated)")
    ax2.legend(fontsize=8)
    fig.suptitle("CF4 / H53: conformal validity is signal-agnostic, efficiency is not")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CF4 conformalizability (H53).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cf4_signal_split.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CF4Config.from_json_file(args.config) if args.config else CF4Config()
    stats = run_cf4(cfg)
    curves = calibration_curves(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, curves, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
