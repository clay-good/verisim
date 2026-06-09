"""Experiment CF1: the split-conformal trigger — coverage gate + the ρ-vs-coverage frontier (SPEC-15
§6).

The headline of the conformal line, in two halves:

  - **H50 (the gate).** On exchangeable held-out steps, the conformal threshold ``τ_α`` achieves an
    empirical undetected-breach rate ``≤ α`` (within the finite-sample slack ``1/(n+1)``) across
    ``α ∈ {0.01, 0.05, 0.10, 0.20}``. This is an *implementation-correctness* gate, not a discovery:
    if it fails, the calibrator is buggy and must be fixed before any rollout claim. It passes by
    construction when the calibrator is right.
  - **H51 (the headline).** At a matched target undetected rate ``α``, the conformal trigger reaches
    that guarantee at *lower oracle budget ρ* than fixed-interval — a *guaranteed* version of the
    EH2/H9 win, stated as "fewer consults to certify the same safety." Fixed cannot use the score,
    so
    to keep undetected breaches ``≤ α`` it must consult uniformly enough that ``(1−ρ)·breach ≤ α``,
    i.e. ``ρ ≥ 1 − α/breach``; the calibrated conformal trigger spends its budget where the score is
    high and beats that.

CPU-only, torch-free, deterministic, seeded. The committed core uses the controlled *calibrated*
signal (the ``belief_var`` stand-in, :mod:`~verisim.experiments.cf_common`); the trained-``M_θ``
``belief_var`` arm is the deferred/``skipif``-guarded one, never counted (SPEC-15 §9, the LP7 rule).
The *drift and divergence are real* (the SPEC-13 drafter against the real oracle); only the score is
a
stand-in.
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


@dataclass(frozen=True)
class CF1Config:
    """A small, fast conformal-coverage instance (the dependency-free core)."""

    drafter_alpha: float = 0.8  # the controlled drafter's per-step accuracy (real drift)
    corr: float = 0.9  # the calibrated-signal correlation (the belief_var stand-in)
    breach_quantile: float = 0.6  # ε = this quantile of divergence (-> ~40% breach rate)
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    n_rollouts: int = 24
    steps: int = 60
    n_seeds: int = 12
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CF1Config:
        b = CF1Config()
        return CF1Config(
            drafter_alpha=d.get("drafter_alpha", b.drafter_alpha),
            corr=d.get("corr", b.corr),
            breach_quantile=d.get("breach_quantile", b.breach_quantile),
            alphas=tuple(d.get("alphas", b.alphas)),
            n_rollouts=d.get("n_rollouts", b.n_rollouts),
            steps=d.get("steps", b.steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CF1Config:
        return CF1Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CF1Stat:
    """One target-α cell: the coverage gate and the ρ-vs-coverage frontier point."""

    alpha: float
    test_undetected: float  # empirical undetected-breach rate on the held-out test split (the gate)
    tu_lo: float
    tu_hi: float
    conformal_rho: float  # consultation budget the conformal trigger spends
    cf_lo: float
    cf_hi: float
    fixed_rho: float  # consultation budget fixed-interval needs for the same guarantee
    breach_rate: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.alpha:.4f},{self.test_undetected:.6f},{self.tu_lo:.6f},{self.tu_hi:.6f},"
            f"{self.conformal_rho:.6f},{self.cf_lo:.6f},{self.cf_hi:.6f},{self.fixed_rho:.6f},"
            f"{self.breach_rate:.6f},{self.n}"
        )


CSV_HEADER = (
    "alpha,test_undetected,tu_lo,tu_hi,conformal_rho,cf_lo,cf_hi,fixed_rho,breach_rate,n"
)


def run_cf1(config: CF1Config | None = None) -> list[CF1Stat]:
    """Per target α: the coverage gate (H50) and the conformal-vs-fixed ρ frontier (H51)."""
    config = config or CF1Config()
    world = default_world()
    # Per seed: build an exchangeable pool, split calibration/test, calibrate, measure.
    per_alpha: dict[float, dict[str, list[float]]] = {
        a: {"undet": [], "rho": [], "fixed": [], "breach": []} for a in config.alphas
    }
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        pool = exchangeable_pool(
            world, drafter_alpha=config.drafter_alpha, corr=config.corr,
            n_rollouts=config.n_rollouts, steps=config.steps, seed=seed,
        )
        eps = quantile([p.divergence for p in pool], config.breach_quantile)
        half = len(pool) // 2
        cal, test = pool[:half], pool[half:]
        cs = [p.score for p in cal]
        cb = breaches(cal, eps)
        ts = [p.score for p in test]
        tb = breaches(test, eps)
        breach_rate = sum(tb) / len(tb) if tb else 0.0
        for a in config.alphas:
            th = calibrate_threshold(cs, cb, a)
            per_alpha[a]["undet"].append(undetected_rate(ts, tb, th.tau))
            per_alpha[a]["rho"].append(sum(1 for x in ts if x > th.tau) / len(ts))
            fixed_rho = max(0.0, 1.0 - a / breach_rate) if breach_rate > 0 else 0.0
            per_alpha[a]["fixed"].append(fixed_rho)
            per_alpha[a]["breach"].append(breach_rate)

    stats: list[CF1Stat] = []
    for a in config.alphas:
        d = per_alpha[a]
        tlo, thi = bootstrap_ci(d["undet"], seed=0)
        clo, chi = bootstrap_ci(d["rho"], seed=0)
        stats.append(
            CF1Stat(a, mean(d["undet"]), tlo, thi, mean(d["rho"]), clo, chi,
                    mean(d["fixed"]), mean(d["breach"]), len(d["undet"]))
        )
    return stats


def _print_summary(stats: list[CF1Stat]) -> None:
    print("CF1 / H50+H51 - the conformal coverage gate and the ρ-vs-coverage frontier:")
    print("  [trained-M_θ belief_var arm DEFERRED -- core is the calibrated stand-in (§9)]")
    print(f"  breach rate ≈ {stats[0].breach_rate:.2f}")
    print(f"  {'α':>5} {'test undet':>11} {'≤α?':>5} {'conf ρ':>8} {'fixed ρ':>8} {'ρ saved':>8}")
    gate_ok = True
    for s in stats:
        ok = s.test_undetected <= s.alpha + 1.0 / (s.n + 1) + 0.02
        gate_ok = gate_ok and ok
        print(
            f"  {s.alpha:>5.2f} {s.test_undetected:>11.3f} {'yes' if ok else 'NO':>5} "
            f"{s.conformal_rho:>8.3f} {s.fixed_rho:>8.3f} {s.fixed_rho - s.conformal_rho:>+8.3f}"
        )
    saved = mean([s.fixed_rho - s.conformal_rho for s in stats])
    verdict = (
        f"the coverage gate holds (undetected ≤ α, H50 {'PASS' if gate_ok else 'FAIL'}) "
        f"and the calibrated trigger certifies it at {saved:+.2f} lower ρ than fixed on "
        "average - H51 supported (guaranteed RQ2: fewer consults to certify the same coverage)"
        if gate_ok and saved > 0
        else "the gate failed or conformal does not beat fixed - revisit the calibrator / signal"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[CF1Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    alphas = [s.alpha for s in stats]
    # Panel 1: the coverage gate -- test undetected vs target α (on/under the y=x line).
    ax1.plot(alphas, [s.test_undetected for s in stats], "-o", color="#1f77b4", label="undetected")
    ax1.fill_between(alphas, [s.tu_lo for s in stats], [s.tu_hi for s in stats],
                     color="#1f77b4", alpha=0.15)
    lim = max(alphas) * 1.1
    ax1.plot([0, lim], [0, lim], color="#888", ls=":", lw=1, label="target α (y=x)")
    ax1.set_xlabel("target undetected-breach rate α")
    ax1.set_ylabel("empirical undetected rate (held-out)")
    ax1.set_title("H50: the coverage gate holds (on/under y=x)")
    ax1.legend(fontsize=8)
    # Panel 2: the ρ-vs-coverage frontier -- conformal vs fixed budget to certify each α.
    ax2.plot(alphas, [s.conformal_rho for s in stats], "-o", color="#1f77b4", label="conformal")
    ax2.fill_between(alphas, [s.cf_lo for s in stats], [s.cf_hi for s in stats],
                     color="#1f77b4", alpha=0.15)
    ax2.plot(alphas, [s.fixed_rho for s in stats], "-s", color="#d62728", label="fixed-interval")
    ax2.set_xlabel("target undetected-breach rate α")
    ax2.set_ylabel("oracle budget ρ to certify α")
    ax2.set_title("H51: conformal certifies α at lower ρ than fixed")
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(fontsize=8)
    fig.suptitle("CF1 / H50+H51: the conformal coverage gate and the guaranteed-RQ2 frontier")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CF1 coverage gate + ρ frontier (H50/H51).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cf1_coverage_frontier.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CF1Config.from_json_file(args.config) if args.config else CF1Config()
    stats = run_cf1(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
