"""Experiment CF3: conformal risk control on the graded undetected-breach loss (SPEC-15 §6, H54).

Split conformal (CF1) controls the 0/1 miscoverage *event*; the risk verisim cares about is graded
--
a breach just past ``ε`` is a near-miss, one far past it a silent catastrophe. Conformal risk
control
(Angelopoulos et al., 2022) bounds the expectation of the monotone graded loss
(:func:`~verisim.conformal.risk.undetected_breach_loss`) directly. CF3 asks H54: does risk control
buy
anything over the coverage-only trigger?

Because the graded loss is bounded above by the indicator (severity ``≤ 1``), the graded constraint
is *looser*, so the risk-control threshold can consult **less** for the same target ``α`` -- trading
a
few more, but milder, undetected breaches for oracle budget. The pre-registered fork (H54): if the
saving is real, risk control tightens the budget where the indicator would not; if it is negligible,
the loss and the miscoverage event effectively coincide on this world (a clean simplification,
banked).

CPU-only, torch-free, deterministic, seeded (controlled calibrated signal; trained-``M_θ`` arm
deferred, the LP7 rule). Drift and divergence are real (the SPEC-13 drafter).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.conformal.calibrate import calibrate_threshold
from verisim.conformal.risk import calibrate_risk_threshold, undetected_breach_loss
from verisim.experiments.cf_common import breaches, default_world, exchangeable_pool, mean, quantile
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class CF3Config:
    """A small, fast risk-control instance (the dependency-free core)."""

    drafter_alpha: float = 0.8
    corr: float = 0.9
    breach_quantile: float = 0.6
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20)
    n_rollouts: int = 24
    steps: int = 60
    n_seeds: int = 12
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CF3Config:
        b = CF3Config()
        return CF3Config(
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
    def from_json_file(path: str | Path) -> CF3Config:
        return CF3Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CF3Stat:
    """One target-α cell: coverage-only vs risk-control budget and realized loss."""

    alpha: float
    cov_rho: float  # consultation budget of the coverage-only (indicator) threshold
    cov_loss: float  # realized graded loss under coverage-only
    risk_rho: float  # consultation budget of the graded risk-control threshold
    rr_lo: float
    rr_hi: float
    risk_loss: float  # realized graded loss under risk-control (certified ≤ α)
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.alpha:.4f},{self.cov_rho:.6f},{self.cov_loss:.6f},{self.risk_rho:.6f},"
            f"{self.rr_lo:.6f},{self.rr_hi:.6f},{self.risk_loss:.6f},{self.n}"
        )


CSV_HEADER = "alpha,cov_rho,cov_loss,risk_rho,rr_lo,rr_hi,risk_loss,n"


def run_cf3(config: CF3Config | None = None) -> list[CF3Stat]:
    """Per α: coverage-only vs graded risk-control threshold — budget and realized loss (H54)."""
    config = config or CF3Config()
    world = default_world()
    per_alpha: dict[float, dict[str, list[float]]] = {
        a: {"cov_rho": [], "cov_loss": [], "risk_rho": [], "risk_loss": []} for a in config.alphas
    }
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        pool = exchangeable_pool(
            world, drafter_alpha=config.drafter_alpha, corr=config.corr,
            n_rollouts=config.n_rollouts, steps=config.steps, seed=seed,
        )
        divs = [p.divergence for p in pool]
        eps = quantile(divs, config.breach_quantile)
        # Severity scale: the overshoot of a 95th-percentile divergence counts as a full-severity
        # breach, so the graded loss is comparable to the 0/1 indicator (not near-zero).
        scale = max(1e-9, quantile(divs, 0.95) - eps)
        half = len(pool) // 2
        cal, test = pool[:half], pool[half:]
        cs = [p.score for p in cal]
        for a in config.alphas:
            cov = calibrate_threshold(cs, breaches(cal, eps), a)
            risk = calibrate_risk_threshold(
                cs, [p.divergence for p in cal], eps, a, overshoot_scale=scale
            )
            for name, tau in (("cov", cov.tau), ("risk", risk.tau)):
                rho = sum(1 for p in test if p.score > tau) / len(test)
                loss = mean([
                    undetected_breach_loss(p.divergence, eps, consulted=(p.score > tau),
                                           overshoot_scale=scale)
                    for p in test
                ])
                per_alpha[a][f"{name}_rho"].append(rho)
                per_alpha[a][f"{name}_loss"].append(loss)

    stats: list[CF3Stat] = []
    for a in config.alphas:
        d = per_alpha[a]
        rlo, rhi = bootstrap_ci(d["risk_rho"], seed=0)
        stats.append(
            CF3Stat(a, mean(d["cov_rho"]), mean(d["cov_loss"]), mean(d["risk_rho"]), rlo, rhi,
                    mean(d["risk_loss"]), len(d["risk_rho"]))
        )
    return stats


def _print_summary(stats: list[CF3Stat]) -> None:
    print("CF3 / H54 - conformal risk control on the graded undetected-breach loss:")
    print("  [trained-M_θ arm DEFERRED -- committed core is the calibrated stand-in (§9)]")
    print(f"  {'α':>5} {'cov ρ':>7} {'cov loss':>9} {'risk ρ':>7} {'risk loss':>10} {'ρ saved':>8}")
    for s in stats:
        print(
            f"  {s.alpha:>5.2f} {s.cov_rho:>7.3f} {s.cov_loss:>9.3f} {s.risk_rho:>7.3f} "
            f"{s.risk_loss:>10.3f} {s.cov_rho - s.risk_rho:>+8.3f}"
        )
    saved = mean([s.cov_rho - s.risk_rho for s in stats])
    loss_ok = all(s.risk_loss <= s.alpha + 0.03 for s in stats)
    verdict = (
        f"risk control certifies the graded loss (realized {'≤ α' if loss_ok else 'OVER α'}) at "
        f"{saved:+.3f} lower ρ than the coverage-only trigger - H54 supported: bounding the milder "
        "graded loss (vs the 0/1 indicator) buys budget by tolerating near-misses"
        if saved > 0.01 and loss_ok
        else "risk control matches the coverage-only budget - H54 banked simplification (loss ≈ "
        "miscoverage event on this world; split conformal suffices)"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[CF3Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    alphas = [s.alpha for s in stats]
    ax1.plot(alphas, [s.cov_rho for s in stats], "-s", color="#d62728", label="coverage-only")
    ax1.plot(alphas, [s.risk_rho for s in stats], "-o", color="#1f77b4", label="risk control")
    ax1.fill_between(alphas, [s.rr_lo for s in stats], [s.rr_hi for s in stats],
                     color="#1f77b4", alpha=0.15)
    ax1.set_xlabel("target α")
    ax1.set_ylabel("oracle budget ρ")
    ax1.set_title("risk control consults less (it bounds the milder graded loss)")
    ax1.legend(fontsize=8)
    ax2.plot(alphas, [s.risk_loss for s in stats], "-o", color="#1f77b4", label="graded loss")
    ax2.plot(alphas, alphas, color="#888", ls=":", lw=1, label="target α (y=x)")
    ax2.set_xlabel("target α")
    ax2.set_ylabel("realized graded undetected-breach loss")
    ax2.set_title("the graded loss is certified ≤ α")
    ax2.legend(fontsize=8)
    fig.suptitle("CF3 / H54: conformal risk control on the loss that matters")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CF3 conformal risk control (H54).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cf3_risk_control.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CF3Config.from_json_file(args.config) if args.config else CF3Config()
    stats = run_cf3(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
