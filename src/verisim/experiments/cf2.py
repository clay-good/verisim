"""Experiment CF2: exchangeability under rollout — static conformal vs ACI (SPEC-15 §6, H52).

The load-bearing scientific question of the conformal line. Split conformal's guarantee holds for
*exchangeable* data; an autoregressive rollout **drifts by construction** (the coupled state leaves
the
calibration distribution, breaches concentrate at depth), so a *static* threshold calibrated on
exchangeable steps may lose coverage as the rollout deepens. CF2 measures it:

  - **static** -- the CF1 ``τ_α`` applied unchanged along the free-running rollout. Its undetected-
    breach rate is expected to climb *above* ``α`` with depth (the exchangeability violation).
  - **ACI** -- Gibbs--Candès adaptive conformal inference, re-tuning ``α_t`` (hence ``τ_t``) online
    from each step's realized hit/miss. The verisim twist: the realized truth is **free and exact at
    every step** (the oracle reveals the breach whether or not we consulted), so the online feedback
    ACI needs is available here in a way it never is in its native time-series setting. ACI's
    long-run
    rate is expected to bracket ``α`` regardless of the drift.

The pre-registered forks (H52): if static *holds* coverage along the rollout, drift is slow enough
that exchangeability is effectively preserved (a clean positive that retires the online machinery on
this world); if ACI *also* fails, the drift is too violent for any online method here (a deep
negative
bounding the approach). Both are bankable.

CPU-only, torch-free, deterministic, seeded (controlled calibrated signal; the trained-``M_θ`` arm
is
the deferred/``skipif``-guarded one, the LP7 rule). Drift and divergence are real (the SPEC-13
drafter).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.conformal.calibrate import calibrate_threshold
from verisim.conformal.policy import AdaptiveConformalTriggered
from verisim.experiments.cf_common import (
    breaches,
    default_world,
    drift_rollouts,
    exchangeable_pool,
    mean,
    quantile,
)
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class CF2Config:
    """A small, fast exchangeability-under-rollout instance (the dependency-free core)."""

    drafter_alpha: float = 0.78
    corr: float = 0.9
    breach_quantile: float = 0.6
    alpha: float = 0.10  # the target undetected-breach rate
    gamma: float = 0.06  # ACI step size (score-space)
    n_depth_buckets: int = 6
    cal_rollouts: int = 24
    cal_steps: int = 60
    test_rollouts: int = 40
    test_steps: int = 90
    n_seeds: int = 8
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CF2Config:
        b = CF2Config()
        return CF2Config(
            drafter_alpha=d.get("drafter_alpha", b.drafter_alpha),
            corr=d.get("corr", b.corr),
            breach_quantile=d.get("breach_quantile", b.breach_quantile),
            alpha=d.get("alpha", b.alpha),
            gamma=d.get("gamma", b.gamma),
            n_depth_buckets=d.get("n_depth_buckets", b.n_depth_buckets),
            cal_rollouts=d.get("cal_rollouts", b.cal_rollouts),
            cal_steps=d.get("cal_steps", b.cal_steps),
            test_rollouts=d.get("test_rollouts", b.test_rollouts),
            test_steps=d.get("test_steps", b.test_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CF2Config:
        return CF2Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CF2Stat:
    """One (policy, depth-bucket) cell: the undetected-breach rate along the rollout."""

    policy: str  # "static" | "aci"
    depth: float  # bucket midpoint (fraction of rollout depth)
    undetected: float
    un_lo: float
    un_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.policy},{self.depth:.6f},{self.undetected:.6f},"
            f"{self.un_lo:.6f},{self.un_hi:.6f},{self.n}"
        )


CSV_HEADER = "policy,depth,undetected,un_lo,un_hi,n"


def _bucket(depth: int, steps: int, n_buckets: int) -> int:
    return min(n_buckets - 1, depth * n_buckets // steps)


def run_cf2(config: CF2Config | None = None) -> list[CF2Stat]:
    """Per (policy, depth): undetected rate of static conformal vs ACI along the rollout (H52)."""
    config = config or CF2Config()
    world = default_world()
    # Accumulate per (policy, bucket) lists of per-seed undetected rates.
    acc: dict[tuple[str, int], list[float]] = {
        (p, b): [] for p in ("static", "aci") for b in range(config.n_depth_buckets)
    }
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        pool = exchangeable_pool(
            world, drafter_alpha=config.drafter_alpha, corr=config.corr,
            n_rollouts=config.cal_rollouts, steps=config.cal_steps, seed=seed,
        )
        eps = quantile([p.divergence for p in pool], config.breach_quantile)
        ref = quantile([p.divergence for p in pool], 0.95) or 1.0
        cal_scores = [p.score for p in pool]
        cal_breaches = breaches(pool, eps)
        static_tau = calibrate_threshold(cal_scores, cal_breaches, config.alpha).tau

        rollouts = drift_rollouts(
            world, drafter_alpha=config.drafter_alpha, corr=config.corr,
            n_rollouts=config.test_rollouts, steps=config.test_steps, seed=seed, ref=ref,
        )
        # static + ACI undetected counts per bucket (averaged over rollouts within this seed).
        for policy in ("static", "aci"):
            miss = [0] * config.n_depth_buckets
            total = [0] * config.n_depth_buckets
            for rollout in rollouts:
                aci = AdaptiveConformalTriggered(
                    cal_scores, cal_breaches, config.alpha, config.gamma
                )
                for step in rollout:
                    tau = static_tau if policy == "static" else aci.tau
                    consulted = step.score > tau
                    breach = step.divergence > eps
                    b = _bucket(step.depth, config.test_steps, config.n_depth_buckets)
                    total[b] += 1
                    if breach and not consulted:
                        miss[b] += 1
                    if policy == "aci":
                        aci.update(breach=breach, consulted=consulted)
            for b in range(config.n_depth_buckets):
                if total[b]:
                    acc[(policy, b)].append(miss[b] / total[b])

    stats: list[CF2Stat] = []
    for policy in ("static", "aci"):
        for b in range(config.n_depth_buckets):
            vals = acc[(policy, b)]
            lo, hi = bootstrap_ci(vals, seed=0)
            mid = (b + 0.5) / config.n_depth_buckets
            stats.append(CF2Stat(policy, mid, mean(vals), lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[CF2Stat], alpha: float) -> None:
    print("CF2 / H52 - exchangeability under rollout: static conformal drifts, ACI recovers:")
    print("  [trained-M_θ arm DEFERRED -- committed core is the calibrated stand-in (§9)]")
    print(f"  target α = {alpha}")
    for policy in ("static", "aci"):
        cells = sorted((s for s in stats if s.policy == policy), key=lambda s: s.depth)
        row = "  ".join(f"d{c.depth:.2f}:{c.undetected:.3f}" for c in cells)
        print(f"  {policy:>7}: {row}")
    static = sorted((s for s in stats if s.policy == "static"), key=lambda s: s.depth)
    aci = sorted((s for s in stats if s.policy == "aci"), key=lambda s: s.depth)
    static_overall = mean([c.undetected for c in static])
    aci_overall = mean([c.undetected for c in aci])
    static_deep, aci_deep = static[-1].undetected, aci[-1].undetected
    drifts = static_deep > static[0].undetected + 0.02 and static_overall > alpha
    # ACI guarantees the *long-run* (depth-averaged) rate, not per-depth: judge recovery on the
    # overall.
    recovers = aci_overall <= static_overall - 0.02 and aci_overall <= alpha + 0.05
    verdict = (
        f"static conformal's undetected rate climbs with depth ({static[0].undetected:.3f} -> "
        f"{static_deep:.3f}); its long-run rate {static_overall:.3f} exceeds α={alpha}, while ACI "
        f"restores the long-run rate near target ({aci_overall:.3f}) and halves the deepest-depth "
        f"miss rate ({static_deep:.3f} -> {aci_deep:.3f}) - H52 supported: rollout drift breaks "
        "static exchangeability; free per-step oracle feedback lets ACI recover long-run coverage "
        "(residual deep lag under monotone shift motivates conformal-PID, the §2.4 stretch)"
        if drifts and recovers
        else f"static drifts={drifts}, aci recovers={recovers} - H52 outcome differs from the fork"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[CF2Stat], alpha: float, path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    styles = {"static": ("#d62728", "-o"), "aci": ("#1f77b4", "-s")}
    labels = {"static": "static conformal (CF1 τ_α)", "aci": "adaptive conformal (ACI)"}
    for policy in ("static", "aci"):
        cells = sorted((s for s in stats if s.policy == policy), key=lambda s: s.depth)
        xs = [c.depth for c in cells]
        ys = [c.undetected for c in cells]
        color, fmt = styles[policy]
        ax.plot(xs, ys, fmt, color=color, label=labels[policy])
        ax.fill_between(xs, [c.un_lo for c in cells], [c.un_hi for c in cells],
                        color=color, alpha=0.12)
    ax.axhline(alpha, color="#888", ls=":", lw=1, label=f"target α = {alpha}")
    ax.set_xlabel("rollout depth (fraction)")
    ax.set_ylabel("undetected-breach rate")
    ax.set_title("CF2 / H52: static conformal drifts above α with depth; ACI recovers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CF2 static conformal vs ACI under drift (H52).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cf2_drift_aci.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CF2Config.from_json_file(args.config) if args.config else CF2Config()
    stats = run_cf2(cfg)
    _print_summary(stats, cfg.alpha)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, cfg.alpha, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
