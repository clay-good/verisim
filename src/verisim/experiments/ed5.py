"""ED5 -- consistency vs bit-faithful horizon (H19) + the competitive-ratio fit (H18).

SPEC-7 §12 (DS8) names two things ED5 must measure, both on the existing dependency-free apparatus
(the DS0-increment-1 replicated-KV world + the synthetic tunable-noise proposer
:class:`~verisim.distloop.model.DistNoisyModel`), so the result is GPU-free and runs in CI:

  - **H19 -- consistency-faithful outlasts bit-faithful.** Under a weak consistency model many
    *bit*-states map to one *admissible consistency view*, so a model can predict the cluster's
    observable consistency behavior (§9.1, the headline-new metric) for longer than it predicts the
    exact replica bytes. ED5 measures both horizons on the *same free-running rollout* (ρ=0, which
    exposes the model, not the loop) and reports the gap. The world's mechanism is exact: the
    asynchronous-replication **in-flight message** is the gap -- a corrupted in-flight payload is
    immediately *bit*-visible (the message fact differs) but **consistency-invisible** (the
    per-object converged/split view reads only replicas) until that message is delivered by
    ``advance`` and writes a replica. So the ``subtle`` (in-flight) error class produces a real gap
    (consistency outlasts bit), while the ``gross`` (durable-replica) error class breaks both at
    once (the control where the two horizons coincide). The mode-dependent verdict mirrors ED1/ED3.

  - **H18 -- the loop is a learning-augmented algorithm with a competitive ratio.** Propose-verify-
    correct is "algorithms with predictions": the model is the predictor, the bit-exact oracle is
    the worst-case-safe fallback, ``ρ`` is how often the fallback fires (`DD-D4`, §2.4, §9.3). The
    figure of merit is the **competitive ratio** ``H_ε(ρ) / H_ε(ρ=1)`` against the full-oracle
    ceiling, and the learning-augmented property is that it **degrades gracefully with prediction
    error**: recovering ``1`` when the model is perfect (no fallback needed) and falling toward the
    free-running floor as the model becomes useless. ED5 fits the ratio across ``ρ × prediction
    error`` (the noise dial) at the bit-exact tier (where ``ρ`` maps linearly to oracle-dollars, so
    the quarter ρ *is* the ``B/4`` quarter budget ED2 reads). The honest reading on this world has
    two halves: the graceful-degradation-with-error half is **confirmed** (the ratio at any budget
    is monotone in the model's competence), while the bounded-ratio-at-*sub-linear*-cost half
    reproduces the program's recurring **floor→cliff / no-knee** negative -- at the quarter budget
    the ratio sits near the
    floor, because a discrete-error world's faithful horizon is a *prefix* property that only
    near-full consultation protects (the same finding as E1/EN1/EH1/ED1, now in competitive-ratio
    form). Both halves are reported, neither hidden.

The committed sweep runs in milliseconds on CPU (no torch); CI runs the smoke instance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import (
    DistNoisyModel,
    FixedTierPolicy,
    budget_for_rho,
    run_dist_rollout,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon

ERROR_MODES: tuple[str, ...] = ("gross", "subtle")


@dataclass(frozen=True)
class ED5Config:
    name: str = "ed5-dist"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    driver: str = "contention"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105, 106, 107)
    n_steps: int = 40
    epsilon: float = 0.0
    #: H19 -- the per-mode free-running noise (the model is exposed, not the loop).
    h19_noise: float = 0.4
    modes: tuple[str, ...] = ERROR_MODES
    #: H18 -- the prediction-error axis (per-step corruption probability) and the budget axis.
    h18_mode: str = "gross"  # the prime-directive curve_mode (the clean-interior error class)
    h18_noises: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    rhos: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    quarter_rho: float = 0.25  # at the bit-exact tier ρ maps linearly to $: this is the B/4 budget

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED5Config:
        b = ED5Config()
        return ED5Config(
            name=d.get("name", b.name),
            driver=d.get("driver", b.driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            epsilon=d.get("epsilon", b.epsilon),
            h19_noise=d.get("h19_noise", b.h19_noise),
            modes=tuple(d.get("modes", b.modes)),
            h18_mode=d.get("h18_mode", b.h18_mode),
            h18_noises=tuple(d.get("h18_noises", b.h18_noises)),
            rhos=tuple(d.get("rhos", b.rhos)),
            quarter_rho=d.get("quarter_rho", b.quarter_rho),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED5Config:
        return ED5Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED5Result:
    """The ED5 deliverables: the H19 consistency-vs-bit gap and the H18 competitive-ratio fit."""

    #: per mode: mean bit-faithful + consistency-faithful horizon (free-running), the gap, CIs.
    h19: list[dict[str, Any]] = field(default_factory=list)
    #: per (noise, ρ): mean H_ε and the competitive ratio H_ε(ρ)/ceiling, with CIs.
    h18: list[dict[str, Any]] = field(default_factory=list)
    #: the H18 verdict: ratio at the quarter budget per noise + whether it degrades gracefully.
    h18_verdict: dict[str, Any] = field(default_factory=dict)


def _eval_actions(
    oracle: DistOracle, config: DistConfig, driver: str, seed: int, n: int
) -> list[DistAction]:
    drv = DistDriver(driver, config, random.Random(seed))
    state = DistributedState.initial(config)
    actions: list[DistAction] = []
    for _ in range(n):
        action = drv.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def _both_horizons(
    oracle: DistOracle, cfg: ED5Config, mode: str, noise: float, rho: float, eval_seed: int
) -> tuple[int, int]:
    """One rollout: ``(bit_faithful_horizon, consistency_faithful_horizon)`` at (mode, noise, ρ).

    Both come from the *same* tiered-loop rollout: the bit-exact ``divergences`` give the
    bit-faithful horizon, and the runner-recorded ``consistency_divergences`` (1 - the §9.1
    consistency-faithfulness) give the consistency-faithful horizon under the same ``ε``.
    """
    s0 = DistributedState.initial(cfg.dist)
    actions = _eval_actions(oracle, cfg.dist, cfg.driver, eval_seed, cfg.n_steps)
    model = DistNoisyModel(oracle, noise=noise, mode=mode, rng=random.Random(eval_seed + 7))
    record = run_dist_rollout(
        model, oracle, s0, actions, fixed_interval_for_rho(rho), epsilon=cfg.epsilon,
        config=cfg.dist, tier_policy=FixedTierPolicy("bit_exact"),
        budget=budget_for_rho(rho, len(actions)), seed=eval_seed,
    )
    bit_h = record.faithful_horizon
    cons_h = faithful_horizon(record.config["consistency_divergences"], cfg.epsilon)
    return bit_h, cons_h


def run_ed5(config: ED5Config | None = None, *, oracle: DistOracle | None = None) -> ED5Result:
    """Run ED5: the H19 consistency-vs-bit free-running gap + the H18 competitive-ratio fit."""
    config = config or ED5Config()
    oracle = oracle or ReferenceDistOracle(config.dist)
    result = ED5Result()

    # --- H19: consistency-faithful vs bit-faithful horizon, free-running (ρ=0) --------------------
    for mode in config.modes:
        cells = [_both_horizons(oracle, config, mode, config.h19_noise, 0.0, s)
                 for s in config.eval_seeds]
        bit = [float(b) for b, _ in cells]
        cons = [float(c) for _, c in cells]
        bit_lo, bit_hi = bootstrap_ci(bit, seed=0)
        cons_lo, cons_hi = bootstrap_ci(cons, seed=0)
        gap = [c - b for b, c in zip(bit, cons, strict=True)]
        gap_lo, gap_hi = bootstrap_ci(gap, seed=0)
        result.h19.append({
            "mode": mode,
            "bit_h": fmean(bit), "bit_lo": bit_lo, "bit_hi": bit_hi,
            "cons_h": fmean(cons), "cons_lo": cons_lo, "cons_hi": cons_hi,
            "gap": fmean(gap), "gap_lo": gap_lo, "gap_hi": gap_hi,
            # H19 holds for this mode iff consistency materially outlasts bit (CI on the gap > 0)
            "consistency_outlasts": gap_lo > 0.0,
        })

    # --- H18: the competitive ratio H_ε(ρ)/ceiling across ρ × prediction error -------------------
    ceiling = float(config.n_steps)  # ρ=1 + bit-exact + HardReset reproduces truth every step
    ratio_at_quarter: dict[float, float] = {}
    for noise in config.h18_noises:
        for rho in config.rhos:
            hs = [float(_both_horizons(oracle, config, config.h18_mode, noise, rho, s)[0])
                  for s in config.eval_seeds]
            lo, hi = bootstrap_ci(hs, seed=0)
            h = fmean(hs)
            result.h18.append({
                "noise": noise, "rho": rho, "h_eps": h,
                "ratio": h / ceiling if ceiling else float("nan"),
                "ci_lo": lo, "ci_hi": hi,
            })
        # the competitive ratio at the sub-linear quarter budget (the H18 / DD-D4 readout)
        q = [float(_both_horizons(oracle, config, config.h18_mode, noise, config.quarter_rho, s)[0])
             for s in config.eval_seeds]
        ratio_at_quarter[noise] = fmean(q) / ceiling if ceiling else float("nan")

    ordered = [ratio_at_quarter[n] for n in config.h18_noises]
    # learning-augmented graceful degradation: the quarter-budget ratio is non-increasing in error.
    monotone = all(a >= b - 1e-9 for a, b in pairwise(ordered))
    result.h18_verdict = {
        "ceiling": ceiling,
        "quarter_rho": config.quarter_rho,
        "ratio_at_quarter": {str(n): ratio_at_quarter[n] for n in config.h18_noises},
        "degrades_gracefully": monotone,
        "ratio_perfect_model": ratio_at_quarter[min(config.h18_noises)],
        "ratio_useless_model": ratio_at_quarter[max(config.h18_noises)],
    }
    return result


CSV_HEADER = "panel,mode,noise,rho,bit_h,cons_h,gap,h_eps,ratio,ci_lo,ci_hi"


def write_csv(result: ED5Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in result.h19:
        rows.append(f"h19,{r['mode']},,,{r['bit_h']:.4f},{r['cons_h']:.4f},{r['gap']:.4f},,,"
                    f"{r['gap_lo']:.4f},{r['gap_hi']:.4f}")
    for c in result.h18:
        rows.append(f"h18,,{c['noise']},{c['rho']},,,,{c['h_eps']:.4f},{c['ratio']:.4f},"
                    f"{c['ci_lo']:.4f},{c['ci_hi']:.4f}")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED5 (consistency-vs-bit horizon H19 + competitive-ratio fit H18)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed5.csv")
    parser.add_argument("--plot", type=str, default="figures/ed5.png")
    args = parser.parse_args()
    config = ED5Config.from_json_file(args.config) if args.config else ED5Config()
    result = run_ed5(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  H19 (free-running consistency vs bit horizon):")
    for r in result.h19:
        verdict = "consistency OUTLASTS bit" if r["consistency_outlasts"] else "no gap (control)"
        print(f"    [{r['mode']:6s}] bit H={r['bit_h']:.1f}  cons H={r['cons_h']:.1f}  "
              f"gap={r['gap']:.1f} [{r['gap_lo']:.1f},{r['gap_hi']:.1f}] → {verdict}")
    v = result.h18_verdict
    print(f"  H18 competitive ratio @ quarter budget (ρ={v['quarter_rho']}):")
    for n, ratio in v["ratio_at_quarter"].items():
        print(f"    noise={n}: ratio={ratio:.2f}")
    grace = "CONFIRMED" if v["degrades_gracefully"] else "not monotone"
    print(f"    graceful degradation with prediction error: {grace}")
    try:
        from figures.plot_ed5 import plot_ed5

        plot_ed5(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
