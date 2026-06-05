"""ED2 -- when × which-tier at **equal oracle-dollar budget** (SPEC-7 §12, §10.2; DS7).

ED1 (:mod:`verisim.experiments.ed1`) plots the prime-directive ``H_ε(ρ)`` curve and reports the
H17 tiered-oracle cost *per faithful step at full consultation*. ED2 asks the sharper, budget-form
of the central hypothesis **H17**: *at an equal **oracle-dollar** budget, does a cheap or
cheapest-refutation (`escalate`) tier policy buy more faithful horizon than spending the same
dollars on bit-exact full-state truth?* This is the distributed analogue of E2/EN2/EH2 (the
when-to-consult policy comparison), with the new ``π_w`` (which-tier) axis crossed in.

The apparatus reuses ED1's: a seeded sweep on the DS0-increment-1 replicated-KV world with the
synthetic tunable-noise proposer (:class:`~verisim.distloop.model.DistNoisyModel`), so the result is
dependency-free and GPU-free. The structural difference from ED1 is the *measurement*: instead of a
single ρ=1 cost cell, ED2 sweeps ρ and plots, per tier policy, the **faithful-horizon-vs-oracle-
dollar frontier** -- the Pareto front the H17 question is really about. Reading a vertical line at
any dollar budget off the frontier answers "which policy buys the most horizon for this budget."

Because the dollar *spent* differs across policies at the same ρ (a metamorphic consult costs ¢1,
a bit-exact consult ¢16, an escalate consult a refutation-dependent sum), the policies are compared
at a matched budget by **interpolating each policy's horizon at the target dollar budget** -- a true
equal-dollar comparison, not an equal-ρ one. Two reference budgets are reported:

  - the **full-truth budget** ``B_full`` -- what always-bit-exact spends at ρ=1 (every step
    corrected); and
  - a **quarter budget** ``B/4 = 0.25·B_full`` -- the sub-linear-cost regime H18 is about.

The honest, mode-dependent finding (the same throughline as ED1, now in budget form):

  - **gross** (out-of-vocab) errors -- the cheap metamorphic tier and the `escalate` policy
    dominate the frontier: they reach the full horizon at a fraction of the bit-exact dollars, so at
    an equal budget they buy strictly more horizon. **H17 holds.**
  - **subtle** (in-flight) errors -- the cheap tiers refute nothing (the drift passes every probe
    below bit-exact), so their frontier is flat at the floor regardless of dollars; only bit-exact
    buys horizon, and `escalate` *loses* to it (it pays the cheap probes before the bit-exact it
    always needs). **H17 fails for this error class -- reported, not hidden.**

ED2 also reads the H18 **competitive ratio** off the same frontier: the faithful horizon the best
policy reaches as a fraction of the full-truth ceiling, *at the quarter budget* -- the
learning-augmented figure of merit (`DD-D4`, §2.4, §9.3). A bounded ratio well above the ρ=0 floor
at sub-linear cost is the H18 signal; the cheap-tier-useless subtle mode is its honest negative.

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
    EscalatingTierPolicy,
    FixedTierPolicy,
    budget_for_rho,
    run_dist_rollout,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci

#: The tier policies compared on the frontier: three fixed tiers (cheap → full-truth) plus the
#: cheapest-refutation `escalate` policy (DD-D1). ``bit_exact`` is the single-tier full-truth
#: baseline H17 is measured against.
POLICIES: tuple[str, ...] = ("metamorphic", "symbolic", "bit_exact", "escalate")
ERROR_MODES: tuple[str, ...] = ("gross", "subtle")


@dataclass(frozen=True)
class ED2Config:
    name: str = "ed2-dist"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    driver: str = "contention"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105)
    n_steps: int = 40
    noise: float = 0.4
    rhos: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    policies: tuple[str, ...] = POLICIES
    modes: tuple[str, ...] = ERROR_MODES
    epsilon: float = 0.0
    quarter_fraction: float = 0.25  # the sub-linear-cost budget the H18 ratio is read at

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED2Config:
        b = ED2Config()
        return ED2Config(
            name=d.get("name", b.name),
            driver=d.get("driver", b.driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            noise=d.get("noise", b.noise),
            rhos=tuple(d.get("rhos", b.rhos)),
            policies=tuple(d.get("policies", b.policies)),
            modes=tuple(d.get("modes", b.modes)),
            epsilon=d.get("epsilon", b.epsilon),
            quarter_fraction=d.get("quarter_fraction", b.quarter_fraction),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED2Config:
        return ED2Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED2Result:
    """The ED2 deliverables: the per-policy horizon-vs-dollar frontier + the H17/H18 verdict."""

    #: per (mode, policy, ρ): mean faithful horizon, mean oracle-dollars, CI on horizon.
    frontier: list[dict[str, Any]] = field(default_factory=list)
    #: per mode: the equal-budget H17 winner + the H18 competitive ratio at the quarter budget.
    verdict: list[dict[str, Any]] = field(default_factory=list)


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


def _rollout(
    oracle: DistOracle, cfg: ED2Config, mode: str, policy: str, rho: float, eval_seed: int
) -> tuple[int, int]:
    """One rollout: (faithful_horizon, oracle_dollars) for the noisy proposer at (policy, ρ)."""
    s0 = DistributedState.initial(cfg.dist)
    actions = _eval_actions(oracle, cfg.dist, cfg.driver, eval_seed, cfg.n_steps)
    model = DistNoisyModel(oracle, noise=cfg.noise, mode=mode, rng=random.Random(eval_seed + 7))
    tier_policy = EscalatingTierPolicy() if policy == "escalate" else FixedTierPolicy(policy)
    record = run_dist_rollout(
        model, oracle, s0, actions, fixed_interval_for_rho(rho), epsilon=cfg.epsilon,
        config=cfg.dist, tier_policy=tier_policy,
        budget=budget_for_rho(rho, len(actions)), seed=eval_seed,
    )
    return record.faithful_horizon, record.config["oracle_dollars"]


def _interp_horizon(points: list[dict[str, Any]], dollars: float) -> float:
    """Faithful horizon a policy reaches at ``dollars`` budget, off its Pareto envelope.

    ``points`` are one policy's frontier cells. They are reduced to the **upper envelope** -- sort
    by dollars and take the running max horizon, so the budget→horizon map is non-decreasing
    (spending more can never buy *less*: you can always decline to spend it). The horizon at
    ``dollars`` is then the linear interpolation along that envelope -- so two policies are compared
    at the *same dollars*, not the same ρ. Below the cheapest point the horizon is its floor; above
    the dearest it saturates at its ceiling.
    """
    pts = sorted(points, key=lambda p: p["dollars"])
    env: list[tuple[float, float]] = []  # (dollars, running-max horizon)
    best = float("-inf")
    for p in pts:
        best = max(best, float(p["h_eps"]))
        env.append((float(p["dollars"]), best))
    if dollars <= env[0][0]:
        return env[0][1]
    if dollars >= env[-1][0]:
        return env[-1][1]
    for (d_lo, h_lo), (d_hi, h_hi) in pairwise(env):
        if d_lo <= dollars <= d_hi:
            span = d_hi - d_lo
            if span <= 0:
                return h_hi
            return h_lo + (dollars - d_lo) / span * (h_hi - h_lo)
    return env[-1][1]  # pragma: no cover - unreachable given the bracketing above


def run_ed2(config: ED2Config | None = None, *, oracle: DistOracle | None = None) -> ED2Result:
    """Run the ED2 sweep: the horizon-vs-oracle-dollar frontier per policy + the H17/H18 verdict."""
    config = config or ED2Config()
    oracle = oracle or ReferenceDistOracle(config.dist)
    result = ED2Result()

    # the frontier: per (mode, policy), one (dollars, horizon) point per ρ, averaged over seeds.
    by_mode_policy: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for mode in config.modes:
        for policy in config.policies:
            for rho in config.rhos:
                cells = [_rollout(oracle, config, mode, policy, rho, s) for s in config.eval_seeds]
                hs = [float(h) for h, _ in cells]
                lo, hi = bootstrap_ci(hs, seed=0)
                point = {
                    "mode": mode, "policy": policy, "rho": rho,
                    "h_eps": fmean(hs), "dollars": fmean(d for _, d in cells),
                    "ci_lo": lo, "ci_hi": hi,
                }
                result.frontier.append(point)
                by_mode_policy.setdefault((mode, policy), []).append(point)

    # the verdict: compare policies at a matched dollar budget (equal-budget H17), and read the
    # H18 competitive ratio off the frontier at the sub-linear quarter budget.
    for mode in config.modes:
        be_points = by_mode_policy[(mode, "bit_exact")]
        b_full = max(p["dollars"] for p in be_points)  # ρ=1 full-truth dollars
        ceiling = max(p["h_eps"] for p in be_points)   # full-truth horizon (the ceiling)
        b_quarter = config.quarter_fraction * b_full
        at_quarter = {
            policy: _interp_horizon(by_mode_policy[(mode, policy)], b_quarter)
            for policy in config.policies
        }
        winner = max(at_quarter, key=lambda p: at_quarter[p])
        result.verdict.append({
            "mode": mode,
            "b_full": b_full,
            "b_quarter": b_quarter,
            "ceiling": ceiling,
            "horizon_at_quarter": at_quarter,
            "h17_winner": winner,                       # most horizon per equal dollar budget
            "h17_tiering_wins": winner != "bit_exact",  # a non-bit-exact policy beats full-truth?
            "competitive_ratio": (at_quarter[winner] / ceiling) if ceiling else float("nan"),
        })
    return result


CSV_HEADER = "panel,mode,policy,rho,dollars,h_eps,ci_lo,ci_hi,b_quarter,ceiling,winner,ratio"


def write_csv(result: ED2Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in result.frontier:
        rows.append(f"frontier,{p['mode']},{p['policy']},{p['rho']},{p['dollars']:.4f},"
                    f"{p['h_eps']:.4f},{p['ci_lo']:.4f},{p['ci_hi']:.4f},,,,")
    for v in result.verdict:
        rows.append(f"verdict,{v['mode']},{v['h17_winner']},,,,,,"
                    f"{v['b_quarter']:.4f},{v['ceiling']:.4f},{v['h17_winner']},"
                    f"{v['competitive_ratio']:.4f}")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED2 (equal-dollar-budget when × which-tier; H17/H18)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed2.csv")
    parser.add_argument("--plot", type=str, default="figures/ed2.png")
    args = parser.parse_args()
    config = ED2Config.from_json_file(args.config) if args.config else ED2Config()
    result = run_ed2(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    for v in result.verdict:
        hq = v["horizon_at_quarter"]
        print(f"  [{v['mode']:6s}] B_full=${v['b_full']:.0f}  B/4=${v['b_quarter']:.0f}  "
              f"ceiling H={v['ceiling']:.1f}")
        print("           horizon@B/4: " + "  ".join(f"{p}={hq[p]:.1f}" for p in hq))
        verdict = "tiering WINS" if v["h17_tiering_wins"] else "bit_exact wins"
        print(f"           H17 → {verdict} (winner={v['h17_winner']}), "
              f"H18 ratio@B/4={v['competitive_ratio']:.2f}")
    try:
        from figures.plot_ed2 import plot_ed2

        plot_ed2(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
