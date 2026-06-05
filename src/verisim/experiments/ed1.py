"""ED1 -- the distributed `H_ε(ρ)` curve + the tiered-oracle measurement (SPEC-7 §0, §12; DS6).

SPEC-7's **prime directive**: plot the faithful-horizon-vs-consultation curve once in a world where
the bit-exact oracle is *intractable*, and measure whether a **tiered** oracle (cheap consistency
checks + rare bit-exact replay) buys more faithful horizon per **oracle-dollar** than spending the
same budget on full-state truth alone (**H17**). It reports honestly if it does not.

Two committed panels, both from one seeded sweep on the DS0-increment-1 replicated-KV world, using a
**synthetic tunable-noise proposer** (:class:`~verisim.distloop.model.DistNoisyModel`) -- the
apparatus that exists before the learned `M_θ` (DS4), so the loop + tiered-oracle + oracle-dollar
machinery is exercised and the H17 tradeoff is demonstrated on a *controlled* error distribution (a
learned model later supplies a real one):

  - **left -- `H_ε(ρ)`**: free-running → fully-consulted faithful horizon at the bit-exact tier
    (directly comparable to v0/EN1/EH1) for the gross-error proposer -- the prime-directive shape;
  - **right -- the H17 tradeoff**: **oracle-dollar per faithful step** for each fixed tier
    (`metamorphic`/`symbolic`/`bit_exact`) × proposer error class (`gross`/`subtle`) at ρ=1. The
    finding: *whether cheap tiers win depends on where the model's errors fall* -- for gross
    (out-of-vocab) errors the metamorphic tier buys faithful horizon far cheaper than bit-exact; for
    subtle (in-flight) errors the cheap tiers miss the drift and you must pay for full truth.

The committed sweep runs in milliseconds on CPU (no torch); CI runs the smoke instance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
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

CURVE_TIERS = ("metamorphic", "symbolic", "bit_exact")
ERROR_MODES = ("gross", "subtle")


@dataclass(frozen=True)
class ED1Config:
    name: str = "ed1-dist"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    driver: str = "contention"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    n_steps: int = 40
    noise: float = 0.4
    rhos: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    tiers: tuple[str, ...] = CURVE_TIERS
    modes: tuple[str, ...] = ERROR_MODES
    epsilon: float = 0.0
    curve_mode: str = "gross"  # the proposer the H_ε(ρ) headline uses

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED1Config:
        b = ED1Config()
        return ED1Config(
            name=d.get("name", b.name),
            driver=d.get("driver", b.driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            noise=d.get("noise", b.noise),
            rhos=tuple(d.get("rhos", b.rhos)),
            tiers=tuple(d.get("tiers", b.tiers)),
            modes=tuple(d.get("modes", b.modes)),
            epsilon=d.get("epsilon", b.epsilon),
            curve_mode=d.get("curve_mode", b.curve_mode),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED1Config:
        return ED1Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED1Result:
    """The ED1 deliverables: the `H_ε(ρ)` curve points and the H17 (tier × mode) cost cells."""

    curve: list[dict[str, float]] = field(default_factory=list)  # per ρ: h_eps mean + CI
    h17: list[dict[str, Any]] = field(default_factory=list)  # per (mode, tier): h_eps, dollars


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
    oracle: DistOracle, cfg: ED1Config, mode: str, tier: str, rho: float, eval_seed: int
) -> tuple[int, int]:
    """One rollout: (faithful_horizon, oracle_dollars) for the noisy proposer at (tier, ρ)."""
    s0 = DistributedState.initial(cfg.dist)
    actions = _eval_actions(oracle, cfg.dist, cfg.driver, eval_seed, cfg.n_steps)
    model = DistNoisyModel(oracle, noise=cfg.noise, mode=mode, rng=random.Random(eval_seed + 7))
    record = run_dist_rollout(
        model, oracle, s0, actions, fixed_interval_for_rho(rho), epsilon=cfg.epsilon,
        config=cfg.dist, tier_policy=FixedTierPolicy(tier),
        budget=budget_for_rho(rho, len(actions)), seed=eval_seed,
    )
    return record.faithful_horizon, record.config["oracle_dollars"]


def run_ed1(config: ED1Config | None = None, *, oracle: DistOracle | None = None) -> ED1Result:
    """Run the ED1 sweep: the `H_ε(ρ)` curve (bit-exact tier) + the H17 tier×mode cost cells."""
    config = config or ED1Config()
    oracle = oracle or ReferenceDistOracle(config.dist)
    result = ED1Result()

    # the H_ε(ρ) curve, at the bit-exact tier (full truth -- the standard prime-directive shape)
    for rho in config.rhos:
        hs = [_rollout(oracle, config, config.curve_mode, "bit_exact", rho, s)[0]
              for s in config.eval_seeds]
        lo, hi = bootstrap_ci([float(h) for h in hs], seed=0)
        result.curve.append({"rho": rho, "h_eps": fmean(hs), "ci_lo": lo, "ci_hi": hi})

    # the H17 panel: oracle-dollar per faithful step for each (mode, tier) at full consultation
    for mode in config.modes:
        for tier in config.tiers:
            cells = [_rollout(oracle, config, mode, tier, 1.0, s) for s in config.eval_seeds]
            h = fmean(h for h, _ in cells)
            dollars = fmean(d for _, d in cells)
            result.h17.append({
                "mode": mode, "tier": tier, "h_eps": h, "dollars": dollars,
                "dollars_per_step": (dollars / h) if h else float("inf"),
            })
    return result


CSV_HEADER = "panel,key,x,tier,mode,h_eps,dollars,dollars_per_step,ci_lo,ci_hi"


def write_csv(result: ED1Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in result.curve:
        rows.append(f"curve,rho,{p['rho']},bit_exact,{ED1Config().curve_mode},"
                    f"{p['h_eps']:.4f},,,{p['ci_lo']:.4f},{p['ci_hi']:.4f}")
    for c in result.h17:
        rows.append(f"h17,{c['mode']}/{c['tier']},,{c['tier']},{c['mode']},"
                    f"{c['h_eps']:.4f},{c['dollars']:.4f},{c['dollars_per_step']:.4f},,")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run ED1 (distributed H_eps(rho) + H17).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed1_dist.csv")
    parser.add_argument("--plot", type=str, default="figures/ed1_dist.png")
    args = parser.parse_args()
    config = ED1Config.from_json_file(args.config) if args.config else ED1Config()
    result = run_ed1(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  H_eps(rho):", [(p["rho"], round(p["h_eps"], 1)) for p in result.curve])
    for c in result.h17:
        print(f"  {c['mode']:7s} {c['tier']:11s} H={c['h_eps']:.1f} "
              f"$={c['dollars']:.0f} $/step={c['dollars_per_step']:.1f}")
    try:
        from figures.plot_ed1 import plot_ed1

        plot_ed1(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
