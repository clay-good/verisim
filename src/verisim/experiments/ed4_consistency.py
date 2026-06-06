"""ED4 (consistency-level arm) -- weaker consistency opens the H19 gap (SPEC-7 §12, §10.2; H20/H19).

SPEC-7's H20 is "weaker consistency is harder to predict, because weaker models admit exponentially
more legal histories and the model must track which one occurred." The `CONSISTENCY_MODELS` axis
(SPEC-7 §3.4) had no consumer until now: only `eventual` was implemented. This arm adds the strong
end -- `linearizable` (synchronous all-replica writes, CP write-rejection under partition, so no
replica is ever stale and there is **no in-flight medium**) -- and sweeps the declared consistency
model to make the H20 mechanism measurable, dependency-free, on the synthetic proposer.

The clean, mechanism-driven reading connects H20 to the H19 result (ED5): the consistency-faithful
horizon outlasts the bit-faithful one *only because* a weak-consistency world has a
**consistency-invisible in-flight medium** for errors to hide in (a corrupted replication message is
bit-visible immediately but consistency-invisible until `advance` delivers it). Strong consistency
removes that medium, so the prediction is:

  - **`eventual`** (weak): a large H19 gap -- consistency-faithful horizon ≫ bit-faithful (errors
    hide in flight);
  - **`linearizable`** (strong): the gap **collapses** -- with no in-flight medium every error is
    immediately consistency-visible, so a model is consistency-faithful for no longer than it is
    bit-faithful.

So "the gap is a property of weak consistency" is the first consistency-level reading of H19 and the
H20 mechanism made concrete: strong consistency buys the model no forgiveness because there is no
hidden state to forgive. (The *absolute* free-running horizon at equal synthetic-noise is confounded
by the differing delta composition across levels -- a put is one local write + N async messages
under `eventual` but N synchronous writes under `linearizable` -- so the absolute-predictability
form of H20 is left to the learned `M_θ`; this arm reports the gap, which the synthetic proposer
measures cleanly. The reading is stated, not hidden.)

The committed sweep runs in milliseconds on CPU (no torch); CI runs the smoke instance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
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

#: strong -> weak (the §3.4 consistency curriculum ordering; weaker is hypothesised harder, H20).
CONSISTENCY_LEVELS: tuple[str, ...] = ("linearizable", "eventual")
ERROR_MODES: tuple[str, ...] = ("gross", "subtle")


@dataclass(frozen=True)
class ED4ConsistencyConfig:
    name: str = "ed4-consistency"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    driver: str = "contention"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105, 106, 107)
    n_steps: int = 40
    noise: float = 0.4
    epsilon: float = 0.0
    levels: tuple[str, ...] = CONSISTENCY_LEVELS
    modes: tuple[str, ...] = ERROR_MODES

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED4ConsistencyConfig:
        b = ED4ConsistencyConfig()
        return ED4ConsistencyConfig(
            name=d.get("name", b.name),
            driver=d.get("driver", b.driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            noise=d.get("noise", b.noise),
            epsilon=d.get("epsilon", b.epsilon),
            levels=tuple(d.get("levels", b.levels)),
            modes=tuple(d.get("modes", b.modes)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED4ConsistencyConfig:
        return ED4ConsistencyConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED4ConsistencyResult:
    """Per (level, mode): the free-running bit- and consistency-faithful horizons and their gap."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    #: per mode: the gap under the weakest vs strongest level + whether it collapses under strong.
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


def _inflight_rate(dist: DistConfig, cfg: ED4ConsistencyConfig, eval_seed: int) -> float:
    """Mean in-flight-message count per step over the ground-truth rollout (the structural medium).

    The consistency-invisible medium the H19 gap needs to exist: > 0 under ``eventual`` (async
    replication leaves messages in flight), exactly ``0`` under ``linearizable`` (synchronous writes
    enqueue nothing). This is *why* the gap appears or not, measured directly.
    """
    oracle = ReferenceDistOracle(dist)
    state = DistributedState.initial(dist)
    actions = _eval_actions(oracle, dist, cfg.driver, eval_seed, cfg.n_steps)
    counts: list[int] = []
    for action in actions:
        state = oracle.step(state, action).state
        counts.append(len(state.inflight))
    return fmean(counts) if counts else 0.0


def _both_horizons(
    dist: DistConfig, cfg: ED4ConsistencyConfig, mode: str, eval_seed: int
) -> tuple[int, int]:
    """Free-running (ρ=0) ``(bit_faithful_horizon, consistency_faithful_horizon)`` for ``dist``.

    Same idiom as ED5's ``_both_horizons``: both horizons come from one rollout -- the bit-exact
    ``divergences`` and the runner-recorded ``consistency_divergences`` -- so the H19 gap is read on
    the consistency model ``dist`` declares.
    """
    oracle = ReferenceDistOracle(dist)
    s0 = DistributedState.initial(dist)
    actions = _eval_actions(oracle, dist, cfg.driver, eval_seed, cfg.n_steps)
    # fallback=False -> an *exact* error class: the model errs only when its targeted edit exists.
    # Under linearizable the `subtle` (in-flight) class is structurally empty, so the model makes no
    # error and the H19 gap cannot arise -- the clean cross-level reading (no fallback confound).
    model = DistNoisyModel(
        oracle, noise=cfg.noise, mode=mode, rng=random.Random(eval_seed + 7), fallback=False
    )
    record = run_dist_rollout(
        model, oracle, s0, actions, fixed_interval_for_rho(0.0), epsilon=cfg.epsilon,
        config=dist, tier_policy=FixedTierPolicy("bit_exact"),
        budget=budget_for_rho(0.0, len(actions)), seed=eval_seed,
    )
    bit_h = record.faithful_horizon
    cons_h = faithful_horizon(record.config["consistency_divergences"], cfg.epsilon)
    return bit_h, cons_h


def run_ed4_consistency(
    config: ED4ConsistencyConfig | None = None,
) -> ED4ConsistencyResult:
    """Sweep the declared consistency model; measure the free-running H19 gap per level (H20)."""
    config = config or ED4ConsistencyConfig()
    result = ED4ConsistencyResult()

    by_mode_level: dict[tuple[str, str], dict[str, Any]] = {}
    for level in config.levels:
        dist = replace(config.dist, consistency_model=level)
        inflight_rate = fmean(_inflight_rate(dist, config, s) for s in config.eval_seeds)
        for mode in config.modes:
            cells = [_both_horizons(dist, config, mode, s) for s in config.eval_seeds]
            bit = [float(b) for b, _ in cells]
            cons = [float(c) for _, c in cells]
            gap = [c - b for b, c in zip(bit, cons, strict=True)]
            gap_lo, gap_hi = bootstrap_ci(gap, seed=0)
            row = {
                "level": level, "mode": mode, "inflight_rate": inflight_rate,
                "bit_h": fmean(bit), "cons_h": fmean(cons),
                "gap": fmean(gap), "gap_lo": gap_lo, "gap_hi": gap_hi,
                "consistency_outlasts": gap_lo > 0.0,
            }
            result.rows.append(row)
            by_mode_level[(mode, level)] = row

    # the verdict: does the H19 gap collapse as consistency strengthens? (strong = levels[0])
    strong, weak = config.levels[0], config.levels[-1]
    for mode in config.modes:
        gap_weak = by_mode_level[(mode, weak)]["gap"]
        gap_strong = by_mode_level[(mode, strong)]["gap"]
        result.verdict.append({
            "mode": mode,
            "strong_level": strong, "weak_level": weak,
            "gap_strong": gap_strong, "gap_weak": gap_weak,
            "inflight_strong": by_mode_level[(mode, strong)]["inflight_rate"],
            "inflight_weak": by_mode_level[(mode, weak)]["inflight_rate"],
            # H20/H19: weak consistency opens the gap; strong consistency collapses it.
            "gap_collapses_under_strong": gap_strong < gap_weak,
        })
    return result


CSV_HEADER = "panel,level,mode,inflight_rate,bit_h,cons_h,gap,gap_lo,gap_hi"


def write_csv(result: ED4ConsistencyResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in result.rows:
        rows.append(f"row,{r['level']},{r['mode']},{r['inflight_rate']:.4f},{r['bit_h']:.4f},"
                    f"{r['cons_h']:.4f},{r['gap']:.4f},{r['gap_lo']:.4f},{r['gap_hi']:.4f}")
    for v in result.verdict:
        rows.append(f"verdict,{v['strong_level']}->{v['weak_level']},{v['mode']},"
                    f"{v['inflight_weak']:.4f},,,{v['gap_strong']:.4f},{v['gap_weak']:.4f},")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED4 consistency-level arm (H20/H19: weaker consistency opens the gap)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed4_consistency.csv")
    parser.add_argument("--plot", type=str, default="figures/ed4_consistency.png")
    args = parser.parse_args()
    config = (
        ED4ConsistencyConfig.from_json_file(args.config) if args.config else ED4ConsistencyConfig()
    )
    result = run_ed4_consistency(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    for r in result.rows:
        print(f"  [{r['level']:13s} {r['mode']:6s}] bit H={r['bit_h']:.1f} cons H={r['cons_h']:.1f}"
              f" gap={r['gap']:.1f} [{r['gap_lo']:.1f},{r['gap_hi']:.1f}]")
    for v in result.verdict:
        verdict = "COLLAPSES under strong" if v["gap_collapses_under_strong"] else "no collapse"
        print(f"  H20/H19 [{v['mode']:6s}]: gap {v['weak_level']}={v['gap_weak']:.1f} → "
              f"{v['strong_level']}={v['gap_strong']:.1f}  ({verdict})")
    try:
        from figures.plot_ed4_consistency import plot_ed4_consistency

        plot_ed4_consistency(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
