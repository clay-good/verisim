"""ED3 -- correction / belief operators (SPEC-7 §8.3, §10 ED3; DS7; role of E3/EN3/EH3).

Once a consultation refutes the model's prediction, *how* should the coupled state be corrected? v0
ships three operators -- ``hard_reset`` (snap to truth), ``residual`` (snap + record discrepancy),
``projection`` (snap, record the repaired fraction) -- and proves an honest theoretical *identity*:
because a v0 consult returns the full one-step truth, all three snap to the same ``s'`` and are
behaviorally identical on faithful horizon (SPEC-2 §6.2). ED3 asks the sharper distributed question
(SPEC-7 §8.3): **does the distributed world break that identity?** It does -- because the cluster
state has a part a *partial* correction can decline to fix: the **in-flight replication messages**,
the source of stale reads under partition and exactly the ``subtle`` error class the cheap tiers
miss (§5). The :class:`~verisim.distloop.operator.ReplicasOnlyCorrection` operator snaps the durable
replicas to truth but **trusts the model's predicted in-flight** -- so when the model corrupts an
in-flight message, that partial correction leaves the error in place and the coupled state keeps
drifting.

The apparatus reuses ED2's: a seeded sweep on the DS0-increment-1 replicated-KV world with the
synthetic tunable-noise proposer (:class:`~verisim.distloop.model.DistNoisyModel`), dependency-free
and GPU-free, whose corruption ``mode`` targets a specific part of the state:

  - **gross** -- corrupts a **replica write** (an out-of-vocab value). Both ``hard_reset`` and
    ``replicas_only`` snap the replicas, so they recover the *same* horizon -- the identity holds.
  - **subtle** -- corrupts an **in-flight message's** payload. ``hard_reset`` fixes the in-flight,
    recovers full horizon; ``replicas_only`` trusts the corrupted in-flight and recovers **less** --
    the v0 identity **breaks**, exactly under the partition/weak-consistency regime §8.3 predicts.

So ED3's result is mode-dependent and structural: the full-correction operators (``hard_reset`` /
``residual`` / ``projection``) are identical on ``H_ε`` (the v0 identity, reported with CIs), and
the partial ``replicas_only`` operator costs horizon **only** for the in-flight (``subtle``) class
-- the distributed world's hidden state a partial correction cannot see. The residual / projection
diagnostics (bits-to-correct, repaired fraction) quantify how much truth each correction injects.

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
    HardReset,
    Projection,
    ReplicasOnlyCorrection,
    Residual,
    budget_for_rho,
    run_dist_rollout,
)
from verisim.distloop.operator import CorrectionOperator
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci

#: The §8.3 correction operators compared. The first three are full-correction (snap to truth, the
#: identity); ``replicas_only`` is the distributed partial operator that breaks it.
OPERATORS: tuple[str, ...] = ("hard_reset", "residual", "projection", "replicas_only")
ERROR_MODES: tuple[str, ...] = ("gross", "subtle")


def _make_operator(name: str) -> CorrectionOperator:
    if name == "hard_reset":
        return HardReset()
    if name == "residual":
        return Residual()
    if name == "projection":
        return Projection()
    if name == "replicas_only":
        return ReplicasOnlyCorrection()
    raise ValueError(f"unknown operator {name!r}; choose from {OPERATORS}")


@dataclass(frozen=True)
class ED3Config:
    name: str = "ed3-dist"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    driver: str = "contention"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105)
    n_steps: int = 40
    noise: float = 0.4
    rho: float = 0.5  # the fixed interior budget the operators are compared at
    operators: tuple[str, ...] = OPERATORS
    modes: tuple[str, ...] = ERROR_MODES
    epsilon: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED3Config:
        b = ED3Config()
        return ED3Config(
            name=d.get("name", b.name),
            driver=d.get("driver", b.driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            noise=d.get("noise", b.noise),
            rho=d.get("rho", b.rho),
            operators=tuple(d.get("operators", b.operators)),
            modes=tuple(d.get("modes", b.modes)),
            epsilon=d.get("epsilon", b.epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED3Config:
        return ED3Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED3Result:
    """The ED3 deliverables: per (mode, operator) horizon + the identity-break verdict per mode."""

    #: per (mode, operator): mean faithful horizon, CI, mean repaired fraction (partial ops).
    cells: list[dict[str, Any]] = field(default_factory=list)
    #: per mode: the full-correction horizon, the partial (`replicas_only`) horizon, identity-break.
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
    oracle: DistOracle, cfg: ED3Config, mode: str, operator: str, eval_seed: int
) -> tuple[int, float]:
    """One rollout: (faithful_horizon, mean repaired fraction) for the operator at interior ρ."""
    actions = _eval_actions(oracle, cfg.dist, cfg.driver, eval_seed, cfg.n_steps)
    model = DistNoisyModel(oracle, noise=cfg.noise, mode=mode, rng=random.Random(eval_seed + 7))
    op = _make_operator(operator)
    record = run_dist_rollout(
        model, oracle, DistributedState.initial(cfg.dist), actions,
        fixed_interval_for_rho(cfg.rho), epsilon=cfg.epsilon, config=cfg.dist,
        tier_policy=FixedTierPolicy("bit_exact"), operator=op,
        budget=budget_for_rho(cfg.rho, len(actions)), seed=eval_seed,
    )
    fractions = getattr(op, "repaired_fractions", None) or getattr(op, "discrepancies", None) or []
    mean_frac = fmean(float(x) for x in fractions) if fractions else 0.0
    return record.faithful_horizon, mean_frac


def run_ed3(config: ED3Config | None = None, *, oracle: DistOracle | None = None) -> ED3Result:
    """Run the ED3 operator comparison at the interior ρ; the per-mode identity-break verdict."""
    config = config or ED3Config()
    oracle = oracle or ReferenceDistOracle(config.dist)
    result = ED3Result()

    by_mode_op: dict[tuple[str, str], float] = {}
    for mode in config.modes:
        for operator in config.operators:
            cells = [_rollout(oracle, config, mode, operator, s) for s in config.eval_seeds]
            hs = [float(h) for h, _ in cells]
            lo, hi = bootstrap_ci(hs, seed=0)
            mean_h = fmean(hs)
            by_mode_op[(mode, operator)] = mean_h
            result.cells.append({
                "mode": mode, "operator": operator, "h_eps": mean_h, "ci_lo": lo, "ci_hi": hi,
                "repaired_fraction": fmean(f for _, f in cells),
            })

    # the verdict: per mode, does the partial `replicas_only` recover as much as full `hard_reset`?
    for mode in config.modes:
        full_h = by_mode_op[(mode, "hard_reset")]
        partial_h = by_mode_op[(mode, "replicas_only")]
        # the full-correction operators should all coincide (the v0 identity); report the spread.
        full_ops = [o for o in config.operators if o != "replicas_only"]
        full_spread = max(by_mode_op[(mode, o)] for o in full_ops) - min(
            by_mode_op[(mode, o)] for o in full_ops
        )
        result.verdict.append({
            "mode": mode,
            "full_h": full_h,
            "partial_h": partial_h,
            "full_correction_spread": full_spread,    # ~0 ⇒ the v0 identity holds across full ops
            "identity_holds": full_spread < 1e-9,
            "partial_costs_horizon": partial_h < full_h - 1e-9,  # the identity-break signal
            "horizon_gap": full_h - partial_h,
        })
    return result


CSV_HEADER = "panel,mode,operator,h_eps,ci_lo,ci_hi,repaired_fraction,full_h,partial_h,gap,break"


def write_csv(result: ED3Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in result.cells:
        rows.append(f"cell,{c['mode']},{c['operator']},{c['h_eps']:.4f},"
                    f"{c['ci_lo']:.4f},{c['ci_hi']:.4f},{c['repaired_fraction']:.4f},,,,")
    for v in result.verdict:
        rows.append(f"verdict,{v['mode']},replicas_only,,,,,"
                    f"{v['full_h']:.4f},{v['partial_h']:.4f},{v['horizon_gap']:.4f},"
                    f"{int(v['partial_costs_horizon'])}")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run ED3 (distributed correction operators).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed3.csv")
    parser.add_argument("--plot", type=str, default="figures/ed3.png")
    args = parser.parse_args()
    config = ED3Config.from_json_file(args.config) if args.config else ED3Config()
    result = run_ed3(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    for v in result.verdict:
        identity = "identity holds" if v["identity_holds"] else "identity BROKEN(!)"
        brk = "partial COSTS horizon" if v["partial_costs_horizon"] else "partial recovers full"
        print(f"  [{v['mode']:6s}] full H={v['full_h']:.1f}  "
              f"partial(replicas_only) H={v['partial_h']:.1f}  "
              f"gap={v['horizon_gap']:.1f}  → {brk}; full-op {identity}")
    try:
        from figures.plot_ed3 import plot_ed3

        plot_ed3(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
