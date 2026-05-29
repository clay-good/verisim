"""Baseline sweep over the propose-verify-correct loop (SPEC-2 §8, §9).

A minimal, runnable demonstration that the M5 loop works end to end: it rolls the
baseline models through the loop across a budget sweep and emits one
:class:`RunRecord` per cell. This is *not* experiment E1 (M6) -- E1's headline
``H_ε(ρ)`` curve needs the learned model ``M_θ`` (M4), which drifts in the
interesting interior. With baselines the picture is degenerate by design:

  - ``OracleBackedModel`` (b2) is perfect, so ``H_ε = T`` at every ``ρ`` including
    ``ρ = 0`` -- it never needs the oracle;
  - ``NullModel`` (b3) predicts no change, so it drifts as soon as ground truth
    changes anything, and only oracle consultations buy faithful steps back.

These two bracket what the neural model must live between, and exercising them
here is what makes the M5 invariants concrete.
"""

from __future__ import annotations

from collections.abc import Sequence

from verisim.data.generate import generate_trajectory
from verisim.env.action import Action, parse_action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.loop.model import Model, NullModel, OracleBackedModel
from verisim.loop.policy import fixed_interval_for_rho
from verisim.loop.runner import budget_for_rho, run_rollout
from verisim.metrics.record import RunRecord
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

DEFAULT_RHOS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def sweep_model(
    model: Model,
    model_name: str,
    oracle: Oracle,
    actions: Sequence[Action],
    *,
    epsilon: float,
    rhos: Sequence[float] = DEFAULT_RHOS,
    seed: int = 0,
) -> list[RunRecord]:
    """Run one model through the budget sweep, returning a record per ``ρ``."""
    records: list[RunRecord] = []
    n = len(actions)
    for rho in rhos:
        records.append(
            run_rollout(
                model,
                oracle,
                State.empty(),
                actions,
                fixed_interval_for_rho(rho),
                epsilon=epsilon,
                budget=budget_for_rho(rho, n),
                seed=seed,
                config={"model": model_name, "rho": rho, "n_steps": n},
            )
        )
    return records


def run_baseline_sweep(
    config: EnvConfig = DEFAULT_CONFIG,
    *,
    driver: str = "weighted",
    seed: int = 0,
    n_steps: int = 40,
    epsilon: float = 0.0,
    rhos: Sequence[float] = DEFAULT_RHOS,
) -> list[RunRecord]:
    """Generate one trajectory and sweep both baseline models over ``rhos``."""
    oracle = ReferenceOracle()
    trajectory = generate_trajectory(oracle, config, driver, seed, n_steps)
    actions = [parse_action(step["action"]) for step in trajectory.steps]
    records: list[RunRecord] = []
    records += sweep_model(
        NullModel(), "null", oracle, actions, epsilon=epsilon, rhos=rhos, seed=seed
    )
    records += sweep_model(
        OracleBackedModel(oracle), "oracle", oracle, actions, epsilon=epsilon, rhos=rhos, seed=seed
    )
    return records


def main() -> None:  # pragma: no cover - manual demonstration entry point
    for record in run_baseline_sweep():
        cfg = record.config
        print(
            f"model={cfg['model']:>6} rho={cfg['rho']:<4} "
            f"oracle_calls={record.oracle_calls:>2} H_eps={record.faithful_horizon}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
