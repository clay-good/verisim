"""Experiment E3 -- correction-operator comparison (SPEC-2 §9, H3, milestone M7).

Fixes the consultation policy (E2's winner) and budget ``ρ`` and compares the §6.2
correction operators -- ``hard_reset`` vs. ``residual`` vs. ``projection`` -- at
equal ``ρ``. Output: faithful horizon and the per-operator diagnostic by operator.

Honest v0 expectation (see :mod:`verisim.loop.operator`): with the v0 oracle
returning the *full* one-step truth, all three operators snap the coupled state to
the same ``s'``, so E3's headline ``H_ε`` is **identical across operators** (a
theoretical identity, reported with CIs). What differs is the diagnostic each
operator exposes, recorded here for the deferred H3 work:

  - ``residual`` -- mean discrepancy ``|s' △ ŝ'|`` per correction (the Stage-2
    online-learning signal magnitude);
  - ``projection`` -- mean repaired fraction per correction (the per-correction
    cost; lower means the model was nearly right).

The figure (`figures/plot_e3.py`) and CSV are produced from these records only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop.operator import CorrectionOperator, HardReset, Projection, Residual
from verisim.loop.runner import budget_for_rho, run_rollout
from verisim.metrics.record import RunRecord, write_records
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

from .e1 import E1Config, eval_actions, train_model
from .e2 import build_policy, unaided_signals


@dataclass(frozen=True)
class E3Config:
    name: str = "e3-small"
    base: E1Config = field(default_factory=E1Config)
    rho: float = 0.2
    policy: str = "fixed"  # E2's winner; the policy held fixed across operators
    operators: tuple[str, ...] = ("hard_reset", "residual", "projection")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> E3Config:
        base = E3Config()
        return E3Config(
            name=d.get("name", base.name),
            base=E1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", base.rho),
            policy=d.get("policy", base.policy),
            operators=tuple(d.get("operators", base.operators)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> E3Config:
        return E3Config.from_dict(json.loads(Path(path).read_text()))


def _operator_diagnostic(name: str, operator: CorrectionOperator) -> float | None:
    """The mean per-correction diagnostic an operator exposed (``None`` for none)."""
    if isinstance(operator, Residual):
        return fmean(operator.discrepancies) if operator.discrepancies else 0.0
    if isinstance(operator, Projection):
        return fmean(operator.repaired_fractions) if operator.repaired_fractions else 0.0
    return None


def build_operator(name: str) -> CorrectionOperator:
    if name == "hard_reset":
        return HardReset()
    if name == "residual":
        return Residual()
    if name == "projection":
        return Projection()
    raise ValueError(f"unknown operator {name!r}")


def run_e3(config: E3Config | None = None, *, oracle: Oracle | None = None) -> list[RunRecord]:
    """Train the model and compare the operators at the fixed policy + knee ``ρ``."""
    config = config or E3Config()
    base = config.base
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    model = train_model(base, vocab, oracle, env)
    world_model = NeuralWorldModel(model, vocab)

    records: list[RunRecord] = []
    for difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, env, driver, seed, base.eval_steps)
            signals = unaided_signals(world_model, State.empty(), actions)
            policy = build_policy(config.policy, config.rho, signals)
            budget = budget_for_rho(config.rho, len(actions))
            for operator_name in config.operators:
                operator = build_operator(operator_name)
                rollout = run_rollout(
                    world_model,
                    oracle,
                    State.empty(),
                    actions,
                    policy,
                    epsilon=base.epsilons[0],
                    operator=operator,
                    budget=budget,
                    seed=seed,
                )
                diagnostic = _operator_diagnostic(operator_name, operator)
                for epsilon in base.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "operator": operator_name,
                                "policy": config.policy,
                                "difficulty": difficulty,
                                "driver": driver,
                                "rho": config.rho,
                                "n_steps": len(actions),
                                "diagnostic": diagnostic,
                            },
                            seed=seed,
                            epsilon=epsilon,
                            divergences=list(rollout.divergences),
                            consultation_schedule=list(rollout.consultation_schedule),
                        )
                    )
    return records


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment E3 (operator comparison).")
    parser.add_argument("--config", type=str, default=None, help="path to an E3 config JSON")
    parser.add_argument("--out", type=str, default="runs/e3/records.jsonl")
    args = parser.parse_args()
    config = E3Config.from_json_file(args.config) if args.config else E3Config()
    records = run_e3(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
