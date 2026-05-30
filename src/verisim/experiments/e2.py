"""Experiment E2 -- consultation-policy comparison (SPEC-2 §9, H2, milestone M7).

Fixes the budget ``ρ`` at the interesting knee from E1 and compares the §6.1
consultation policies -- ``fixed`` vs. ``uncertainty_triggered`` vs.
``drift_triggered`` -- at *equal* ``ρ``. The runner's spend-down backstop makes
every arm spend exactly ``floor(ρ·T)`` oracle calls, so the comparison isolates
*where* a policy spends its budget, which is precisely H2 (does spending the budget
on the steps the model is least sure about extend the faithful horizon?).

The triggered policies need a per-step uncertainty signal; the neural ``M_θ``
(M4) supplies it (mean decode entropy, SPEC-2 §7.2). Their thresholds ``τ`` are
calibrated *per rollout* from the model's own uncertainties along the unaided
(ρ=0) rollout, so each policy's natural trigger rate matches the budget:

  - ``uncertainty``: ``τ`` = the ``(1-ρ)`` quantile of the per-step signals, so
    about a ``ρ`` fraction of steps exceed it;
  - ``drift``: ``τ`` = (total signal) / (budget), so the accumulator crosses the
    threshold about ``budget`` times over the rollout.

Like E1, ``ε`` does not affect loop dynamics, so each rollout runs once and spins
out one :class:`RunRecord` per ``ε``. The figure (`figures/plot_e2.py`) and CSV are
produced from these records only (SPEC-2 §7.3, §12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.delta.apply import apply
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop.model import UncertaintyModel
from verisim.loop.policy import (
    ConsultationPolicy,
    DriftTriggered,
    UncertaintyTriggered,
    fixed_interval_for_rho,
)
from verisim.loop.runner import budget_for_rho, run_rollout
from verisim.metrics.record import RunRecord, write_records
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

from .e1 import E1Config, eval_actions, train_model


@dataclass(frozen=True)
class E2Config:
    name: str = "e2-small"
    base: E1Config = field(default_factory=E1Config)
    rho: float = 0.2  # the E1 knee the policies are compared at
    policies: tuple[str, ...] = ("fixed", "uncertainty", "drift")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> E2Config:
        base = E2Config()
        return E2Config(
            name=d.get("name", base.name),
            base=E1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", base.rho),
            policies=tuple(d.get("policies", base.policies)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> E2Config:
        return E2Config.from_dict(json.loads(Path(path).read_text()))


def _quantile(sorted_values: list[float], q: float) -> float:
    """The ``q``-quantile (0..1) of an already-sorted, non-empty list (nearest-rank)."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def unaided_signals(
    model: UncertaintyModel, s0: State, actions: list[Action]
) -> list[float]:
    """The model's per-step uncertainty along the unaided (ρ=0) rollout."""
    state = s0
    signals: list[float] = []
    for action in actions:
        delta, signal = model.predict_delta_with_uncertainty(state, action)
        signals.append(signal)
        state = apply(state, delta)
    return signals


def build_policy(name: str, rho: float, signals: list[float]) -> ConsultationPolicy:
    """Construct one §6.1 policy, calibrating ``τ`` to the budget (see module doc)."""
    budget = budget_for_rho(rho, len(signals))
    if name == "fixed":
        return fixed_interval_for_rho(rho)
    if name == "uncertainty":
        return UncertaintyTriggered(tau=_quantile(sorted(signals), 1.0 - rho))
    if name == "drift":
        total = sum(signals)
        return DriftTriggered(tau=total / budget if budget else float("inf"))
    raise ValueError(f"unknown policy {name!r}")


def run_e2(config: E2Config | None = None, *, oracle: Oracle | None = None) -> list[RunRecord]:
    """Train the model and compare the policies at the fixed knee ``ρ``."""
    config = config or E2Config()
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
            budget = budget_for_rho(config.rho, len(actions))
            for policy_name in config.policies:
                policy = build_policy(policy_name, config.rho, signals)
                rollout = run_rollout(
                    world_model,
                    oracle,
                    State.empty(),
                    actions,
                    policy,
                    epsilon=base.epsilons[0],
                    budget=budget,
                    seed=seed,
                )
                for epsilon in base.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "policy": policy_name,
                                "difficulty": difficulty,
                                "driver": driver,
                                "rho": config.rho,
                                "n_steps": len(actions),
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

    parser = argparse.ArgumentParser(description="Run experiment E2 (policy comparison).")
    parser.add_argument("--config", type=str, default=None, help="path to an E2 config JSON")
    parser.add_argument("--out", type=str, default="runs/e2/records.jsonl")
    args = parser.parse_args()
    config = E2Config.from_json_file(args.config) if args.config else E2Config()
    records = run_e2(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
