"""Experiment E4 -- ablations (SPEC-2 §9, milestone open work).

E1 found no favorable H1 knee: the model drifts immediately at ρ=0, so the interior
sits near the floor. The standing question (SPEC-2 §17.5) is *why* -- is the model
too small (capacity) or the task mis-tuned (difficulty)? E4 sweeps the two buildable
ablation axes of §9 -- **model size** and **difficulty/driver** -- and measures clean
(ρ=0) faithfulness, so the capacity-vs-difficulty question becomes a figure.

Two metrics per cell, both at ρ=0 (unaided):

  - **per-step teacher-forced accuracy** -- fraction of steps whose predicted delta
    exactly reproduces the oracle's truth, measured from the *true* current state so
    it is not dominated by step-0 compounding (the §7.2 "per-step exact-match
    accuracy" diagnostic). The clean capacity signal.
  - **clean faithful horizon** ``H_ε`` -- how far the unaided autoregressive rollout
    stays faithful. The headline-relevant measure.

The objective axis (supervised vs. +RLVR) is run separately in ``objective.py``,
which branches a Stage-2 RLVR copy off the Stage-1 model and measures the same clean
metrics; the representation axis (delta vs. full-state) still needs a full-state head
and is left for later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from verisim.delta.apply import apply
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop.model import Model
from verisim.loop.policy import Never
from verisim.loop.runner import run_rollout
from verisim.metrics.divergence import divergence
from verisim.metrics.record import RunRecord, write_records
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

from .e1 import E1Config, eval_actions, train_model


@dataclass(frozen=True)
class E4Size:
    label: str
    n_layer: int
    n_embd: int


@dataclass(frozen=True)
class E4Config:
    name: str = "e4-ablation"
    base: E1Config = field(default_factory=E1Config)
    sizes: tuple[E4Size, ...] = (
        E4Size("tiny", 1, 32),
        E4Size("small", 2, 64),
        E4Size("medium", 4, 128),
    )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> E4Config:
        base = E4Config()
        sizes = d.get("sizes")
        return E4Config(
            name=d.get("name", base.name),
            base=E1Config.from_dict(d.get("base", {})),
            sizes=tuple(E4Size(**s) for s in sizes) if sizes else base.sizes,
        )

    @staticmethod
    def from_json_file(path: str | Path) -> E4Config:
        return E4Config.from_dict(json.loads(Path(path).read_text()))


def teacher_forced_accuracy(
    model: Model, oracle: Oracle, s0: State, actions: list[Action]
) -> float:
    """Fraction of steps whose predicted delta exactly matches the oracle (ρ=0, per-step)."""
    if not actions:
        return 1.0
    state = s0
    correct = 0
    for action in actions:
        truth = oracle.step(state, action).state
        if divergence(apply(state, model.predict_delta(state, action)), truth) == 0.0:
            correct += 1
        state = truth  # teacher-forced: per-step, uncompounded
    return correct / len(actions)


def run_e4(config: E4Config | None = None, *, oracle: Oracle | None = None) -> list[RunRecord]:
    """Train one model per size and measure clean faithfulness across difficulties."""
    config = config or E4Config()
    base = config.base
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    epsilon = base.epsilons[0]

    records: list[RunRecord] = []
    for size in config.sizes:
        sized = replace(base, n_layer=size.n_layer, n_embd=size.n_embd)
        model = NeuralWorldModel(train_model(sized, vocab, oracle, env), vocab)
        for difficulty, driver in base.difficulties.items():
            for seed in base.eval_seeds:
                actions = eval_actions(oracle, env, driver, seed, base.eval_steps)
                accuracy = teacher_forced_accuracy(model, oracle, State.empty(), actions)
                rollout = run_rollout(
                    model, oracle, State.empty(), actions, Never(), epsilon=epsilon, budget=0
                )
                records.append(
                    RunRecord(
                        config={
                            "experiment": config.name,
                            "size": size.label,
                            "n_layer": size.n_layer,
                            "n_embd": size.n_embd,
                            "difficulty": difficulty,
                            "driver": driver,
                            "rho": 0.0,
                            "n_steps": len(actions),
                            "step_accuracy": accuracy,
                            "clean_horizon": rollout.faithful_horizon,
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

    parser = argparse.ArgumentParser(description="Run experiment E4 (size/difficulty ablation).")
    parser.add_argument("--config", type=str, default=None, help="path to an E4 config")
    parser.add_argument("--out", type=str, default="runs/e4/records.jsonl")
    args = parser.parse_args()
    config = E4Config.from_json_file(args.config) if args.config else E4Config()
    records = run_e4(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
