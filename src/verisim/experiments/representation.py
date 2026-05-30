"""E4 representation axis -- delta vs. full-state prediction target (SPEC-2 §9, §10).

The §10 representation ablation, the last open E4 axis. It trains two world models
from the *same* config and data, differing only in the prediction target:

  - **delta** -- the primary ``M_θ`` (:class:`NeuralWorldModel`): predict the
    structured edits the action makes (constrained to the delta grammar);
  - **full_state** -- the alternative (:class:`FullStateWorldModel`): regenerate the
    whole next state (constrained to the state grammar).

Both arms are scored on the *same* two clean (ρ=0) metrics as the other E4 axes --
per-step teacher-forced accuracy and clean horizon -- so the comparison isolates the
effect of the prediction target. SPEC.md §6.1 predicts delta should win (it bounds
the hallucination surface and localizes verification); this experiment is what turns
that prediction into a measured figure.

The committed config is a small, fast instance of the machinery (SPEC-2 §17.5), not
a tuned publication run; the honest number is the deliverable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop.model import Model
from verisim.loop.policy import Never
from verisim.loop.runner import run_rollout
from verisim.metrics.record import RunRecord, write_records
from verisim.model.full_state import FullStateWorldModel
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

from .e1 import E1Config, eval_actions, train_model
from .e4 import teacher_forced_accuracy


@dataclass(frozen=True)
class RepresentationConfig:
    name: str = "e4-representation"
    base: E1Config = field(default_factory=E1Config)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RepresentationConfig:
        base = RepresentationConfig()
        return RepresentationConfig(
            name=d.get("name", base.name),
            base=E1Config.from_dict(d.get("base", {})),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> RepresentationConfig:
        return RepresentationConfig.from_dict(json.loads(Path(path).read_text()))


def _measure_arm(
    representation: str,
    model: Model,
    oracle: Oracle,
    config: RepresentationConfig,
) -> list[RunRecord]:
    """Clean (ρ=0) per-step accuracy + clean horizon for one representation arm."""
    base = config.base
    env = DEFAULT_CONFIG
    epsilon = base.epsilons[0]

    records: list[RunRecord] = []
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
                        "representation": representation,
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


def run_representation(
    config: RepresentationConfig | None = None, *, oracle: Oracle | None = None
) -> list[RunRecord]:
    """Train a delta model and a full-state model on identical data; measure both."""
    config = config or RepresentationConfig()
    base = config.base
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)

    delta_model = NeuralWorldModel(train_model(base, vocab, oracle, env, target="delta"), vocab)
    state_model = FullStateWorldModel(
        train_model(base, vocab, oracle, env, target="state"), vocab
    )

    records: list[RunRecord] = []
    records += _measure_arm("delta", delta_model, oracle, config)
    records += _measure_arm("full_state", state_model, oracle, config)
    return records


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the E4 representation axis (delta vs full-state)."
    )
    parser.add_argument("--config", type=str, default=None, help="path to a representation config")
    parser.add_argument("--out", type=str, default="runs/representation/records.jsonl")
    args = parser.parse_args()
    config = (
        RepresentationConfig.from_json_file(args.config) if args.config else RepresentationConfig()
    )
    records = run_representation(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
