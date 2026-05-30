"""E4 objective axis -- supervised vs. supervised+RLVR (SPEC-2 §9, §17.4).

E4 (``e4.py``) sweeps the *size* and *difficulty* ablation axes; this module runs
the third §9 axis: the **training objective**. It trains one Stage-1 supervised
model, deep-copies it, continues the copy with Stage-2 RLVR (`train_rlvr`, the
oracle faithful-horizon reward, SPEC-2 §5.3), and measures the *same* two clean
(ρ=0) metrics as E4 -- per-step teacher-forced accuracy and clean horizon -- for
each arm. Both arms branch from the identical Stage-1 init, so the comparison
isolates the effect of training against the verifiable oracle reward.

The committed config is a small, fast instance of the machinery, not a tuned
publication run (SPEC-2 §17.5): at the H1-floor scale the RLVR reward is sparse
(episodes terminate at the first unfaithful step), so a large lift is not expected
here -- the honest number, like the rest of v0, is the deliverable.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop.policy import Never
from verisim.loop.runner import run_rollout
from verisim.metrics.record import RunRecord, write_records
from verisim.model.transformer import GPT
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle
from verisim.train.rlvr import train_rlvr

from .e1 import E1Config, eval_actions, train_model
from .e4 import teacher_forced_accuracy


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str = "e4-objective"
    base: E1Config = field(default_factory=E1Config)
    # Stage-2 RLVR hyperparameters (applied to a copy of the supervised model).
    rlvr_steps: int = 80
    rlvr_samples_per_env: int = 4
    rlvr_lr: float = 1e-3
    rlvr_seeds: tuple[int, ...] = (0, 1, 2, 3)
    rlvr_n_steps: int = 24
    rlvr_max_edits: int = 24
    rlvr_max_run: int = 24
    rlvr_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ObjectiveConfig:
        base = ObjectiveConfig()
        return ObjectiveConfig(
            name=d.get("name", base.name),
            base=E1Config.from_dict(d.get("base", {})),
            rlvr_steps=d.get("rlvr_steps", base.rlvr_steps),
            rlvr_samples_per_env=d.get("rlvr_samples_per_env", base.rlvr_samples_per_env),
            rlvr_lr=d.get("rlvr_lr", base.rlvr_lr),
            rlvr_seeds=tuple(d.get("rlvr_seeds", base.rlvr_seeds)),
            rlvr_n_steps=d.get("rlvr_n_steps", base.rlvr_n_steps),
            rlvr_max_edits=d.get("rlvr_max_edits", base.rlvr_max_edits),
            rlvr_max_run=d.get("rlvr_max_run", base.rlvr_max_run),
            rlvr_seed=d.get("rlvr_seed", base.rlvr_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ObjectiveConfig:
        return ObjectiveConfig.from_dict(json.loads(Path(path).read_text()))


def _measure_arm(
    objective: str,
    gpt: GPT,
    vocab: Vocab,
    oracle: Oracle,
    config: ObjectiveConfig,
) -> list[RunRecord]:
    """Clean (ρ=0) per-step accuracy + clean horizon for one training arm."""
    base = config.base
    env = DEFAULT_CONFIG
    epsilon = base.epsilons[0]
    model = NeuralWorldModel(gpt, vocab)

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
                        "objective": objective,
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


def run_objective(
    config: ObjectiveConfig | None = None, *, oracle: Oracle | None = None
) -> list[RunRecord]:
    """Train supervised, branch into +RLVR, and measure clean faithfulness per arm."""
    config = config or ObjectiveConfig()
    base = config.base
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)

    supervised = train_model(base, vocab, oracle, env)
    rlvr = copy.deepcopy(supervised)
    train_rlvr(
        rlvr,
        vocab,
        oracle=oracle,
        config=env,
        driver=base.train_driver,
        seeds=config.rlvr_seeds,
        n_steps=config.rlvr_n_steps,
        epsilon=base.epsilons[0],
        steps=config.rlvr_steps,
        samples_per_env=config.rlvr_samples_per_env,
        lr=config.rlvr_lr,
        max_edits=config.rlvr_max_edits,
        max_run=config.rlvr_max_run,
        seed=config.rlvr_seed,
    )

    records: list[RunRecord] = []
    records += _measure_arm("supervised", supervised, vocab, oracle, config)
    records += _measure_arm("rlvr", rlvr, vocab, oracle, config)
    return records


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run the E4 objective axis (supervised vs +RLVR).")
    parser.add_argument("--config", type=str, default=None, help="path to an objective config")
    parser.add_argument("--out", type=str, default="runs/objective/records.jsonl")
    args = parser.parse_args()
    config = ObjectiveConfig.from_json_file(args.config) if args.config else ObjectiveConfig()
    records = run_objective(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
