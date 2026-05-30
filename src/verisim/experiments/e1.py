"""Experiment E1 -- the headline ``H_ε(ρ)`` curve (SPEC-2 §9, milestone M6).

Trains one neural world model (Stage 1), then sweeps consultation budget ``ρ`` x
tolerance ``ε`` x difficulty x seed, running the propose-verify-correct loop for
each cell and emitting one :class:`RunRecord` per rollout. The aggregated curve
(`verisim.metrics.aggregate`) and the figure are produced from these records only
(SPEC-2 §7.3, §12) -- everything regenerates from the config + seeds.

Note (SPEC-2 §17.5): whether the swept difficulties make ``ρ=0`` drift "within the
tested horizon without being pathological" is the empirical tuning this experiment
exists to do; the committed config is a small, fast instance of the machinery, not
a tuned publication run. v0 difficulty is realized via the driver mix (the full
§2.4 depth/breadth dial remains future work).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.data.drivers import Driver
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.loop.policy import fixed_interval_for_rho
from verisim.loop.runner import budget_for_rho, run_rollout
from verisim.metrics.record import RunRecord, write_records
from verisim.model.transformer import GPT, GPTConfig
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle
from verisim.train.dataset import build_dataset
from verisim.train.supervised import train_supervised


@dataclass(frozen=True)
class E1Config:
    name: str = "e1-small"
    # training
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 512
    train_iters: int = 400
    lr: float = 3e-3
    model_seed: int = 0
    # evaluation (difficulty name -> driver)
    difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    # sweep
    rhos: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.1)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> E1Config:
        base = E1Config()
        return E1Config(
            name=d.get("name", base.name),
            train_driver=d.get("train_driver", base.train_driver),
            train_seeds=tuple(d.get("train_seeds", base.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", base.train_steps_per_traj),
            n_layer=d.get("n_layer", base.n_layer),
            n_head=d.get("n_head", base.n_head),
            n_embd=d.get("n_embd", base.n_embd),
            block_size=d.get("block_size", base.block_size),
            train_iters=d.get("train_iters", base.train_iters),
            lr=d.get("lr", base.lr),
            model_seed=d.get("model_seed", base.model_seed),
            difficulties=dict(d.get("difficulties", base.difficulties)),
            eval_seeds=tuple(d.get("eval_seeds", base.eval_seeds)),
            eval_steps=d.get("eval_steps", base.eval_steps),
            rhos=tuple(d.get("rhos", base.rhos)),
            epsilons=tuple(d.get("epsilons", base.epsilons)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> E1Config:
        return E1Config.from_dict(json.loads(Path(path).read_text()))


def eval_actions(
    oracle: Oracle, config: EnvConfig, driver: str, seed: int, n_steps: int
) -> list[Action]:
    driver_obj = Driver(name=driver, config=config, rng=random.Random(seed))
    state = State.empty()
    actions: list[Action] = []
    for _ in range(n_steps):
        action = driver_obj.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def train_model(config: E1Config, vocab: Vocab, oracle: Oracle, env: EnvConfig) -> GPT:
    examples = build_dataset(
        oracle,
        vocab,
        env,
        driver=config.train_driver,
        seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab),
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )
    )
    train_supervised(
        model, examples, vocab.pad, steps=config.train_iters, lr=config.lr, seed=config.model_seed
    )
    return model


def run_e1(config: E1Config | None = None, *, oracle: Oracle | None = None) -> list[RunRecord]:
    """Train the model and run the full sweep; return one record per (cell, ε)."""
    config = config or E1Config()
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    model = train_model(config, vocab, oracle, env)
    world_model = NeuralWorldModel(model, vocab)

    records: list[RunRecord] = []
    for difficulty, driver in config.difficulties.items():
        for seed in config.eval_seeds:
            actions = eval_actions(oracle, env, driver, seed, config.eval_steps)
            for rho in config.rhos:
                # epsilon does not affect loop dynamics (only H_ε), so run the
                # rollout once per (difficulty, seed, rho) and spin out per-ε records.
                rollout = run_rollout(
                    world_model,
                    oracle,
                    State.empty(),
                    actions,
                    fixed_interval_for_rho(rho),
                    epsilon=config.epsilons[0],
                    budget=budget_for_rho(rho, len(actions)),
                    seed=seed,
                )
                for epsilon in config.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "model": "neural",
                                "difficulty": difficulty,
                                "driver": driver,
                                "rho": rho,
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

    parser = argparse.ArgumentParser(description="Run experiment E1 (H_eps(rho) curve).")
    parser.add_argument("--config", type=str, default=None, help="path to an E1 config JSON")
    parser.add_argument("--out", type=str, default="runs/e1/records.jsonl")
    args = parser.parse_args()
    config = E1Config.from_json_file(args.config) if args.config else E1Config()
    records = run_e1(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
