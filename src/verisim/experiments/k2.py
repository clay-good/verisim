"""Experiment K1+K2 — coverage data, trained properly, past the acceptance floor (SPEC-2.1 §5-6).

K0 proved the pipeline can fit a deterministic transition (depth-1 control → exact 1.0) and
localized the floor to exact *multi-segment argument copying* into the delta. K1/K2 attack that
directly:

  - **K1 (coverage).** Report coverage of the transition space over a broad driver mix — every
    command x success/failure, plus the create-depth histogram (the copy-distribution axis).
    The K1 gate is that the dataset spans the space (``verisim.data.coverage``).
  - **K2 (train properly).** Train on the copy distribution (the ``structural`` driver:
    collision-free multi-depth creates) with the ``train_batched`` minibatch+schedule+early-stop
    trainer, optionally augmented with **hard negatives** (the (state,action) the current model
    gets most wrong by bits-to-correct), and measure clean (ρ=0) per-step faithfulness on a
    **held-out non-trivial difficulty**. **The K2 gate: exact > 0.5** — clearing the
    speculative-execution acceptance floor (SPEC-3 §8) so a knee becomes possible (K3/K4).

Records-only (SPEC-2 §7.3): one coverage record + one faithfulness record, regenerable from
``configs/k2.json`` + seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.data.coverage import coverage_report, missing_commands
from verisim.data.drivers import Driver
from verisim.delta.apply import apply
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.metrics.bits import bits_to_correct
from verisim.metrics.divergence import divergence
from verisim.metrics.record import RunRecord, write_records
from verisim.model.tokenizer import encode_prompt, encode_target
from verisim.model.transformer import GPT, GPTConfig
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle
from verisim.train.dataset import Example, build_dataset
from verisim.train.supervised import train_batched

from .e1 import eval_actions

_REQUIRED_COMMANDS = {
    "mkdir", "rmdir", "touch", "write", "append", "cat", "ls", "rm", "mv", "cp", "chmod",
    "cd", "export",
}


@dataclass(frozen=True)
class K2Config:
    name: str = "k2"
    # K1 coverage report (broad mix, for the coverage gate)
    coverage_drivers: tuple[str, ...] = ("trivial", "structural", "weighted", "adversarial")
    coverage_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    coverage_steps: int = 40
    # K2 training: the copy distribution
    train_drivers: tuple[str, ...] = ("structural",)
    train_seeds: tuple[int, ...] = tuple(range(160))
    val_seeds: tuple[int, ...] = (160, 161, 162, 163)
    steps_per_traj: int = 16
    # K2 eval: a held-out non-trivial difficulty
    eval_driver: str = "structural"
    eval_seeds: tuple[int, ...] = (300, 301, 302, 303, 304, 305, 306, 307)
    eval_steps: int = 16
    epsilon: float = 0.05
    # model + training budget
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    block_size: int = 512
    train_steps: int = 6000
    lr: float = 3e-3
    batch_size: int = 64
    eval_interval: int = 500
    # hard-negative mining (active learning); 0 rounds = off
    hard_negative_rounds: int = 0
    hard_negatives_per_round: int = 256
    mine_seeds: tuple[int, ...] = (400, 401, 402, 403)
    refine_steps: int = 1500
    model_seed: int = 0
    gate: float = 0.5

    @staticmethod
    def from_dict(d: dict[str, Any]) -> K2Config:
        b = K2Config()
        g = d.get
        return K2Config(
            name=g("name", b.name),
            coverage_drivers=tuple(g("coverage_drivers", b.coverage_drivers)),
            coverage_seeds=tuple(g("coverage_seeds", b.coverage_seeds)),
            coverage_steps=g("coverage_steps", b.coverage_steps),
            train_drivers=tuple(g("train_drivers", b.train_drivers)),
            train_seeds=tuple(g("train_seeds", b.train_seeds)),
            val_seeds=tuple(g("val_seeds", b.val_seeds)),
            steps_per_traj=g("steps_per_traj", b.steps_per_traj),
            eval_driver=g("eval_driver", b.eval_driver),
            eval_seeds=tuple(g("eval_seeds", b.eval_seeds)),
            eval_steps=g("eval_steps", b.eval_steps),
            epsilon=g("epsilon", b.epsilon),
            n_layer=g("n_layer", b.n_layer),
            n_head=g("n_head", b.n_head),
            n_embd=g("n_embd", b.n_embd),
            block_size=g("block_size", b.block_size),
            train_steps=g("train_steps", b.train_steps),
            lr=g("lr", b.lr),
            batch_size=g("batch_size", b.batch_size),
            eval_interval=g("eval_interval", b.eval_interval),
            hard_negative_rounds=g("hard_negative_rounds", b.hard_negative_rounds),
            hard_negatives_per_round=g("hard_negatives_per_round", b.hard_negatives_per_round),
            mine_seeds=tuple(g("mine_seeds", b.mine_seeds)),
            refine_steps=g("refine_steps", b.refine_steps),
            model_seed=g("model_seed", b.model_seed),
            gate=g("gate", b.gate),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> K2Config:
        return K2Config.from_dict(json.loads(Path(path).read_text()))


def mine_hard_negatives(
    model: NeuralWorldModel,
    oracle: Oracle,
    vocab: Vocab,
    env: EnvConfig,
    drivers: tuple[str, ...],
    seeds: tuple[int, ...],
    n_steps: int,
    k: int,
) -> list[Example]:
    """The ``k`` (state, action) the model gets most wrong (highest bits-to-correct).

    The active-learning loop the oracle makes free (SPEC-2.1 §5): roll the drivers forward
    teacher-forced, score each transition by the model's bits-to-correct against the oracle's
    true delta, and return the worst ``k`` as ready-to-train examples. Deterministic.
    """
    import random

    scored: list[tuple[float, Example]] = []
    for driver_name in drivers:
        for seed in seeds:
            driver = Driver(name=driver_name, config=env, rng=random.Random(seed))
            state = State.empty()
            for _ in range(n_steps):
                action = driver.sample(state)
                result = oracle.step(state, action)
                bits = bits_to_correct(model.predict_delta(state, action), result.delta)
                example = (
                    encode_prompt(state, action, vocab),
                    encode_target(result.delta, result.stdout, vocab),
                )
                scored.append((bits, example))
                state = result.state
    scored.sort(key=lambda t: t[0], reverse=True)
    return [example for _, example in scored[:k]]


def clean_faithfulness(
    model: NeuralWorldModel, oracle: Oracle, env: EnvConfig, driver: str,
    seeds: tuple[int, ...], n_steps: int, epsilon: float,
) -> dict[str, float]:
    """Clean (ρ=0) per-step faithfulness on held-out trajectories: exact, acceptance@ε, graded."""
    exact = acc = 0
    graded = 0.0
    total = 0
    for seed in seeds:
        actions = eval_actions(oracle, env, driver, seed, n_steps)
        state = State.empty()
        for action in actions:
            truth = oracle.step(state, action).state
            d = divergence(apply(state, model.predict_delta(state, action)), truth)
            exact += int(d == 0.0)
            acc += int(d <= epsilon)
            graded += 1.0 - d
            total += 1
            state = truth
    if total == 0:
        return {"exact": 1.0, "acceptance": 1.0, "graded": 1.0}
    return {"exact": exact / total, "acceptance": acc / total, "graded": graded / total}


def run_k2(config: K2Config | None = None, *, oracle: Oracle | None = None) -> list[RunRecord]:
    """Run K1 coverage report + K2 training/eval; return (coverage record, faithfulness record)."""
    import torch

    config = config or K2Config()
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)

    # K1: coverage of the transition space (the gate artifact).
    report = coverage_report(
        oracle, env, config.coverage_drivers, config.coverage_seeds, config.coverage_steps
    )
    coverage_record = RunRecord(
        config={
            "experiment": config.name,
            "part": "coverage",
            "cells": report.cells,
            "create_depths": {str(k): v for k, v in report.create_depths.items()},
            "n_failures": report.n_failures,
            "missing_commands": sorted(missing_commands(report, _REQUIRED_COMMANDS)),
        },
        seed=config.model_seed,
        epsilon=0.0,
        divergences=[],
    )

    # K2: train on the copy distribution, then measure on a held-out non-trivial difficulty.
    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)
    train_examples: list[Example] = []
    for driver in config.train_drivers:
        train_examples += build_dataset(
            oracle, vocab, env, driver=driver, seeds=config.train_seeds,
            n_steps=config.steps_per_traj,
        )
    val_examples = build_dataset(
        oracle, vocab, env, driver=config.eval_driver, seeds=config.val_seeds,
        n_steps=config.steps_per_traj,
    )
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=config.block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        )
    )
    train_batched(
        model, train_examples, vocab.pad, steps=config.train_steps, lr=config.lr,
        batch_size=config.batch_size, seed=config.model_seed,
        val_examples=val_examples, eval_interval=config.eval_interval,
    )
    world_model = NeuralWorldModel(model, vocab)

    for round_idx in range(config.hard_negative_rounds):
        negatives = mine_hard_negatives(
            world_model, oracle, vocab, env, config.train_drivers, config.mine_seeds,
            config.steps_per_traj, config.hard_negatives_per_round,
        )
        train_examples += negatives
        train_batched(
            model, train_examples, vocab.pad, steps=config.refine_steps, lr=config.lr,
            batch_size=config.batch_size, seed=config.model_seed + 1 + round_idx,
            val_examples=val_examples, eval_interval=config.eval_interval,
        )

    metrics = clean_faithfulness(
        world_model, oracle, env, config.eval_driver, config.eval_seeds, config.eval_steps,
        config.epsilon,
    )
    faithfulness_record = RunRecord(
        config={
            "experiment": config.name,
            "part": "faithfulness",
            "eval_driver": config.eval_driver,
            "exact": metrics["exact"],
            "acceptance": metrics["acceptance"],
            "graded": metrics["graded"],
            "epsilon": config.epsilon,
            "gate": config.gate,
            "gate_passed": metrics["exact"] > config.gate,
            "n_train_transitions": len(train_examples),
            "train_steps": config.train_steps,
            "hard_negative_rounds": config.hard_negative_rounds,
        },
        seed=config.model_seed,
        epsilon=0.0,
        divergences=[],
    )
    return [coverage_record, faithfulness_record]


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment K1+K2 (coverage + train).")
    parser.add_argument("--config", type=str, default=None, help="path to a K2 config")
    parser.add_argument("--out", type=str, default="runs/k2/records.jsonl")
    args = parser.parse_args()
    config = K2Config.from_json_file(args.config) if args.config else K2Config()
    records = run_k2(config)
    path = write_records(records, args.out)
    faith = records[1].config
    print(
        f"wrote {len(records)} records to {path}; structural clean faithfulness "
        f"exact={faith['exact']:.3f} acc@{faith['epsilon']}={faith['acceptance']:.3f} "
        f"graded={faith['graded']:.3f} gate({faith['gate']})="
        f"{'PASS' if faith['gate_passed'] else 'FAIL'}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
