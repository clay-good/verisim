"""Experiment K0 — diagnose, and prove the learner works (SPEC-2.1 §4).

Two parts, both prerequisites for hunting the knee:

  1. **The control.** Train on the *trivial* difficulty (the ``trivial`` driver — additive,
     non-cascading commands, a deliberately learnable world) with the K2 batched trainer on a
     real coverage dataset, then measure clean (ρ=0) per-step faithfulness on **held-out**
     trajectories. **Gate: ≥ 0.95.** This proves the pipeline can fit the transition function
     at all — distinguishing the SPEC-2.1 §1 diagnosis (under-data/under-training, fixable)
     from a deeper representation/optimization bug (which would have to be fixed first).
  2. **The diagnostics** (``diagnose.run_diagnostics``) on the *baseline* config, to show
     *where* the under-trained model fails (per-command, per-edit, position, train-vs-val gap).

The control's contrast — trivial world trained properly reaches ≥0.95, while the baseline
config sits on the floor — is the K0 evidence that the floor is a data/training problem, not a
capacity or world-difficulty one (E4 already ruled out size).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.delta.apply import apply
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.metrics.divergence import divergence
from verisim.metrics.record import RunRecord, write_records
from verisim.model.transformer import GPT, GPTConfig
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle
from verisim.train.dataset import build_dataset
from verisim.train.supervised import train_batched

from .diagnose import run_diagnostics
from .e1 import E1Config, eval_actions


@dataclass(frozen=True)
class K0Config:
    name: str = "k0"
    # the trivial control world (depth-1 single-segment paths -> <=8 creates per trajectory)
    train_driver: str = "trivial"
    train_seeds: tuple[int, ...] = tuple(range(96))
    val_seeds: tuple[int, ...] = (96, 97, 98, 99)
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    steps_per_traj: int = 8
    eval_steps: int = 8
    # model + training budget
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    block_size: int = 512
    train_steps: int = 2500
    lr: float = 3e-3
    batch_size: int = 64
    eval_interval: int = 250
    model_seed: int = 0
    gate: float = 0.95

    @staticmethod
    def from_dict(d: dict[str, Any]) -> K0Config:
        base = K0Config()
        return K0Config(
            name=d.get("name", base.name),
            train_driver=d.get("train_driver", base.train_driver),
            train_seeds=tuple(d.get("train_seeds", base.train_seeds)),
            val_seeds=tuple(d.get("val_seeds", base.val_seeds)),
            eval_seeds=tuple(d.get("eval_seeds", base.eval_seeds)),
            steps_per_traj=d.get("steps_per_traj", base.steps_per_traj),
            eval_steps=d.get("eval_steps", base.eval_steps),
            n_layer=d.get("n_layer", base.n_layer),
            n_head=d.get("n_head", base.n_head),
            n_embd=d.get("n_embd", base.n_embd),
            block_size=d.get("block_size", base.block_size),
            train_steps=d.get("train_steps", base.train_steps),
            lr=d.get("lr", base.lr),
            batch_size=d.get("batch_size", base.batch_size),
            eval_interval=d.get("eval_interval", base.eval_interval),
            model_seed=d.get("model_seed", base.model_seed),
            gate=d.get("gate", base.gate),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> K0Config:
        return K0Config.from_dict(json.loads(Path(path).read_text()))


def _exact_match_faithfulness(
    model: NeuralWorldModel, oracle: Oracle, env: EnvConfig, driver: str,
    seeds: tuple[int, ...], n_steps: int,
) -> tuple[float, float]:
    """Clean (ρ=0) per-step faithfulness on held-out trajectories.

    Returns ``(exact_match, graded)``: the fraction of steps whose predicted delta exactly
    reproduces the oracle's next state (the gate metric), and the mean ``1 - divergence``
    (a smooth companion). Teacher-forced (state advances along the oracle's truth), so it is
    per-step and uncompounded.
    """
    exact = 0
    graded_sum = 0.0
    total = 0
    for seed in seeds:
        actions = eval_actions(oracle, env, driver, seed, n_steps)
        state = State.empty()
        for action in actions:
            truth = oracle.step(state, action).state
            d = divergence(apply(state, model.predict_delta(state, action)), truth)
            exact += int(d == 0.0)
            graded_sum += 1.0 - d
            total += 1
            state = truth
    if total == 0:
        return 1.0, 1.0
    return exact / total, graded_sum / total


def train_control_model(config: K0Config, vocab: Vocab, oracle: Oracle, env: EnvConfig) -> GPT:
    """Train the trivial-world control model with the K2 batched trainer (deterministic)."""
    import torch

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)
    train_examples = build_dataset(
        oracle, vocab, env, driver=config.train_driver,
        seeds=config.train_seeds, n_steps=config.steps_per_traj,
    )
    val_examples = build_dataset(
        oracle, vocab, env, driver=config.train_driver,
        seeds=config.val_seeds, n_steps=config.steps_per_traj,
    )
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=config.block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        )
    )
    train_batched(
        model, train_examples, vocab.pad,
        steps=config.train_steps, lr=config.lr, batch_size=config.batch_size,
        seed=config.model_seed, val_examples=val_examples, eval_interval=config.eval_interval,
    )
    return model


def run_k0(
    config: K0Config | None = None,
    *,
    oracle: Oracle | None = None,
    diagnostics_config: E1Config | None = None,
) -> list[RunRecord]:
    """Run the K0 control + diagnostics; return run-records (control first, then diagnostics).

    ``diagnostics_config`` is the *baseline* config whose failure modes are profiled (the
    floor we are trying to lift); it defaults to the in-code :class:`E1Config` baseline.
    """
    config = config or K0Config()
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)

    # 1. The control: trivial world, trained properly, measured held-out.
    model = NeuralWorldModel(train_control_model(config, vocab, oracle, env), vocab)
    exact, graded = _exact_match_faithfulness(
        model, oracle, env, config.train_driver, config.eval_seeds, config.eval_steps
    )
    n_train = len(config.train_seeds) * config.steps_per_traj
    control = RunRecord(
        config={
            "experiment": config.name,
            "part": "control",
            "world": "trivial",
            "clean_faithfulness_exact": exact,
            "clean_faithfulness_graded": graded,
            "gate": config.gate,
            "gate_passed": exact >= config.gate,
            "n_train_transitions": n_train,
            "train_steps": config.train_steps,
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
        },
        seed=config.model_seed,
        epsilon=0.0,
        divergences=[],
    )

    # 2. The diagnostics on the baseline config (where the under-trained model fails).
    diag = run_diagnostics(diagnostics_config or E1Config(), oracle=oracle)
    diagnostics = RunRecord(
        config={
            "experiment": config.name,
            "part": "diagnostics",
            "baseline_train_accuracy": diag.train_accuracy,
            "baseline_val_accuracy": diag.val_accuracy,
            "baseline_mean_bits_to_correct": diag.mean_bits_to_correct,
            "per_command": diag.per_command,
            "per_edit_pr": diag.per_edit_pr,
            "accuracy_by_position": diag.accuracy_by_position,
            "divergence_by_fact_type": diag.divergence_by_fact_type,
        },
        seed=config.model_seed,
        epsilon=0.0,
        divergences=[],
    )
    return [control, diagnostics]


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment K0 (control + diagnostics).")
    parser.add_argument("--config", type=str, default=None, help="path to a K0 config")
    parser.add_argument("--out", type=str, default="runs/k0/records.jsonl")
    args = parser.parse_args()
    config = K0Config.from_json_file(args.config) if args.config else K0Config()
    records = run_k0(config)
    path = write_records(records, args.out)
    control = records[0].config
    print(
        f"wrote {len(records)} records to {path}; "
        f"trivial-control clean faithfulness (exact)={control['clean_faithfulness_exact']:.3f} "
        f"graded={control['clean_faithfulness_graded']:.3f} "
        f"gate({control['gate']})={'PASS' if control['gate_passed'] else 'FAIL'}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
