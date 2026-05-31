"""Tests for the K0 control experiment and its building blocks (SPEC-2.1 §4, §6)."""

import random

import torch

from verisim.data.drivers import Driver
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.experiments.e1 import E1Config
from verisim.experiments.k0 import K0Config, run_k0
from verisim.model.transformer import GPT, GPTConfig
from verisim.model.vocab import Vocab
from verisim.oracle.reference import ReferenceOracle
from verisim.train.dataset import build_dataset
from verisim.train.supervised import train_batched

_TRIVIAL_COMMANDS = {"mkdir", "touch", "write"}


def test_trivial_driver_additive_and_depth_one():
    oracle = ReferenceOracle()
    driver = Driver("trivial", DEFAULT_CONFIG, random.Random(0))
    state = State.empty()
    seen: set[str] = set()
    # <=8 single-segment names under root before the pool fills; stay within it.
    for _ in range(8):
        action = driver.sample(state)
        seen.add(action.name)
        target = action.args[0]
        assert target.count("/") == 1, f"trivial path must be depth-1, got {target}"
        state = oracle.step(state, action).state
    assert seen
    assert seen <= _TRIVIAL_COMMANDS


def test_train_batched_deterministic_and_learns():
    oracle = ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    examples = build_dataset(oracle, vocab, env, driver="trivial", seeds=(0, 1), n_steps=8)

    def run() -> list[float]:
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = GPT(
            GPTConfig(vocab_size=len(vocab), block_size=512, n_layer=1, n_head=2, n_embd=32)
        )
        return train_batched(model, examples, vocab.pad, steps=40, lr=3e-3, batch_size=8, seed=0)

    losses_a = run()
    losses_b = run()
    assert losses_a == losses_b  # deterministic given the seed
    assert min(losses_a) < losses_a[0]  # the loss falls


def test_train_batched_early_stopping_runs():
    oracle = ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    train = build_dataset(oracle, vocab, env, driver="trivial", seeds=(0, 1), n_steps=8)
    val = build_dataset(oracle, vocab, env, driver="trivial", seeds=(50,), n_steps=8)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = GPT(GPTConfig(vocab_size=len(vocab), block_size=512, n_layer=1, n_head=2, n_embd=32))
    losses = train_batched(
        model, train, vocab.pad, steps=30, lr=3e-3, batch_size=8, seed=0,
        val_examples=val, eval_interval=10,
    )
    assert len(losses) == 30


def test_run_k0_tiny():
    k0 = K0Config(
        train_seeds=(0, 1),
        val_seeds=(50,),
        eval_seeds=(100,),
        steps_per_traj=6,
        eval_steps=6,
        n_layer=1,
        n_embd=32,
        train_steps=30,
        batch_size=8,
        eval_interval=10,
    )
    diag = E1Config(
        train_seeds=(0,),
        train_steps_per_traj=6,
        train_iters=15,
        n_layer=1,
        n_embd=32,
        eval_seeds=(100,),
        eval_steps=6,
        difficulties={"low": "weighted"},
    )
    records = run_k0(k0, diagnostics_config=diag)
    assert len(records) == 2
    control = records[0].config
    assert control["part"] == "control"
    assert isinstance(control["gate_passed"], bool)
    assert 0.0 <= control["clean_faithfulness_exact"] <= 1.0
    assert 0.0 <= control["clean_faithfulness_graded"] <= 1.0
    assert records[1].config["part"] == "diagnostics"
