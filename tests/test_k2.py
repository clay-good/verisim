"""Tests for the K1+K2 experiment: coverage, hard-negative mining, training (SPEC-2.1 §5-6)."""

import torch

from verisim.env.config import DEFAULT_CONFIG
from verisim.experiments.k2 import (
    K2Config,
    clean_faithfulness,
    mine_hard_negatives,
    run_k2,
)
from verisim.model.transformer import GPT, GPTConfig
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.reference import ReferenceOracle


def _tiny_model() -> NeuralWorldModel:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    vocab = Vocab(DEFAULT_CONFIG)
    gpt = GPT(GPTConfig(vocab_size=len(vocab), block_size=512, n_layer=1, n_head=2, n_embd=32))
    return NeuralWorldModel(gpt, vocab)


def test_mine_hard_negatives_bounded_and_deterministic():
    oracle = ReferenceOracle()
    model = _tiny_model()
    args = (oracle, model.vocab, DEFAULT_CONFIG, ("structural", "weighted"), (0, 1), 8, 5)
    a = mine_hard_negatives(model, *args)
    b = mine_hard_negatives(model, *args)
    assert len(a) <= 5
    assert a == b  # deterministic
    # each example is (prompt_ids, target_ids).
    for prompt, target in a:
        assert prompt and target


def test_clean_faithfulness_range():
    oracle = ReferenceOracle()
    model = _tiny_model()
    metrics = clean_faithfulness(model, oracle, DEFAULT_CONFIG, "structural", (100,), 6, 0.05)
    assert set(metrics) == {"exact", "acceptance", "graded"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    assert metrics["acceptance"] >= metrics["exact"]  # acceptance@ε is looser than exact


def _tiny_config() -> K2Config:
    return K2Config(
        coverage_drivers=("weighted",),
        coverage_seeds=(0, 1),
        coverage_steps=10,
        train_drivers=("structural",),
        train_seeds=(0, 1, 2, 3),
        val_seeds=(50,),
        steps_per_traj=6,
        eval_driver="structural",
        eval_seeds=(100,),
        eval_steps=6,
        n_layer=1,
        n_embd=32,
        train_steps=30,
        batch_size=8,
        eval_interval=15,
    )


def test_run_k2_tiny():
    records = run_k2(_tiny_config())
    assert len(records) == 2
    coverage, faith = records[0].config, records[1].config
    assert coverage["part"] == "coverage"
    assert coverage["cells"]  # some cells observed
    assert faith["part"] == "faithfulness"
    assert isinstance(faith["gate_passed"], bool)
    assert 0.0 <= faith["exact"] <= 1.0
    assert faith["acceptance"] >= faith["exact"]
