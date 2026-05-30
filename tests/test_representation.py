"""E4 representation-axis tests (SPEC-2 §9, §10, §16).

Covers the full-state prediction path:
  - ``state_to_delta`` is exact: ``apply(old, diff) == new`` over a real rollout.
  - constrained full-state decoding yields a valid ``State`` even untrained.
  - a full-state model overfits a tiny env (decodes the exact next state).
  - ``run_representation`` produces a record per (arm, cell), deterministically.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from verisim.data.drivers import Driver  # noqa: E402
from verisim.delta.apply import apply  # noqa: E402
from verisim.env import DEFAULT_CONFIG, State  # noqa: E402
from verisim.experiments.e1 import E1Config  # noqa: E402
from verisim.experiments.representation import (  # noqa: E402
    RepresentationConfig,
    run_representation,
)
from verisim.metrics import aggregate_values  # noqa: E402
from verisim.metrics.divergence import divergence  # noqa: E402
from verisim.model import (  # noqa: E402
    GPT,
    FullStateWorldModel,
    GPTConfig,
    StateGrammar,
    Vocab,
    constrained_decode_state,
    state_to_delta,
)
from verisim.oracle import ReferenceOracle  # noqa: E402
from verisim.train.dataset import state_examples_from_rollout  # noqa: E402
from verisim.train.supervised import train_supervised  # noqa: E402

VOCAB = Vocab(DEFAULT_CONFIG)


def _tiny_model(block_size: int, *, seed: int, n_layer: int = 2, n_embd: int = 64) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            vocab_size=len(VOCAB), block_size=block_size, n_layer=n_layer, n_head=2, n_embd=n_embd
        )
    )


def test_state_to_delta_is_exact_over_a_rollout():
    oracle = ReferenceOracle()
    driver = Driver(name="weighted", config=DEFAULT_CONFIG, rng=random.Random(5))
    state = State.empty()
    for _ in range(80):
        nxt = oracle.step(state, driver.sample(state)).state
        assert divergence(apply(state, state_to_delta(state, nxt)), nxt) == 0.0
        state = nxt


def test_constrained_decode_state_is_always_valid():
    """Even an untrained model can only decode a parseable State."""
    oracle = ReferenceOracle()
    examples = state_examples_from_rollout(oracle, VOCAB, DEFAULT_CONFIG, "weighted", 2, 12)
    grammar = StateGrammar(VOCAB)
    model = _tiny_model(block_size=512, seed=99, n_layer=1, n_embd=32)
    for prompt, _ in examples:
        state = constrained_decode_state(
            model, prompt, VOCAB, grammar, max_fs_entries=16, max_run=16
        )
        assert isinstance(state, State)


def test_full_state_model_overfits_tiny_env():
    oracle = ReferenceOracle()
    examples = state_examples_from_rollout(oracle, VOCAB, DEFAULT_CONFIG, "weighted", 0, 16)
    block = max(len(p) + len(t) for p, t in examples) + 8
    gpt = _tiny_model(block_size=block, seed=0)
    losses = train_supervised(gpt, examples, VOCAB.pad, steps=300, lr=3e-3, seed=0)
    assert losses[-1] < 0.1, f"did not overfit: final loss {losses[-1]}"

    model = FullStateWorldModel(gpt, VOCAB)
    state = State.empty()
    driver = Driver(name="weighted", config=DEFAULT_CONFIG, rng=random.Random(0))
    matches = 0
    n = 16
    for _ in range(n):
        action = driver.sample(state)
        truth = oracle.step(state, action).state
        if divergence(model.predict_state(state, action), truth) == 0.0:
            matches += 1
        state = truth
    # An overfit full-state model should reproduce essentially all training states.
    assert matches >= n - 1, f"only {matches}/{n} states decoded correctly"


def _tiny_config() -> RepresentationConfig:
    return RepresentationConfig(
        base=E1Config(
            train_seeds=(0, 1),
            train_steps_per_traj=16,
            train_iters=40,
            n_layer=1,
            n_embd=32,
            difficulties={"low": "weighted"},
            eval_seeds=(100, 101),
            eval_steps=6,
        )
    )


def test_run_representation_produces_a_record_per_arm_and_cell():
    records = run_representation(_tiny_config())
    # representations(2) x difficulties(1) x eval_seeds(2)
    assert len(records) == 2 * 1 * 2
    assert {r.config["representation"] for r in records} == {"delta", "full_state"}
    for rec in records:
        assert set(rec.config) >= {"representation", "difficulty", "step_accuracy", "clean_horizon"}
        assert 0.0 <= rec.config["step_accuracy"] <= 1.0
        assert rec.config["rho"] == 0.0


def test_run_representation_is_deterministic():
    a = run_representation(_tiny_config())
    b = run_representation(_tiny_config())
    assert [r.config["step_accuracy"] for r in a] == [r.config["step_accuracy"] for r in b]
    assert [r.config["clean_horizon"] for r in a] == [r.config["clean_horizon"] for r in b]


def test_aggregate_groups_representation_by_arm_and_difficulty():
    stats = aggregate_values(
        run_representation(_tiny_config()),
        group_keys=["representation", "difficulty"],
        value="step_accuracy",
    )
    assert {s.key for s in stats} == {("delta", "low"), ("full_state", "low")}
    for s in stats:
        assert s.n == 2
        assert 0.0 <= s.mean <= 1.0
