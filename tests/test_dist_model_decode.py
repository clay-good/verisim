"""Distributed neural world model tests — DS4 increment 2 (SPEC-7 §6.1, the learned arm).

The torch layer the increment-1 serialization foundation was built for:

  - Constrained decoding produces only grammar-valid distributed deltas, even from an *untrained*
    model -- the guarantee is structural (the LL(1) :class:`DistDeltaGrammar`), not learned, and it
    holds across the rich delta shapes (partitions' nested node runs, status-dependent results).
  - The model overfits a tiny cluster to ~0 loss and reproduces the training deltas under
    constrained decoding -- the flat DS4 arm working end to end.
  - The decoded delta still satisfies the M1 invariant ``apply(state, Δ) == oracle.next_state`` and
    drops into the DS5 loop via the :class:`~verisim.distloop.model.DistModel` protocol unchanged.

Torch-only (``importorskip``), so the dependency-free core stays GPU-free; the increment-1
round-trip tests live in ``test_dist_model.py`` and run without torch.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from verisim.dist.action import parse_dist_action  # noqa: E402
from verisim.dist.config import DEFAULT_DIST_CONFIG, scaled_dist_config  # noqa: E402
from verisim.dist.delta import apply  # noqa: E402
from verisim.dist.state import DistributedState  # noqa: E402
from verisim.distdata.drivers import DistDriver  # noqa: E402
from verisim.distloop.model import DistModel  # noqa: E402
from verisim.distmodel import (  # noqa: E402
    DistDeltaGrammar,
    DistVocab,
    NeuralDistWorldModel,
    constrained_decode,
    constrained_decode_with_uncertainty,
    dist_examples_from_rollout,
    encode_prompt,
)
from verisim.distoracle.reference import ReferenceDistOracle  # noqa: E402
from verisim.model.transformer import GPT, GPTConfig  # noqa: E402
from verisim.train import teacher_forced_accuracy, train_supervised  # noqa: E402

CONFIG = DEFAULT_DIST_CONFIG
VOCAB = DistVocab(CONFIG)


def _tiny_model(block_size: int, *, seed: int, n_layer: int = 2, n_embd: int = 64) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            vocab_size=len(VOCAB), block_size=block_size, n_layer=n_layer, n_head=2, n_embd=n_embd
        )
    )


def _rollout(config, driver_name, seed, n_steps):
    """Yield ``(state, action, delta)`` for each step of one seeded reference rollout."""
    oracle = ReferenceDistOracle(config)
    driver = DistDriver(name=driver_name, config=config, rng=random.Random(seed))
    state = DistributedState.initial(config)
    for _ in range(n_steps):
        action = driver.sample(state)
        result = oracle.step(state, action)
        yield state, action, result.delta
        state = result.state


def test_constrained_decode_is_always_grammar_valid():
    """Even an untrained model can only emit parseable distributed deltas."""
    grammar = DistDeltaGrammar(VOCAB)
    model = _tiny_model(block_size=512, seed=99, n_layer=1, n_embd=32)
    for state, action, _ in _rollout(CONFIG, "adversarial", 2, 24):
        prompt = encode_prompt(state, action, VOCAB)
        delta = constrained_decode(model, prompt, VOCAB, state, action, grammar, max_edits=8)
        # If it parsed without raising, it is grammar-valid by construction.
        assert isinstance(delta, list)
    # The uncertainty variant returns a finite, non-negative mean entropy.
    state, action, _ = next(_rollout(CONFIG, "uniform", 5, 1))
    prompt = encode_prompt(state, action, VOCAB)
    _, entropy = constrained_decode_with_uncertainty(model, prompt, VOCAB, state, action, grammar)
    assert entropy >= 0.0


def test_decode_handles_partition_and_advance_shapes():
    """The two structured delta shapes — a nested partition run and a status-typed result —
    decode to grammar-valid deltas from an untrained model."""
    grammar = DistDeltaGrammar(VOCAB)
    model = _tiny_model(block_size=512, seed=7, n_layer=1, n_embd=32)
    state = DistributedState.initial(CONFIG)
    for raw in ("partition n0 | n1 n2", "advance 3", "put n0 x a"):
        action = parse_dist_action(raw)
        prompt = encode_prompt(state, action, VOCAB)
        delta = constrained_decode(model, prompt, VOCAB, state, action, grammar, max_edits=12)
        assert isinstance(delta, list)


def test_overfits_tiny_cluster_and_decodes_training_deltas():
    oracle = ReferenceDistOracle(CONFIG)
    examples = dist_examples_from_rollout(oracle, VOCAB, CONFIG, "uniform", seed=0, n_steps=24)
    block = max(len(p) + len(t) for p, t in examples) + 8
    model = _tiny_model(block_size=block, seed=0)

    losses = train_supervised(model, examples, VOCAB.pad, steps=300, lr=3e-3, seed=0)
    assert losses[-1] < 0.05, f"did not overfit: final loss {losses[-1]}"
    assert teacher_forced_accuracy(model, examples, VOCAB.pad) == 1.0

    wm = NeuralDistWorldModel(model, VOCAB)
    # An overfit model should reproduce essentially all training deltas free-running, and each
    # decoded delta must still satisfy the M1 invariant.
    matches = 0
    for state, action, delta in _rollout(CONFIG, "uniform", 0, 24):
        predicted = wm.predict_delta(state, action)
        if predicted == delta:
            matches += 1
            assert apply(state, predicted) == oracle.step(state, action).state
    assert matches >= len(examples) - 1, f"only {matches}/{len(examples)} decoded correctly"


def test_neural_model_satisfies_the_loop_protocol():
    """The learned model is a drop-in ``DistModel`` for the DS5 loop (the whole point of DS4)."""
    model = _tiny_model(block_size=256, seed=1, n_layer=1, n_embd=32)
    wm = NeuralDistWorldModel(model, VOCAB)
    assert isinstance(wm, DistModel)


def test_decode_is_config_driven_on_a_larger_cluster():
    """The grammar/decoder are config-driven: a 5-node / 3-object cluster decodes too."""
    config = scaled_dist_config(5, n_objects=3)
    vocab = DistVocab(config)
    grammar = DistDeltaGrammar(vocab)
    model = GPT(GPTConfig(vocab_size=len(vocab), block_size=512, n_layer=1, n_head=2, n_embd=32))
    for state, action, _ in _rollout(config, "adversarial", 3, 12):
        prompt = encode_prompt(state, action, vocab)
        delta = constrained_decode(model, prompt, vocab, state, action, grammar, max_edits=10)
        assert isinstance(delta, list)
