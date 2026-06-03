"""Host neural world model tests (SPEC-6 §6.1, HC4 verify).

  - Tokenizer round-trip: ``parse_target(encode_target(Δ)) == Δ`` for oracle bundle deltas
    (including the flattened embedded FS write delta, reconstructed exactly).
  - Constrained decoding produces only grammar-valid bundle deltas, even from an untrained model
    (the guarantee is structural, not learned).
  - The model overfits a tiny host workload to ~0 loss and reproduces the training deltas under
    constrained decoding -- the flat HC4 arm, the DD-H1 flat-serializer baseline.
  - The decoded delta composes: ``apply(state, M_θ(state, action))`` is a valid next bundle state
    (the M1-analogue invariant survives the learned proposer).
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from verisim.host.config import DEFAULT_HOST_CONFIG  # noqa: E402
from verisim.host.delta import apply  # noqa: E402
from verisim.host.state import HostState  # noqa: E402
from verisim.hostdata import HostDriver  # noqa: E402
from verisim.hostmodel import (  # noqa: E402
    HostDeltaGrammar,
    HostVocab,
    NeuralHostWorldModel,
    constrained_decode,
    constrained_decode_with_uncertainty,
    encode_target,
    host_examples_from_rollout,
    parse_target,
)
from verisim.hostoracle.reference import ReferenceHostOracle  # noqa: E402
from verisim.model.transformer import GPT, GPTConfig  # noqa: E402
from verisim.train import teacher_forced_accuracy, train_supervised  # noqa: E402

VOCAB = HostVocab(DEFAULT_HOST_CONFIG)
CONFIG = DEFAULT_HOST_CONFIG


def _tiny_model(block_size: int, *, seed: int, n_layer: int = 2, n_embd: int = 64) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            vocab_size=len(VOCAB), block_size=block_size, n_layer=n_layer, n_head=2, n_embd=n_embd
        )
    )


def test_target_roundtrip_over_oracle_deltas():
    """``parse_target`` inverts ``encode_target`` on real host transitions (all six syscalls)."""
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(7))
    for _ in range(80):
        action = driver.sample(state)
        result = oracle.step(state, action)
        ids = encode_target(result.delta, VOCAB)
        assert parse_target(ids, VOCAB) == result.delta
        state = result.state


def test_constrained_decode_is_always_grammar_valid():
    """Even an untrained model can only emit parseable bundle deltas."""
    oracle = ReferenceHostOracle()
    examples = host_examples_from_rollout(oracle, VOCAB, CONFIG, "adversarial", seed=2, n_steps=20)
    grammar = HostDeltaGrammar(VOCAB)
    model = _tiny_model(block_size=256, seed=99, n_layer=1, n_embd=32)
    for prompt, _ in examples:
        delta = constrained_decode(model, prompt, VOCAB, grammar, max_edits=8)
        # If it parsed without raising, it is grammar-valid by construction.
        assert isinstance(delta, list)
    # The uncertainty variant returns a finite, non-negative mean entropy.
    _, entropy = constrained_decode_with_uncertainty(model, examples[0][0], VOCAB, grammar)
    assert entropy >= 0.0


def test_overfits_tiny_host_and_decodes_training_deltas():
    oracle = ReferenceHostOracle()
    examples = host_examples_from_rollout(oracle, VOCAB, CONFIG, "forky", seed=0, n_steps=32)
    block = max(len(p) + len(t) for p, t in examples) + 8
    model = _tiny_model(block_size=block, seed=0)

    losses = train_supervised(model, examples, VOCAB.pad, steps=300, lr=3e-3, seed=0)
    assert losses[-1] < 0.05, f"did not overfit: final loss {losses[-1]}"
    assert teacher_forced_accuracy(model, examples, VOCAB.pad) == 1.0

    grammar = HostDeltaGrammar(VOCAB)
    matches = sum(
        constrained_decode(model, p, VOCAB, grammar) == parse_target(t, VOCAB)
        for p, t in examples
    )
    # An overfit model should reproduce essentially all training deltas free-running.
    assert matches >= len(examples) - 1, f"only {matches}/{len(examples)} decoded correctly"


def test_world_model_predictions_compose_to_valid_states():
    """The learned proposer's delta applies: ``apply(state, M_θ(state, a))`` is a valid state.

    The M1-analogue invariant (HC1) is model-agnostic -- it holds for whatever the loop's proposer
    emits, here the neural model under constrained decoding.
    """
    oracle = ReferenceHostOracle()
    examples = host_examples_from_rollout(oracle, VOCAB, CONFIG, "forky", seed=0, n_steps=24)
    block = max(len(p) + len(t) for p, t in examples) + 8
    model = _tiny_model(block_size=block, seed=0)
    train_supervised(model, examples, VOCAB.pad, steps=150, lr=3e-3, seed=0)
    wm = NeuralHostWorldModel(model, VOCAB)

    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(0))
    for _ in range(24):
        action = driver.sample(state)
        predicted = wm.predict_delta(state, action)
        next_state = apply(state, predicted)  # must not raise; a well-formed bundle state
        assert isinstance(next_state.procs, dict)
        state = oracle.step(state, action).state  # advance on ground truth (loop semantics)
