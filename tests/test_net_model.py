"""Network neural world model tests (SPEC-5 §6.1, NW4 verify).

  - Tokenizer round-trip: ``parse_target(encode_target(Δ)) == Δ`` for oracle deltas.
  - Constrained decoding produces only grammar-valid graph deltas, even from an
    untrained model (the guarantee is structural, not learned).
  - The model overfits a tiny network to ~0 loss and reproduces the training deltas
    under constrained decoding -- the flat NW4 arm, the H11 flat-Markov baseline.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from verisim.model.transformer import GPT, GPTConfig  # noqa: E402
from verisim.net.config import DEFAULT_NET_CONFIG  # noqa: E402
from verisim.net.state import NetworkState  # noqa: E402
from verisim.netdata import NetDriver  # noqa: E402
from verisim.netmodel import (  # noqa: E402
    NetDeltaGrammar,
    NetVocab,
    constrained_decode,
    constrained_decode_with_uncertainty,
    encode_target,
    net_examples_from_rollout,
    parse_target,
)
from verisim.netoracle import ReferenceNetworkOracle  # noqa: E402
from verisim.train import teacher_forced_accuracy, train_supervised  # noqa: E402

VOCAB = NetVocab(DEFAULT_NET_CONFIG)
CONFIG = DEFAULT_NET_CONFIG


def _tiny_model(block_size: int, *, seed: int, n_layer: int = 2, n_embd: int = 64) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            vocab_size=len(VOCAB), block_size=block_size, n_layer=n_layer, n_head=2, n_embd=n_embd
        )
    )


def test_target_roundtrip_over_oracle_deltas():
    """``parse_target`` inverts ``encode_target`` on real oracle transitions."""
    oracle = ReferenceNetworkOracle()
    state = NetworkState.initial(CONFIG.hosts)
    driver = NetDriver("weighted", CONFIG, random.Random(7))
    for _ in range(60):
        action = driver.sample(state)
        result = oracle.step(state, action)
        ids = encode_target(result.delta, VOCAB)
        assert parse_target(ids, VOCAB) == result.delta
        state = result.state


def test_constrained_decode_is_always_grammar_valid():
    """Even an untrained model can only emit parseable graph deltas."""
    oracle = ReferenceNetworkOracle()
    examples = net_examples_from_rollout(oracle, VOCAB, CONFIG, "adversarial", seed=2, n_steps=20)
    grammar = NetDeltaGrammar(VOCAB)
    model = _tiny_model(block_size=200, seed=99, n_layer=1, n_embd=32)
    for prompt, _ in examples:
        delta = constrained_decode(model, prompt, VOCAB, grammar, max_edits=8)
        # If it parsed without raising, it is grammar-valid by construction.
        assert isinstance(delta, list)
    # The uncertainty variant returns a finite, non-negative mean entropy.
    _, entropy = constrained_decode_with_uncertainty(model, examples[0][0], VOCAB, grammar)
    assert entropy >= 0.0


def test_overfits_tiny_network_and_decodes_training_deltas():
    oracle = ReferenceNetworkOracle()
    examples = net_examples_from_rollout(oracle, VOCAB, CONFIG, "weighted", seed=0, n_steps=32)
    block = max(len(p) + len(t) for p, t in examples) + 8
    model = _tiny_model(block_size=block, seed=0)

    losses = train_supervised(model, examples, VOCAB.pad, steps=300, lr=3e-3, seed=0)
    assert losses[-1] < 0.05, f"did not overfit: final loss {losses[-1]}"
    assert teacher_forced_accuracy(model, examples, VOCAB.pad) == 1.0

    grammar = NetDeltaGrammar(VOCAB)
    matches = sum(
        constrained_decode(model, p, VOCAB, grammar) == parse_target(t, VOCAB)
        for p, t in examples
    )
    # An overfit model should reproduce essentially all training deltas free-running.
    assert matches >= len(examples) - 1, f"only {matches}/{len(examples)} decoded correctly"
