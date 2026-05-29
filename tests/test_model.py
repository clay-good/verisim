"""Neural world model tests (SPEC-2 §16, M4 verify).

  - Constrained decoding produces only grammar-valid deltas, even from an
    untrained model (the guarantee is structural, not learned).
  - The model overfits a tiny env to ~0 loss and reproduces the training deltas
    under constrained decoding.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.env import DEFAULT_CONFIG  # noqa: E402
from verisim.model import (  # noqa: E402
    GPT,
    DeltaGrammar,
    GPTConfig,
    Vocab,
    constrained_decode,
    parse_target,
)
from verisim.oracle import ReferenceOracle  # noqa: E402
from verisim.train import (  # noqa: E402
    examples_from_rollout,
    teacher_forced_accuracy,
    train_supervised,
)

VOCAB = Vocab(DEFAULT_CONFIG)


def _tiny_model(block_size: int, *, seed: int, n_layer: int = 2, n_embd: int = 64) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            vocab_size=len(VOCAB), block_size=block_size, n_layer=n_layer, n_head=2, n_embd=n_embd
        )
    )


def test_gpt_forward_shape():
    model = _tiny_model(block_size=32, seed=0)
    out = model(torch.zeros((3, 16), dtype=torch.long))
    assert out.shape == (3, 16, len(VOCAB))


def test_constrained_decode_is_always_grammar_valid():
    """Even an untrained model can only emit parseable deltas."""
    oracle = ReferenceOracle()
    examples = examples_from_rollout(oracle, VOCAB, DEFAULT_CONFIG, "weighted", seed=2, n_steps=15)
    grammar = DeltaGrammar(VOCAB)
    model = _tiny_model(block_size=160, seed=99, n_layer=1, n_embd=32)
    for prompt, _ in examples:
        delta, stdout = constrained_decode(
            model, prompt, VOCAB, grammar, max_edits=8, max_run=16
        )
        # If it parsed without raising, it is grammar-valid by construction.
        assert isinstance(delta, list)
        assert isinstance(stdout, str)


def test_overfits_tiny_env_and_decodes_training_deltas():
    oracle = ReferenceOracle()
    examples = examples_from_rollout(oracle, VOCAB, DEFAULT_CONFIG, "weighted", seed=0, n_steps=24)
    block = max(len(p) + len(t) for p, t in examples) + 8
    model = _tiny_model(block_size=block, seed=0)

    losses = train_supervised(model, examples, VOCAB.pad, steps=250, lr=3e-3, seed=0)
    assert losses[-1] < 0.05, f"did not overfit: final loss {losses[-1]}"
    assert teacher_forced_accuracy(model, examples, VOCAB.pad) == 1.0

    grammar = DeltaGrammar(VOCAB)
    matches = sum(
        constrained_decode(model, p, VOCAB, grammar)[0] == parse_target(t, VOCAB)[0]
        for p, t in examples
    )
    # An overfit model should reproduce essentially all training deltas free-running.
    assert matches >= len(examples) - 1, f"only {matches}/{len(examples)} decoded correctly"
