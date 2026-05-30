"""Stage-2 RLVR tests (SPEC-2 §5.3 Stage 2, §16).

  - The grammar-constrained *sampler* emits only parseable deltas and a
    grad-tracking log-prob (so a policy gradient can backprop through it).
  - ``train_rlvr`` is deterministic given a seed (reproducibility regime §12).
  - The oracle faithful-horizon reward is a usable learning signal: from scratch
    on a tiny env the mean return climbs, and parameters move.
  - RLVR does not destroy an already-faithful model (non-collapse).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.env import DEFAULT_CONFIG  # noqa: E402
from verisim.model import GPT, DeltaGrammar, GPTConfig, Vocab  # noqa: E402
from verisim.oracle import ReferenceOracle  # noqa: E402
from verisim.train import (  # noqa: E402
    examples_from_rollout,
    sample_delta_with_logprob,
    train_rlvr,
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


def test_sampler_is_grammar_valid_and_differentiable():
    """Even an untrained model samples only parseable deltas, with a grad log-prob."""
    oracle = ReferenceOracle()
    examples = examples_from_rollout(oracle, VOCAB, DEFAULT_CONFIG, "weighted", seed=2, n_steps=8)
    grammar = DeltaGrammar(VOCAB)
    model = _tiny_model(block_size=256, seed=99, n_layer=1, n_embd=32)
    for prompt, _ in examples:
        delta, stdout, log_prob = sample_delta_with_logprob(
            model, prompt, VOCAB, grammar, max_edits=8, max_run=16
        )
        assert isinstance(delta, list)
        assert isinstance(stdout, str)
        assert log_prob.requires_grad
        assert log_prob.item() <= 1e-6  # a summed log-prob is non-positive


def test_train_rlvr_is_deterministic():
    """Same seed -> identical training trace (SPEC-2 §12)."""
    a = _tiny_model(block_size=256, seed=0)
    b = _tiny_model(block_size=256, seed=0)
    sa = train_rlvr(a, VOCAB, n_steps=6, seeds=(0,), samples_per_env=4, steps=6, seed=0)
    sb = train_rlvr(b, VOCAB, n_steps=6, seeds=(0,), samples_per_env=4, steps=6, seed=0)
    assert sa.returns == sb.returns
    assert sa.losses == sb.losses
    assert sa.baselines == sb.baselines
    assert len(sa.returns) == 6


def test_rlvr_reward_is_a_learning_signal():
    """From scratch on a tiny env, the faithful-horizon return climbs and weights move.

    Sampled deltas are length-capped (``max_edits``/``max_run``) so the untrained
    model's rollouts stay short and the test stays fast; the signal is unchanged.
    """
    model = _tiny_model(block_size=256, seed=1)
    before = model.head.weight.detach().clone()
    stats = train_rlvr(
        model,
        VOCAB,
        n_steps=6,
        seeds=(0,),
        samples_per_env=6,
        steps=24,
        seed=0,
        lr=1e-3,
        max_edits=8,
        max_run=12,
    )
    assert all(loss == loss for loss in stats.losses)  # no NaNs
    assert not torch.equal(before, model.head.weight)  # a gradient actually flowed
    # The mean faithful horizon rises off its zero start (observed 0.0 -> ~0.8).
    assert stats.returns[-1] >= stats.returns[0] + 0.3
    assert max(stats.returns) >= 0.5


def test_rlvr_does_not_collapse_a_faithful_model():
    """Warm-started from a faithful supervised model, RLVR keeps the horizon high."""
    oracle = ReferenceOracle()
    examples = examples_from_rollout(oracle, VOCAB, DEFAULT_CONFIG, "weighted", seed=0, n_steps=6)
    block = max(len(p) + len(t) for p, t in examples) + 8
    model = _tiny_model(block_size=block, seed=0)
    train_supervised(model, examples, VOCAB.pad, steps=80, lr=3e-3, seed=0)

    stats = train_rlvr(
        model, VOCAB, n_steps=6, seeds=(0,), samples_per_env=6, steps=12, seed=0, lr=5e-4
    )
    # A perfect supervised model scores H_ε = 6; RLVR must not wreck it.
    assert max(stats.returns) >= 5.0
    assert stats.returns[-1] >= 4.0
