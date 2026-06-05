"""DS4 increment 1 — the distributed `M_θ` vocab + tokenizer foundation (SPEC-7 §6.1).

Torch-free: the vocab and tokenizer carry no torch dependency (the constrained decoder + the learned
model land in increment 2), so these run in the dependency-free core. The load-bearing property is
the **round-trip**: ``parse_target(encode_target(Δ), state, action) == Δ`` for every delta the
reference oracle produces over real rollouts -- the serialization the learned arm will be built on
must be exact, or the M1 invariant (``apply(state, Δ) == oracle.next_state``) cannot survive decode.
"""

from __future__ import annotations

import random

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DEFAULT_DIST_CONFIG, scaled_dist_config
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DistDriver
from verisim.distmodel.tokenizer import (
    DistTokenizeError,
    encode_prompt,
    encode_target,
    parse_target,
)
from verisim.distmodel.vocab import DistVocab
from verisim.distoracle.reference import ReferenceDistOracle


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


def test_vocab_is_closed_and_bijective():
    vocab = DistVocab(DEFAULT_DIST_CONFIG)
    # every id maps back to its token and vice versa (a bijection over [0, len))
    assert len(vocab) == len({vocab.token(i) for i in range(len(vocab))})
    for i in range(len(vocab)):
        assert vocab.id(vocab.token(i)) == i
    # the leaf classes are disjoint and cover their domains
    assert len(vocab.node_ids) == len(DEFAULT_DIST_CONFIG.nodes)
    assert len(vocab.int_ids) == vocab.max_int
    assert vocab.op_ids.isdisjoint(vocab.int_ids)


def test_target_round_trips_over_every_preset():
    """parse(encode(Δ)) == Δ for every delta of every preset rollout — the core guarantee."""
    config = DEFAULT_DIST_CONFIG
    vocab = DistVocab(config)
    seen_ops: set[str] = set()
    for preset in ("uniform", "contention", "adversarial"):
        for seed in range(6):
            for state, action, delta in _rollout(config, preset, seed, 40):
                ids = encode_target(delta, vocab)
                assert parse_target(ids, vocab, state, action) == delta
                seen_ops.update(type(e).__name__ for e in delta)
    # the rollouts must actually exercise the full edit vocabulary, not a trivial subset
    assert {"ReplicaWrite", "MsgSend", "MsgDeliver", "EventAppend", "PartitionSet",
            "NodeDown", "NodeUp", "ClockSet", "SetResult"} <= seen_ops


def test_round_trip_preserves_the_m1_invariant():
    """A delta decoded from its tokens still satisfies apply(state, Δ) == oracle.next_state."""
    config = DEFAULT_DIST_CONFIG
    vocab = DistVocab(config)
    oracle = ReferenceDistOracle(config)
    for state, action, delta in _rollout(config, "adversarial", 7, 40):
        decoded = parse_target(encode_target(delta, vocab), vocab, state, action)
        assert apply(state, decoded) == oracle.step(state, action).state


def test_prompt_is_well_formed_and_bounded():
    config = DEFAULT_DIST_CONFIG
    vocab = DistVocab(config)
    for state, action, _ in _rollout(config, "uniform", 1, 24):
        prompt = encode_prompt(state, action, vocab)
        assert prompt[0] == vocab.bos and prompt[-1] == vocab.gen
        assert all(0 <= t < len(vocab) for t in prompt)


def test_round_trips_on_a_larger_cluster():
    """The tokenizer is config-driven: a 5-node / 3-object cluster round-trips too."""
    config = scaled_dist_config(5, n_objects=3)
    vocab = DistVocab(config)
    for state, action, delta in _rollout(config, "adversarial", 3, 30):
        assert parse_target(encode_target(delta, vocab), vocab, state, action) == delta


def test_partition_groups_round_trip():
    """A multi-group partition delta (the nested-run target) round-trips exactly."""
    config = DEFAULT_DIST_CONFIG
    vocab = DistVocab(config)
    state = DistributedState.initial(config)
    action = parse_dist_action("partition n0 | n1 n2")
    delta = ReferenceDistOracle(config).step(state, action).delta
    assert parse_target(encode_target(delta, vocab), vocab, state, action) == delta


def test_int_pool_overflow_raises_clearly():
    config = DEFAULT_DIST_CONFIG
    vocab = DistVocab(config, max_int=4)  # deliberately tiny
    state = DistributedState.initial(config)
    action = parse_dist_action("advance 99")  # clock jumps past the pool
    delta = ReferenceDistOracle(config).step(state, action).delta
    try:
        encode_target(delta, vocab)
    except DistTokenizeError as exc:
        assert "out of the vocab pool" in str(exc)
    else:  # pragma: no cover - the assertion below makes the failure explicit
        raise AssertionError("expected a DistTokenizeError on int-pool overflow")
