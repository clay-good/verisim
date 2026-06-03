"""Factored interaction-graph arm tests (SPEC-6 §6.1, DD-H1; HC4 increment 2 verify).

  - The torch-free featurization is deterministic and well-formed (masked, process-indexed).
  - Constrained decoding produces only grammar-valid bundle deltas, even from an untrained model
    (the guarantee is structural, not learned -- the same split the flat arm gets).
  - The factored arm overfits a tiny host workload to high one-step accuracy and reproduces the
    training deltas under constrained decoding.
  - It drops into the HC5 loop as a HostModel: ``apply(state, M_θ(state, a))`` is a valid state.
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
    HostVocab,
    build_host_graph,
    build_host_graph_dataset,
    build_host_graph_model,
    graph_teacher_forced_accuracy,
    parse_target,
    train_host_graph_model,
)
from verisim.hostmodel.graph import feature_dims  # noqa: E402
from verisim.hostoracle.reference import ReferenceHostOracle  # noqa: E402

CONFIG = DEFAULT_HOST_CONFIG
MAX_PID = 32


def test_featurization_is_deterministic_and_masked():
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(3))
    dims = feature_dims(CONFIG, MAX_PID)
    for _ in range(20):
        action = driver.sample(state)
        g1 = build_host_graph(state, action, CONFIG, MAX_PID)
        g2 = build_host_graph(state, action, CONFIG, MAX_PID)
        assert g1 == g2  # pure function of (state, action, config)
        assert g1.n_pids == MAX_PID + 1
        assert len(g1.node_features) == MAX_PID + 1
        assert all(len(f) == dims.node for f in g1.node_features)
        assert len(g1.graph_features) == dims.graph
        # node_mask marks exactly the live process table entries
        live = {pid for pid, p in state.procs.items() if pid <= MAX_PID}
        assert {i for i, m in enumerate(g1.node_mask) if m == 1.0} == live
        state = oracle.step(state, action).state


def test_constrained_decode_is_always_grammar_valid():
    """Even an untrained factored model can only emit parseable bundle deltas."""
    model = build_host_graph_model(
        HostVocab(CONFIG, max_pid=MAX_PID), CONFIG, max_pid=MAX_PID, d_model=24, mp_rounds=2, seed=1
    )
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    driver = HostDriver("adversarial", CONFIG, random.Random(2))
    for _ in range(15):
        action = driver.sample(state)
        delta = model.predict_delta(state, action)
        assert isinstance(delta, list)  # parsed without raising => grammar-valid by construction
        d2, var = model.predict_delta_with_uncertainty(state, action)
        assert d2 == delta and var >= 0.0  # belief variance is a finite, non-negative signal
        state = oracle.step(state, action).state


def test_overfits_tiny_host_and_decodes_training_deltas():
    vocab = HostVocab(CONFIG, max_pid=MAX_PID)
    oracle = ReferenceHostOracle()
    examples = build_host_graph_dataset(
        oracle, vocab, CONFIG, driver="forky", seeds=(0,), n_steps=24
    )
    model = build_host_graph_model(vocab, CONFIG, max_pid=MAX_PID, seed=0)
    losses = train_host_graph_model(model, examples, steps=400, lr=3e-3, batch_size=24, seed=0)
    assert losses[-1] < losses[0]
    # The factored arm fits the tiny world to high one-step token accuracy.
    assert graph_teacher_forced_accuracy(model, examples) > 0.95

    matches = sum(
        model.predict_delta(g_state, g_action) == parse_target(target, vocab)
        for (g_state, g_action), target in _replay(oracle)
    )
    assert matches >= 20  # an overfit model reproduces almost all training deltas free-running


def _replay(oracle: ReferenceHostOracle):
    """Re-walk the seed-0 forky rollout, yielding ((state, action), target_tokens)."""
    vocab = HostVocab(CONFIG, max_pid=MAX_PID)
    from verisim.hostmodel import encode_target

    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(0))
    out = []
    for _ in range(24):
        action = driver.sample(state)
        result = oracle.step(state, action)
        out.append(((state, action), encode_target(result.delta, vocab)))
        state = result.state
    return out


def test_factored_model_predictions_compose_to_valid_states():
    vocab = HostVocab(CONFIG, max_pid=MAX_PID)
    oracle = ReferenceHostOracle()
    examples = build_host_graph_dataset(
        oracle, vocab, CONFIG, driver="forky", seeds=(0,), n_steps=16
    )
    model = build_host_graph_model(vocab, CONFIG, max_pid=MAX_PID, seed=0)
    train_host_graph_model(model, examples, steps=120, lr=3e-3, batch_size=16, seed=0)

    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(0))
    for _ in range(16):
        action = driver.sample(state)
        next_state = apply(state, model.predict_delta(state, action))  # must not raise
        assert isinstance(next_state.procs, dict)
        state = oracle.step(state, action).state
