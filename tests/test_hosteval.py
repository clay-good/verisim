"""Composed-host faithfulness benchmark tests (SPEC-6 §12 / HC8): the packaged metrology.

Mirrors v0's ``test_eval.py`` for the host world. Pins (1) the rollout-faithfulness scorer agrees
with the loop's own metrics on the dependency-free baselines (perfect model -> full horizon at
``ρ=0``; null model -> drifts immediately), (2) the single-step label grader scores an exact
prediction ``1.0`` and a corrupted / unparseable one strictly less, and (3) the benchmark is
regenerable from the sample manifest alone. The Inspect adapter is not imported (it needs the
optional ``[eval]`` extra, absent in CI); its core grader is tested directly.
"""

from __future__ import annotations

import json

from verisim.host.state import HostState, to_canonical_host
from verisim.hosteval import (
    DEFAULT_HOST_SUITE,
    HostFaithfulnessSample,
    applied_host_divergence,
    grade_host_prediction,
    host_step_labels,
    score_host_model,
    score_host_suite,
)
from verisim.hostloop import HostNullModel, HostOracleBackedModel
from verisim.hostoracle import ReferenceHostOracle


def test_perfect_model_is_fully_faithful_unaided() -> None:
    # The oracle-backed model never drifts even at ρ=0: full composed horizon, no oracle calls.
    sample = HostFaithfulnessSample("forky", 7, 24, "mid")
    score = score_host_model(HostOracleBackedModel(ReferenceHostOracle()), sample, rho=0.0)
    assert score.faithful_horizon == score.n_steps
    assert score.oracle_calls == 0
    assert score.normalized_horizon == 1.0


def test_null_model_drifts_without_oracle() -> None:
    sample = HostFaithfulnessSample("forky", 7, 24, "mid")
    score = score_host_model(HostNullModel(), sample, rho=0.0)
    # the null model predicts "nothing happens"; the very first state-changing syscall drifts it
    assert score.faithful_horizon < score.n_steps


def test_oracle_budget_restores_null_model() -> None:
    sample = HostFaithfulnessSample("forky", 7, 24, "mid")
    score = score_host_model(HostNullModel(), sample, rho=1.0)
    # ρ=1 consults every step -> the loop reproduces the oracle, full horizon
    assert score.faithful_horizon == score.n_steps
    assert score.oracle_calls == score.n_steps


def test_score_suite_covers_every_sample() -> None:
    scores = score_host_suite(HostOracleBackedModel(ReferenceHostOracle()), rho=0.0)
    assert len(scores) == len(DEFAULT_HOST_SUITE)
    assert all(s.faithful_horizon == s.n_steps for s in scores)


def test_grade_exact_prediction_is_one() -> None:
    label = host_step_labels(HostFaithfulnessSample("adversarial", 200, 12))[3]
    assert grade_host_prediction(label.next_state, label.next_state) == 1.0


def test_grade_wrong_prediction_is_below_one() -> None:
    labels = host_step_labels(HostFaithfulnessSample("adversarial", 200, 12))
    # grading a different step's state against this one's truth must score < 1
    score = grade_host_prediction(labels[0].next_state, labels[5].next_state)
    assert 0.0 <= score < 1.0


def test_grade_unparseable_prediction_is_zero() -> None:
    label = host_step_labels(HostFaithfulnessSample("forky", 1, 8))[0]
    assert grade_host_prediction("{not json", label.next_state) == 0.0


def test_labels_chain_state_to_next_state() -> None:
    from itertools import pairwise

    labels = host_step_labels(HostFaithfulnessSample("forky", 1, 10))
    for a, b in pairwise(labels):
        assert a.next_state == b.state  # the truth chains, step to step


def test_labels_regenerate_identically() -> None:
    sample = HostFaithfulnessSample("uniform", 100, 16)
    a = host_step_labels(sample)
    b = host_step_labels(sample)
    assert [(s.state, s.action, s.next_state) for s in a] == [
        (s.state, s.action, s.next_state) for s in b
    ]


def test_applied_delta_grader_matches_oracle() -> None:
    # The oracle's own delta, applied to the state, reproduces the truth -> grade 1.0.
    from verisim.host.delta import delta_to_list

    oracle = ReferenceHostOracle()
    sample = HostFaithfulnessSample("forky", 3, 12)
    state = HostState.initial()
    for action in sample.actions(oracle):
        result = oracle.step(state, action)
        s_json = json.dumps(to_canonical_host(state), separators=(",", ":"))
        ns_json = json.dumps(to_canonical_host(result.state), separators=(",", ":"))
        delta_json = json.dumps(delta_to_list(result.delta), separators=(",", ":"))
        assert applied_host_divergence(s_json, delta_json, ns_json) == 1.0
        state = result.state
