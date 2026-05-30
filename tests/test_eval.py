"""Faithfulness benchmark tests (SPEC-2 §15, milestone M8).

The benchmark core is dependency-free (no torch, no inspect): it scores any model
implementing the loop ``Model`` protocol. We use the b2/b3 baselines to pin the
ceiling (a perfect model is fully faithful) and floor (a trivial model drifts).
"""

from __future__ import annotations

import json

import pytest

from verisim.delta.serialize import delta_to_list
from verisim.env.serialize import to_canonical_str
from verisim.env.state import State
from verisim.eval import (
    DEFAULT_SUITE,
    FaithfulnessSample,
    applied_divergence,
    grade_prediction,
    score_model,
    score_suite,
    step_labels,
)
from verisim.loop import NullModel, OracleBackedModel
from verisim.oracle.reference import ReferenceOracle


def test_perfect_model_is_fully_faithful_unaided():
    oracle = ReferenceOracle()
    sample = FaithfulnessSample("adversarial", 200, 24, "high")
    score = score_model(OracleBackedModel(oracle), sample, oracle=oracle, rho=0.0)
    assert score.faithful_horizon == sample.n_steps == 24
    assert score.final_divergence == 0.0
    assert score.normalized_horizon == 1.0
    assert score.oracle_calls == 0


def test_trivial_model_drifts_unaided():
    sample = FaithfulnessSample("weighted", 100, 24, "low")
    score = score_model(NullModel(), sample, rho=0.0)
    assert score.faithful_horizon < sample.n_steps
    assert score.normalized_horizon < 1.0


def test_score_suite_covers_every_sample():
    scores = score_suite(OracleBackedModel(ReferenceOracle()))
    assert len(scores) == len(DEFAULT_SUITE)
    assert all(s.normalized_horizon == 1.0 for s in scores)


def test_step_labels_and_grade_prediction():
    sample = FaithfulnessSample("weighted", 100, 8, "low")
    labels = step_labels(sample)
    assert len(labels) == 8
    # Exact prediction scores 1.0; a clearly-wrong one scores < 1.0; garbage 0.0.
    label = labels[0]
    assert grade_prediction(label.next_state, label.next_state) == 1.0
    assert grade_prediction(to_canonical_str(State.empty()), label.next_state) < 1.0
    assert grade_prediction("not json", label.next_state) == 0.0


def test_applied_divergence_scores_a_predicted_delta():
    oracle = ReferenceOracle()
    sample = FaithfulnessSample("weighted", 101, 6, "low")
    state = State.empty()
    for action in sample.actions(oracle):
        result = oracle.step(state, action)
        state_str = to_canonical_str(state)
        truth_str = to_canonical_str(result.state)
        good = json.dumps(delta_to_list(result.delta))
        assert applied_divergence(state_str, good, truth_str) == 1.0
        assert applied_divergence(state_str, "not json", truth_str) == 0.0
        state = result.state


def test_inspect_adapter_builds_dataset_when_available():
    """If the optional ``[eval]`` extra is installed, the Inspect dataset builds."""
    pytest.importorskip("inspect_ai")
    from verisim.eval.inspect_task import faithfulness_dataset, faithfulness_task

    dataset = faithfulness_dataset()
    assert len(dataset) == sum(s.n_steps for s in DEFAULT_SUITE)
    assert faithfulness_task() is not None
