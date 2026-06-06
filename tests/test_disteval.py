"""Distributed-cluster faithfulness benchmark tests (SPEC-7 §12 / DS8): the packaged metrology.

Mirrors the host ``test_hosteval.py``. Pins (1) the rollout-faithfulness scorer agrees with the
loop's own metrics on the dependency-free baselines (perfect model -> full horizon at ``ρ=0`` with
no oracle dollars; null model -> drifts immediately), (2) the consistency-faithful horizon is a
well-formed second metric that is at least the bit-faithful one (the §9.1 view forgives the
in-flight medium, ED5/H19), (3) the single-step label grader scores an exact prediction ``1.0`` and
corrupted / unparseable one strictly less, and (4) the benchmark is regenerable from the sample
manifest alone. The Inspect adapter is not imported (it needs the optional ``[eval]`` extra, absent
in CI); its core grader is tested directly.
"""

from __future__ import annotations

import json

from verisim.dist.config import DEFAULT_DIST_CONFIG
from verisim.dist.serialize import to_canonical
from verisim.dist.state import DistributedState
from verisim.disteval import (
    DEFAULT_DIST_SUITE,
    DistFaithfulnessSample,
    applied_dist_divergence,
    dist_step_labels,
    grade_dist_prediction,
    score_dist_model,
    score_dist_suite,
)
from verisim.distloop import DistNullModel, DistOracleBackedModel
from verisim.distoracle.reference import ReferenceDistOracle


def test_perfect_model_is_fully_faithful_unaided() -> None:
    # The oracle-backed model never drifts even at ρ=0: full horizon, no oracle calls/dollars.
    sample = DistFaithfulnessSample("adversarial", 7, 24, "high")
    score = score_dist_model(DistOracleBackedModel(ReferenceDistOracle()), sample, rho=0.0)
    assert score.faithful_horizon == score.n_steps
    assert score.consistency_horizon == score.n_steps
    assert score.oracle_calls == 0
    assert score.oracle_dollars == 0
    assert score.normalized_horizon == 1.0


def test_null_model_drifts_without_oracle() -> None:
    sample = DistFaithfulnessSample("adversarial", 7, 24, "high")
    score = score_dist_model(DistNullModel(), sample, rho=0.0)
    assert score.faithful_horizon < score.n_steps


def test_consistency_horizon_at_least_bit_horizon() -> None:
    # the §9.1 consistency view forgives the consistency-invisible in-flight medium (ED5/H19), so a
    # model's consistency-faithful horizon is never shorter than its bit-faithful one.
    sample = DistFaithfulnessSample("adversarial", 11, 24, "high")
    score = score_dist_model(DistNullModel(), sample, rho=0.0)
    assert score.consistency_horizon >= score.faithful_horizon


def test_full_consultation_reaches_full_horizon() -> None:
    # ρ=1 consults every step: the loop corrects to truth, so even the null model is fully faithful.
    sample = DistFaithfulnessSample("adversarial", 9, 16, "high")
    score = score_dist_model(DistNullModel(), sample, rho=1.0)
    assert score.faithful_horizon == score.n_steps
    assert score.oracle_dollars > 0  # spending the bit-exact tier costs dollars


def test_step_label_grader_scores_exact_one_and_corrupt_less() -> None:
    sample = DistFaithfulnessSample("uniform", 3, 6)
    labels = dist_step_labels(sample)
    assert labels
    label = labels[0]
    assert grade_dist_prediction(label.next_state, label.next_state) == 1.0
    assert grade_dist_prediction("not json", label.next_state) == 0.0
    # an empty-cluster prediction is strictly worse than exact for a step that changed something
    initial = json.dumps(to_canonical(DistributedState.initial(DEFAULT_DIST_CONFIG)),
                         separators=(",", ":"))
    assert grade_dist_prediction(initial, label.next_state) <= 1.0


def test_applied_delta_grader() -> None:
    sample = DistFaithfulnessSample("uniform", 3, 6)
    labels = dist_step_labels(sample)
    label = labels[0]
    # the empty delta applied to the state scores 1 - d(state, next_state); unparseable scores 0.
    assert applied_dist_divergence(label.state, "[]", label.next_state) <= 1.0
    assert applied_dist_divergence(label.state, "not json", label.next_state) == 0.0


def test_benchmark_regenerable_from_manifest() -> None:
    a = dist_step_labels(DEFAULT_DIST_SUITE[0])
    b = dist_step_labels(DEFAULT_DIST_SUITE[0])
    assert [x.next_state for x in a] == [x.next_state for x in b]


def test_score_suite_covers_every_sample() -> None:
    scores = score_dist_suite(DistOracleBackedModel(ReferenceDistOracle()), rho=0.0)
    assert len(scores) == len(DEFAULT_DIST_SUITE)
    assert all(s.faithful_horizon == s.n_steps for s in scores)
