"""The §7 LLM-callable cluster simulator — imagine / verify / change-safety (SPEC-7 §7).

Dependency-free (the loop's baselines satisfy ``DistModel``), GPU-free. Pins the §7 protocol: the
perfect model is trusted for the whole plan and agrees with the oracle on the task goal; the null
model drifts immediately; the consistency-faithful plan horizon outlasts the bit-exact one for an
in-flight (subtle) error (ED5/H19 at the plan level); and change-safety (the securifine differential
in consistency health) reads correctly, with the oracle as the verifier.
"""

from __future__ import annotations

import random

from verisim.dist.config import DEFAULT_DIST_CONFIG
from verisim.dist.state import DistributedState
from verisim.distloop.model import DistNoisyModel, DistNullModel, DistOracleBackedModel
from verisim.distoracle import ReferenceDistOracle
from verisim.distsim import (
    DistSimulator,
    consistency_health,
    no_split_brain,
    object_converged_to,
)

_PLAN = ["put n0 x v1", "advance 1", "advance 1", "get n1 x"]


def _oracle() -> ReferenceDistOracle:
    return ReferenceDistOracle(DEFAULT_DIST_CONFIG)


def _initial() -> DistributedState:
    return DistributedState.initial(DEFAULT_DIST_CONFIG)


def test_perfect_model_is_trusted_for_the_whole_plan():
    oracle = _oracle()
    sim = DistSimulator(DistOracleBackedModel(oracle), oracle, config=DEFAULT_DIST_CONFIG)
    report = sim.verify(_initial(), _PLAN, goal=object_converged_to("x", "v1"))
    assert report.trusted
    assert report.plan_faithful_horizon == len(_PLAN)
    assert report.consistency_plan_horizon == len(_PLAN)
    assert report.final_divergence == 0.0
    # the perfect model agrees with the oracle on the task goal and the change-safety verdict
    assert report.goal_predicted is True and report.goal_true is True and report.goal_agreement
    assert report.safety_agreement


def test_null_model_drifts_at_step_zero():
    oracle = _oracle()
    sim = DistSimulator(DistNullModel(), oracle, config=DEFAULT_DIST_CONFIG)
    report = sim.verify(_initial(), _PLAN)
    assert not report.trusted
    assert report.plan_faithful_horizon == 0


def test_imagine_runs_oracle_free_and_returns_a_rollout():
    oracle = _oracle()
    sim = DistSimulator(DistOracleBackedModel(oracle), oracle, config=DEFAULT_DIST_CONFIG)
    rollout = sim.imagine(_initial(), _PLAN)
    assert len(rollout.states) == len(_PLAN) + 1  # s_0 plus one predicted state per step
    assert len(rollout.actions) == len(_PLAN)
    assert rollout.final is rollout.states[-1]


def test_consistency_plan_horizon_outlasts_bit_horizon_for_in_flight_error():
    # a subtle (in-flight) corruption is bit-visible but consistency-invisible until delivery, so
    # agent can trust the model's split-brain prediction longer than its byte prediction (H19/plan).
    oracle = _oracle()
    model = DistNoisyModel(oracle, noise=1.0, mode="subtle", rng=random.Random(3), fallback=False)
    sim = DistSimulator(model, oracle, config=DEFAULT_DIST_CONFIG)
    plan = ["put n0 x v1", "advance 1", "advance 1", "put n1 x v2", "advance 1"]
    report = sim.verify(_initial(), plan)
    assert report.consistency_plan_horizon >= report.plan_faithful_horizon


def test_change_safety_is_a_differential_in_consistency_health():
    # consistency health is the fraction of converged objects; a freshly-booted cluster is fully
    # consistent (1.0), and a plan the perfect model+oracle agree leaves it converged is "safe".
    oracle = _oracle()
    assert consistency_health(_initial(), DEFAULT_DIST_CONFIG) == 1.0
    sim = DistSimulator(DistOracleBackedModel(oracle), oracle, config=DEFAULT_DIST_CONFIG)
    report = sim.verify(_initial(), _PLAN, goal=no_split_brain(DEFAULT_DIST_CONFIG))
    assert report.health_before == 1.0
    assert report.safe_true == (report.true_health_after >= report.health_before)
    assert report.safety_agreement  # the perfect model predicts the safety verdict correctly


def test_report_to_dict_is_serializable():
    oracle = _oracle()
    sim = DistSimulator(DistOracleBackedModel(oracle), oracle, config=DEFAULT_DIST_CONFIG)
    d = sim.verify(_initial(), _PLAN).to_dict()
    assert d["n_steps"] == len(_PLAN)
    assert set(d) >= {"plan_faithful_horizon", "consistency_plan_horizon", "safety_agreement",
                      "oracle_cost"}
