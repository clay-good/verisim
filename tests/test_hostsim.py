"""Whole-machine simulator protocol tests (SPEC-6 §7, HC8).

Exercises the agent-facing API with the dependency-free baselines (no torch needed -- the protocol
logic is model-agnostic):

  - ``imagine`` rolls the model with no oracle and returns the predicted trajectory;
  - ``verify`` against a **perfect** (oracle-backed) model trusts the whole plan (faithful horizon
    == n_steps, zero divergence) and the task oracle agrees on predicted vs true;
  - ``verify`` against the **null** model drifts: the plan-faithful horizon is short and the model
    disagrees with the oracle on whether the task succeeded;
  - the task-level goals read the final host state correctly.
"""

from __future__ import annotations

from verisim.host.state import HostState
from verisim.hostloop import HostNullModel, HostOracleBackedModel
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.hostsim import (
    HostSimulator,
    all_of,
    file_content,
    proc_killed,
    proc_running,
)

# A small incident-response-shaped plan: spawn a worker, have it write a file, then kill it.
PLAN = ["fork 1", "open 2 /log", "write 2 0 alpha", "exit 2 0"]
GOAL = all_of(file_content("/log", "alpha"), proc_killed(2))


def _perfect() -> HostSimulator:
    return HostSimulator(HostOracleBackedModel(ReferenceHostOracle()))


def test_imagine_rolls_the_model_without_an_oracle():
    sim = _perfect()
    rollout = sim.imagine(HostState.initial(), PLAN)
    assert len(rollout.states) == len(PLAN) + 1  # s_0 plus one per step
    assert len(rollout.actions) == len(PLAN)
    # the perfect model's imagination matches the truth: pid 2 forked then killed, /log written
    assert GOAL.holds(rollout.final)


def test_verify_trusts_a_perfect_plan_and_the_task_oracle_agrees():
    sim = _perfect()
    report = sim.verify(HostState.initial(), PLAN, epsilon=0.0, goal=GOAL)
    assert report.n_steps == len(PLAN)
    assert all(d == 0.0 for d in report.divergences)
    assert report.plan_faithful_horizon == len(PLAN)  # the agent can trust the whole plan
    assert report.trusted
    assert report.oracle_calls == len(PLAN) and report.oracle_bits > 0
    # the task oracle (the third oracle) holds on truth, and the model agrees
    assert report.goal_true is True
    assert report.goal_predicted is True
    assert report.goal_agreement is True


def test_verify_exposes_drift_for_a_bad_model():
    sim = HostSimulator(HostNullModel())  # predicts "nothing happens"
    report = sim.verify(HostState.initial(), PLAN, epsilon=0.0, goal=GOAL)
    # the null model diverges from the very first state-changing step (the fork)
    assert report.plan_faithful_horizon < report.n_steps
    assert not report.trusted
    assert report.final_divergence > 0.0
    # the task actually succeeds on truth, but the null model does not believe it did
    assert report.goal_true is True
    assert report.goal_predicted is False
    assert report.goal_agreement is False


def test_verify_without_goal_leaves_task_fields_none():
    report = _perfect().verify(HostState.initial(), PLAN, epsilon=0.0)
    assert report.goal_predicted is None
    assert report.goal_true is None
    assert report.goal_agreement is None


def test_run_record_is_a_well_formed_host_run_record():
    record = _perfect().run_record(HostState.initial(), PLAN, epsilon=0.05)
    assert len(record.divergences) == len(PLAN)
    assert set(record.subsystem_divergences) == {"proc", "fd", "fs", "global"}
    assert record.faithful_horizon == len(PLAN)  # the perfect model never diverges


def test_goal_predicates_read_the_state():
    sim = _perfect()
    true_final = sim.verify(HostState.initial(), PLAN).true_final
    assert file_content("/log", "alpha").holds(true_final)
    assert not file_content("/log", "beta").holds(true_final)
    assert proc_killed(2).holds(true_final)
    assert not proc_running(2).holds(true_final)
    assert proc_running(1).holds(true_final)  # init stays alive
