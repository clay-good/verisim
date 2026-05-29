"""Propose-verify-correct loop invariants (SPEC-2 §16).

  - ``ρ = 1`` reproduces the oracle exactly (``H_ε = T`` always).
  - ``ρ = 0`` matches the unaided model rollout.
  - A perfect model never drifts, even at ``ρ = 0``.
  - The consultation budget is never exceeded.
"""

from __future__ import annotations

import random

from verisim.data.drivers import DRIVERS, Driver
from verisim.delta.apply import apply
from verisim.env.action import Action, parse_action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop import (
    FixedInterval,
    HardReset,
    Never,
    NullModel,
    OracleBackedModel,
    budget_for_rho,
    fixed_interval_for_rho,
    ground_truth_rollout,
    run_rollout,
)
from verisim.metrics.divergence import divergence
from verisim.oracle.reference import ReferenceOracle


def make_actions(driver_name: str, seed: int, n: int) -> list[Action]:
    """A seeded action sequence by rolling a driver against the oracle."""
    oracle = ReferenceOracle()
    driver = Driver(name=driver_name, config=DEFAULT_CONFIG, rng=random.Random(seed))
    state = State.empty()
    actions: list[Action] = []
    for _ in range(n):
        action = driver.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def test_rho1_reproduces_oracle_exactly():
    """Consult every step + hard_reset => the coupled rollout never drifts."""
    oracle = ReferenceOracle()
    for driver_name in DRIVERS:
        actions = make_actions(driver_name, seed=1, n=50)
        record = run_rollout(
            NullModel(),  # even the trivial model is held exact at rho=1
            oracle,
            State.empty(),
            actions,
            FixedInterval(1),
            epsilon=0.0,
            budget=len(actions),
        )
        assert all(d == 0.0 for d in record.divergences)
        assert record.faithful_horizon == len(actions)


def test_perfect_model_never_drifts_at_rho0():
    oracle = ReferenceOracle()
    actions = make_actions("adversarial", seed=2, n=60)
    record = run_rollout(
        OracleBackedModel(oracle),
        oracle,
        State.empty(),
        actions,
        Never(),
        epsilon=0.0,
        budget=0,
    )
    assert record.oracle_calls == 0
    assert all(d == 0.0 for d in record.divergences)
    assert record.faithful_horizon == len(actions)


def test_rho0_matches_unaided_model_rollout():
    """At rho=0 the coupled states equal a plain autoregressive model rollout."""
    oracle = ReferenceOracle()
    model = NullModel()
    actions = make_actions("weighted", seed=3, n=40)
    gt = ground_truth_rollout(oracle, State.empty(), actions)

    state = State.empty()
    expected: list[float] = []
    for t, action in enumerate(actions):
        state = apply(state, model.predict_delta(state, action))
        expected.append(divergence(gt[t + 1], state))

    record = run_rollout(model, oracle, State.empty(), actions, Never(), epsilon=0.0, budget=0)
    assert record.divergences == expected


def test_null_model_drifts_at_rho0():
    oracle = ReferenceOracle()
    actions = make_actions("weighted", seed=4, n=40)
    record = run_rollout(NullModel(), oracle, State.empty(), actions, Never(), epsilon=0.0)
    # The first action builds structure the null model misses -> immediate drift.
    assert record.faithful_horizon < len(actions)


def test_budget_is_never_exceeded():
    oracle = ReferenceOracle()
    actions = make_actions("uniform", seed=5, n=50)
    for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        budget = budget_for_rho(rho, len(actions))
        record = run_rollout(
            NullModel(),
            oracle,
            State.empty(),
            actions,
            fixed_interval_for_rho(rho),
            epsilon=0.0,
            budget=budget,
        )
        assert record.oracle_calls <= budget
        assert record.oracle_calls == sum(record.consultation_schedule)
        assert len(record.divergences) == len(actions)
        assert len(record.consultation_schedule) == len(actions)


def test_budget_caps_an_eager_policy():
    """FixedInterval(1) wants to consult every step; the budget stops it early."""
    oracle = ReferenceOracle()
    actions = make_actions("weighted", seed=6, n=30)
    record = run_rollout(
        NullModel(), oracle, State.empty(), actions, FixedInterval(1), epsilon=0.0, budget=5
    )
    assert record.oracle_calls == 5
    # The eager policy spends the budget on the earliest steps.
    assert record.consultation_schedule[:5] == [True] * 5
    assert not any(record.consultation_schedule[5:])


def test_hard_reset_returns_truth():
    predicted = State.empty()
    truth = ReferenceOracle().step(predicted, parse_action("mkdir /x")).state
    assert HardReset().correct(predicted=predicted, truth=truth) is truth


def test_budget_and_policy_helpers():
    assert budget_for_rho(0.0, 100) == 0
    assert budget_for_rho(1.0, 100) == 100
    assert budget_for_rho(0.2, 50) == 10
    assert isinstance(fixed_interval_for_rho(0.0), Never)
    assert fixed_interval_for_rho(1.0).should_consult(0)
    assert fixed_interval_for_rho(0.25) == FixedInterval(4)
