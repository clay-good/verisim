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
from verisim.delta.edits import Delta
from verisim.env.action import Action, parse_action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop import (
    DriftTriggered,
    FixedInterval,
    HardReset,
    Never,
    NullModel,
    OracleBackedModel,
    Projection,
    Residual,
    StepContext,
    UncertaintyTriggered,
    budget_for_rho,
    fixed_interval_for_rho,
    ground_truth_rollout,
    run_rollout,
)
from verisim.metrics.divergence import divergence
from verisim.oracle.reference import ReferenceOracle


class SignalModel:
    """A null-delta model that also emits a scripted per-step uncertainty signal.

    Implements the loop's ``UncertaintyModel`` protocol so the runner threads its
    signal into the consultation policy; the empty delta makes it drift like the
    ``NullModel`` so consultation timing is what the tests probe.
    """

    def __init__(self, signals: list[float]) -> None:
        self.signals = signals
        self.i = 0

    def predict_delta(self, state: State, action: Action) -> Delta:
        self.i += 1
        return []

    def predict_delta_with_uncertainty(
        self, state: State, action: Action
    ) -> tuple[Delta, float]:
        signal = self.signals[self.i]
        self.i += 1
        return [], signal


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
    assert fixed_interval_for_rho(1.0).should_consult(StepContext(0))
    assert fixed_interval_for_rho(0.25) == FixedInterval(4)


# --- M7: triggered policies, correction operators, spend-down backstop ---


def test_uncertainty_triggered_fires_on_instantaneous_signal():
    """Consult exactly the steps whose signal exceeds τ (unlimited budget)."""
    oracle = ReferenceOracle()
    signals = [0.1, 0.9, 0.2, 0.8]
    actions = make_actions("weighted", seed=7, n=len(signals))
    record = run_rollout(
        SignalModel(signals),
        oracle,
        State.empty(),
        actions,
        UncertaintyTriggered(tau=0.5),
        epsilon=0.0,
        budget=None,
    )
    assert record.consultation_schedule == [False, True, False, True]


def test_drift_triggered_fires_on_accumulated_signal_and_resets():
    """Small per-step signals accumulate to τ, fire, and the accumulator resets."""
    oracle = ReferenceOracle()
    signals = [0.6, 0.6, 0.6, 0.6]
    actions = make_actions("weighted", seed=8, n=len(signals))
    record = run_rollout(
        SignalModel(signals),
        oracle,
        State.empty(),
        actions,
        DriftTriggered(tau=1.0),
        epsilon=0.0,
        budget=None,
    )
    # cum: .6, 1.2>1 -> consult+reset, .6, 1.2>1 -> consult.
    assert record.consultation_schedule == [False, True, False, True]


def test_spend_down_backstop_spends_full_budget():
    """A never-triggering policy still spends the whole budget, at the tail."""
    oracle = ReferenceOracle()
    actions = make_actions("weighted", seed=9, n=5)
    record = run_rollout(
        NullModel(), oracle, State.empty(), actions, Never(), epsilon=0.0, budget=2
    )
    assert record.oracle_calls == 2
    assert record.consultation_schedule == [False, False, False, True, True]


def test_triggered_policies_match_fixed_budget():
    """At equal ρ every policy spends exactly the budget (true equal-ρ, §16)."""
    oracle = ReferenceOracle()
    signals = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    actions = make_actions("weighted", seed=10, n=len(signals))
    budget = budget_for_rho(0.5, len(actions))
    for policy in (
        fixed_interval_for_rho(0.5),
        UncertaintyTriggered(tau=0.5),
        DriftTriggered(tau=1.0),
    ):
        record = run_rollout(
            SignalModel(signals),
            oracle,
            State.empty(),
            actions,
            policy,
            epsilon=0.0,
            budget=budget,
        )
        assert record.oracle_calls == budget


def test_correction_operators_all_return_truth_and_log_diagnostics():
    """hard_reset/residual/projection coincide on the corrected state at v0."""
    oracle = ReferenceOracle()
    predicted = oracle.step(State.empty(), parse_action("mkdir /a")).state
    truth = oracle.step(State.empty(), parse_action("mkdir /b")).state

    assert HardReset().correct(predicted=predicted, truth=truth) is truth

    residual = Residual()
    assert residual.correct(predicted=predicted, truth=truth) is truth
    assert residual.discrepancies == [divergence_facts(predicted, truth)]

    projection = Projection()
    assert projection.correct(predicted=predicted, truth=truth) is truth
    assert projection.repaired_fractions == [divergence(predicted, truth)]


def divergence_facts(a: State, b: State) -> int:
    from verisim.metrics.divergence import state_facts

    return len(state_facts(a) ^ state_facts(b))
