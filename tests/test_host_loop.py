"""Composed propose-verify-correct loop invariants (SPEC-6 §8, HC5).

Mirrors v0's ``test_loop`` and NW5's ``test_net_loop`` for the full-consultation mode, plus the
host-specific composed / per-subsystem invariants (§5.3, §8.2, §8.3):

  - ``ρ = 1`` + full consult reproduces the oracle exactly (composed ``H_ε = T``).
  - a perfect model never drifts even at ``ρ = 0``; a null model drifts.
  - the budget is never exceeded; the spend-down backstop spends it exactly.
  - the run-record carries the composed *and* every per-subsystem divergence trajectory (HC3).
  - a per-subsystem probe reveals/corrects only one subsystem, costs fewer oracle-bits than a full
    consult, and corrects strictly less -- so per-subsystem horizon is no greater than full-mode
    horizon at equal ``ρ`` (the §8.3 no-identity-collapse property).

Dependency-free (baselines only), like the rest of the deterministic core -- the learned ``M_θ``
drop-in is exercised in ``test_host_model``.
"""

from __future__ import annotations

import random

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostdata import HostDriver
from verisim.hostloop import (
    FixedSubsystem,
    HostNullModel,
    HostOracleBackedModel,
    PartialHostOracle,
    RoundRobinSubsystem,
    SubsystemFilter,
    budget_for_rho,
    ground_truth_rollout,
    run_host_rollout,
    subsystem_filter,
)
from verisim.hostloop.observe import full_bits
from verisim.hostmetrics.divergence import SUBSYSTEMS, divergence, divergence_by_subsystem
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import FixedInterval, Never, fixed_interval_for_rho

CONFIG = DEFAULT_HOST_CONFIG


def make_host_actions(driver_name: str, seed: int, n: int) -> list[HostAction]:
    """A seeded action sequence by rolling a workload driver against the oracle."""
    oracle = ReferenceHostOracle()
    driver = HostDriver(driver_name, CONFIG, random.Random(seed))
    state = HostState.initial()
    actions: list[HostAction] = []
    for _ in range(n):
        action = driver.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def _partial() -> PartialHostOracle:
    return PartialHostOracle(ReferenceHostOracle())


def s0() -> HostState:
    return HostState.initial()


# --- full-consultation mode (mirrors v0 test_loop) ---------------------------


def test_rho1_full_consult_reproduces_oracle_exactly():
    oracle = _partial()
    for driver_name in ("uniform", "forky", "adversarial"):
        actions = make_host_actions(driver_name, seed=1, n=40)
        record = run_host_rollout(
            HostNullModel(), oracle, s0(), actions, FixedInterval(1),
            epsilon=0.0, budget=len(actions),
        )
        assert all(d == 0.0 for d in record.divergences)
        assert record.faithful_horizon == len(actions)


def test_perfect_model_never_drifts_at_rho0():
    oracle = _partial()
    actions = make_host_actions("adversarial", seed=2, n=50)
    record = run_host_rollout(
        HostOracleBackedModel(ReferenceHostOracle()),
        oracle, s0(), actions, Never(), epsilon=0.0, budget=0,
    )
    assert record.oracle_calls == 0
    assert all(d == 0.0 for d in record.divergences)
    assert record.faithful_horizon == len(actions)


def test_rho0_matches_unaided_model_rollout():
    oracle = _partial()
    model = HostNullModel()
    actions = make_host_actions("forky", seed=3, n=30)
    gt = ground_truth_rollout(oracle, s0(), actions)

    state = s0()
    expected: list[float] = []
    for t, action in enumerate(actions):
        state = apply(state, model.predict_delta(state, action))
        expected.append(divergence(gt[t + 1], state))

    record = run_host_rollout(model, oracle, s0(), actions, Never(), epsilon=0.0, budget=0)
    assert record.divergences == expected


def test_null_model_drifts_at_rho0():
    oracle = _partial()
    actions = make_host_actions("forky", seed=4, n=30)
    record = run_host_rollout(HostNullModel(), oracle, s0(), actions, Never(), epsilon=0.0)
    assert record.faithful_horizon < len(actions)


def test_budget_is_never_exceeded():
    oracle = _partial()
    actions = make_host_actions("uniform", seed=5, n=40)
    for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        budget = budget_for_rho(rho, len(actions))
        record = run_host_rollout(
            HostNullModel(), oracle, s0(), actions, fixed_interval_for_rho(rho),
            epsilon=0.0, budget=budget,
        )
        assert record.oracle_calls <= budget
        assert record.oracle_calls == sum(record.consultation_schedule)
        assert len(record.divergences) == len(actions)


def test_spend_down_backstop_spends_full_budget():
    oracle = _partial()
    actions = make_host_actions("forky", seed=9, n=5)
    record = run_host_rollout(
        HostNullModel(), oracle, s0(), actions, Never(), epsilon=0.0, budget=2
    )
    assert record.oracle_calls == 2
    assert record.consultation_schedule == [False, False, False, True, True]


def test_record_carries_composed_and_per_subsystem_trajectories():
    """The HC3 run-record schema is fully populated by the loop (composed + per-subsystem)."""
    oracle = _partial()
    actions = make_host_actions("forky", seed=11, n=24)
    record = run_host_rollout(
        HostNullModel(), oracle, s0(), actions, fixed_interval_for_rho(0.25),
        epsilon=0.1, budget=budget_for_rho(0.25, len(actions)),
    )
    assert set(record.subsystem_divergences) == set(SUBSYSTEMS)
    for sub in SUBSYSTEMS:
        assert len(record.subsystem_divergences[sub]) == len(actions)
    # The component horizons are defined for every subsystem (the H13 diagnostic input). They are
    # independently normalized (each over its own subsystem's facts), so they are *not* ordered
    # against the composed horizon -- both views are first-class precisely because neither implies
    # the other (SPEC-6 §9.1), which is what makes the composition-law diagnostic non-trivial.
    assert set(record.subsystem_horizons) == set(SUBSYSTEMS)
    assert all(h >= 0 for h in record.subsystem_horizons.values())


# --- per-subsystem mode (no v0 analogue, SPEC-6 §5.3 / §8.2 / §8.3) -----------


def test_subsystem_filter_snaps_only_observed_subsystem():
    """A probe corrects exactly its subsystem to truth and keeps the belief for every other."""
    oracle = PartialHostOracle(ReferenceHostOracle())
    # Drive a few steps so the bundle is non-trivial (procs forked, fds open, a file written).
    actions = make_host_actions("forky", seed=6, n=12)
    state = s0()
    for action in actions[:-1]:
        state = oracle.full(state, action).state
    action = actions[-1]
    truth = oracle.full(state, action).state
    predicted = state  # null prediction: nothing changed (so most subsystems differ from truth)

    for sub in SUBSYSTEMS:
        probe = oracle.probe(state, action, sub)
        corrected = subsystem_filter(predicted, probe)
        per_sub = divergence_by_subsystem(corrected, truth)
        # The observed subsystem now agrees with truth exactly ...
        assert per_sub[sub] == 0.0
        # ... and every unobserved subsystem is the prediction verbatim (not snapped to truth).
        for other in SUBSYSTEMS:
            if other != sub:
                assert per_sub[other] == divergence_by_subsystem(predicted, truth)[other]


def test_per_subsystem_horizon_no_greater_than_full_at_equal_rho():
    oracle = _partial()
    actions = make_host_actions("forky", seed=7, n=40)
    rho = 0.3
    budget = budget_for_rho(rho, len(actions))

    full = run_host_rollout(
        HostNullModel(), oracle, s0(), actions, fixed_interval_for_rho(rho),
        epsilon=0.0, budget=budget,
    )
    op = SubsystemFilter()
    partial = run_host_rollout(
        HostNullModel(), oracle, s0(), actions, fixed_interval_for_rho(rho),
        epsilon=0.0, budget=budget, subsystem_policy=RoundRobinSubsystem(), subsystem_op=op,
    )
    # Equal budget spent, but a per-subsystem consult corrects strictly less.
    assert partial.oracle_calls == full.oracle_calls
    assert partial.faithful_horizon <= full.faithful_horizon
    # The operator logs a repaired fraction per consultation (the EH3 cost diagnostic).
    assert len(op.repaired_fractions) == partial.oracle_calls


def test_probe_costs_fewer_bits_than_full_consult():
    oracle = _partial()
    actions = make_host_actions("forky", seed=8, n=20)
    state = s0()
    for action in actions:
        truth = oracle.full(state, action).state
        full_cost = full_bits(truth)
        for sub in SUBSYSTEMS:
            assert oracle.probe(state, action, sub).bits <= full_cost
        state = truth


def test_fixed_subsystem_rejects_unknown_subsystem():
    import pytest

    with pytest.raises(ValueError, match="unknown subsystem"):
        FixedSubsystem("network")  # not a real subsystem in the HC0 bundle
