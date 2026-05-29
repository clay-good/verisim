"""Divergence and faithful-horizon metric tests (SPEC-2 §16, §7)."""

from __future__ import annotations

import random

from verisim.data import Driver
from verisim.env import DEFAULT_CONFIG, State, parse_action
from verisim.metrics import RunRecord, divergence, faithful_horizon
from verisim.oracle import ReferenceOracle


def test_divergence_zero_iff_identical():
    oracle = ReferenceOracle()
    driver = Driver(name="weighted", config=DEFAULT_CONFIG, rng=random.Random(5))
    state = State.empty()
    states = [state]
    for _ in range(60):
        state = oracle.step(state, driver.sample(state)).state
        states.append(state)
    # d = 0 exactly when comparing a state to itself; > 0 for any distinct pair.
    for i, a in enumerate(states):
        assert divergence(a, a) == 0.0
        for b in states[i + 1 :]:
            if a == b:
                assert divergence(a, b) == 0.0
            else:
                assert divergence(a, b) > 0.0


def test_divergence_in_unit_interval():
    a = State.empty()
    b = ReferenceOracle().step(a, parse_action("mkdir /x")).state
    d = divergence(a, b)
    assert 0.0 < d <= 1.0


def test_divergence_symmetric():
    a = State.empty()
    b = ReferenceOracle().step(a, parse_action("write /f alpha")).state
    assert divergence(a, b) == divergence(b, a)


def test_faithful_horizon_known_cases():
    assert faithful_horizon([0.0, 0.0, 0.0], 0.1) == 3  # never diverges
    assert faithful_horizon([0.0, 0.2, 0.0], 0.1) == 1  # first violation at index 1
    assert faithful_horizon([0.5, 0.0], 0.1) == 0  # diverges immediately
    assert faithful_horizon([], 0.1) == 0  # empty rollout
    assert faithful_horizon([0.1, 0.1], 0.1) == 2  # boundary: d == eps is faithful


def test_run_record_serialization_and_horizon():
    rec = RunRecord(
        config={"rho": 0.2, "policy": "fixed"},
        seed=42,
        epsilon=0.1,
        divergences=[0.0, 0.05, 0.2, 0.0],
        consultation_schedule=[True, False, True, False],
    )
    assert rec.faithful_horizon == 2
    assert rec.oracle_calls == 2
    restored = RunRecord.from_json(rec.to_json())
    assert restored == rec
    assert restored.faithful_horizon == 2
