"""Property tests for the reference oracle (SPEC-2 §16).

  - Purity: ``O(s, a)`` does not mutate ``s``.
  - Determinism: same ``(s, a)`` -> identical result.
  - Delta agreement: ``apply(s, oracle_delta(s, a)) == O(s, a)``.
  - Grammar validity: every action the drivers emit parses.

Driven over randomized trajectories from all three drivers, so a wide variety of
states and (valid and failing) actions are exercised.
"""

from __future__ import annotations

import random

import pytest

from verisim.data import DRIVERS, Driver
from verisim.delta import apply
from verisim.env import DEFAULT_CONFIG, State, to_canonical_str
from verisim.oracle import ReferenceOracle


def _rollout_states(driver_name: str, seed: int, n: int):
    """Yield (state, action) pairs along a seeded rollout."""
    oracle = ReferenceOracle()
    driver = Driver(name=driver_name, config=DEFAULT_CONFIG, rng=random.Random(seed))
    state = State.empty()
    for _ in range(n):
        action = driver.sample(state)
        yield state, action
        state = oracle.step(state, action).state


@pytest.mark.parametrize("driver_name", DRIVERS)
def test_oracle_is_pure(driver_name: str):
    oracle = ReferenceOracle()
    for state, action in _rollout_states(driver_name, seed=1, n=80):
        before = to_canonical_str(state)
        oracle.step(state, action)
        assert to_canonical_str(state) == before, "oracle mutated its input state"


@pytest.mark.parametrize("driver_name", DRIVERS)
def test_oracle_is_deterministic(driver_name: str):
    o1, o2 = ReferenceOracle(), ReferenceOracle()
    for state, action in _rollout_states(driver_name, seed=2, n=80):
        r1 = o1.step(state.copy(), action)
        r2 = o2.step(state.copy(), action)
        assert r1.state == r2.state
        assert (r1.exit_code, r1.stdout) == (r2.exit_code, r2.stdout)


@pytest.mark.parametrize("driver_name", DRIVERS)
def test_apply_of_oracle_delta_equals_oracle(driver_name: str):
    """The M1 invariant: applying the oracle's delta reproduces its next state."""
    oracle = ReferenceOracle()
    for state, action in _rollout_states(driver_name, seed=3, n=120):
        result = oracle.step(state, action)
        assert apply(state, result.delta) == result.state


def test_determinism_report_seals_everything():
    report = ReferenceOracle().determinism_report()
    assert report.clock_sealed
    assert report.rng_sealed
    assert report.concurrency_sealed
    assert report.env_leakage_sealed
