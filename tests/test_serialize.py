"""Serialization round-trip identity (SPEC-2 §16): state and delta."""

from __future__ import annotations

import random

from verisim.data import DRIVERS, Driver
from verisim.delta import delta_from_list, delta_to_list
from verisim.env import (
    DEFAULT_CONFIG,
    State,
    from_canonical,
    to_canonical,
    to_canonical_str,
)
from verisim.oracle import ReferenceOracle


def test_state_roundtrip_identity():
    oracle = ReferenceOracle()
    driver = Driver(name="weighted", config=DEFAULT_CONFIG, rng=random.Random(7))
    state = State.empty()
    for _ in range(150):
        assert from_canonical(to_canonical(state)) == state
        state = oracle.step(state, driver.sample(state)).state


def test_canonical_str_is_order_independent():
    a = State.empty()
    a.fs["/z"] = a.fs["/"]  # reuse a Dir node
    a.fs["/a"] = a.fs["/"]
    b = State.empty()
    b.fs["/a"] = b.fs["/"]
    b.fs["/z"] = b.fs["/"]
    assert to_canonical_str(a) == to_canonical_str(b)


def test_delta_roundtrip_identity():
    oracle = ReferenceOracle()
    for driver_name in DRIVERS:
        driver = Driver(name=driver_name, config=DEFAULT_CONFIG, rng=random.Random(11))
        state = State.empty()
        for _ in range(60):
            result = oracle.step(state, driver.sample(state))
            assert delta_from_list(delta_to_list(result.delta)) == result.delta
            state = result.state
