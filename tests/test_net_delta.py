"""NW1: the ``apply(state, oracle.delta) == oracle.next_state`` invariant + delta round-trip."""

import random

from verisim.net import NetworkState
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netdata import NetDriver
from verisim.netdelta import apply, delta_from_list, delta_to_list
from verisim.netoracle import ReferenceNetworkOracle

ORACLE = ReferenceNetworkOracle()
CONFIG = DEFAULT_NET_CONFIG


def test_apply_equals_oracle_over_random_trajectories():
    for driver_name in ("uniform", "weighted", "adversarial"):
        for seed in range(8):
            driver = NetDriver(driver_name, CONFIG, random.Random(seed))
            state = NetworkState.initial(CONFIG.hosts)
            for _ in range(40):
                action = driver.sample(state)
                result = ORACLE.step(state, action)
                # the invariant: applying the oracle's delta reproduces its next state exactly.
                assert apply(state, result.delta) == result.state
                state = result.state


def test_delta_serialization_round_trip():
    driver = NetDriver("adversarial", CONFIG, random.Random(0))
    state = NetworkState.initial(CONFIG.hosts)
    for _ in range(50):
        action = driver.sample(state)
        result = ORACLE.step(state, action)
        assert delta_from_list(delta_to_list(result.delta)) == result.delta
        state = result.state
