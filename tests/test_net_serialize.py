"""NW0: canonical state serialization round-trips to identity (SPEC-5 §3.1)."""

import random

from verisim.net import NetworkState, from_canonical, to_canonical
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netdata import NetDriver
from verisim.netoracle import ReferenceNetworkOracle


def test_state_round_trip_identity():
    oracle = ReferenceNetworkOracle()
    driver = NetDriver("weighted", DEFAULT_NET_CONFIG, random.Random(3))
    state = NetworkState.initial(DEFAULT_NET_CONFIG.hosts)
    for _ in range(60):
        state = oracle.step(state, driver.sample(state)).state
        assert from_canonical(to_canonical(state)) == state


def test_canonical_form_is_order_independent():
    a = NetworkState.initial(("h0", "h1"))
    a.links.add(("h0", "h1"))
    b = NetworkState.initial(("h1", "h0"))  # different insertion order
    b.links.add(("h0", "h1"))
    assert to_canonical(a) == to_canonical(b)
