"""NW2: network drivers and trajectory generation (SPEC-5 §3.2)."""

import random

import pytest

from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.net.state import NetworkState
from verisim.netdata import NET_DRIVERS, NetDriver, generate_net_trajectory
from verisim.netoracle import ReferenceNetworkOracle


def test_driver_emits_only_valid_commands():
    oracle = ReferenceNetworkOracle()
    driver = NetDriver("adversarial", DEFAULT_NET_CONFIG, random.Random(0))
    state = NetworkState.initial(DEFAULT_NET_CONFIG.hosts)
    for _ in range(100):
        action = driver.sample(state)  # parse_net_action inside; raises on garbage
        assert action.name in {
            "host_up", "host_down", "link_up", "link_down", "svc_up", "svc_down",
            "fw_deny", "fw_allow", "connect", "close", "advance",
        }
        state = oracle.step(state, action).state


def test_generation_deterministic_per_seed():
    oracle = ReferenceNetworkOracle()
    a = generate_net_trajectory(oracle, DEFAULT_NET_CONFIG, "weighted", 0, 30)
    b = generate_net_trajectory(oracle, DEFAULT_NET_CONFIG, "weighted", 0, 30)
    assert a == b
    assert len(a) == 30
    assert a[0]["action"]  # canonical action string present


def test_all_presets_run():
    oracle = ReferenceNetworkOracle()
    for name in NET_DRIVERS:
        steps = generate_net_trajectory(oracle, DEFAULT_NET_CONFIG, name, 1, 20)
        assert len(steps) == 20


def test_unknown_driver_rejected():
    with pytest.raises(ValueError):
        NetDriver("nope", DEFAULT_NET_CONFIG, random.Random(0))
