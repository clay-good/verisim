"""Unit tests for the EN8/EN9 scale harness reduction (SPEC-8 §7.1, OG5).

Pure, torch-free: the world factory and the seed→(mean, CI) reduction the scale runners depend on.
"""

from __future__ import annotations

import math
import random

from verisim.experiments.scale_common import GapStat, disjoint_from_zero, summarize
from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState
from verisim.netdata.drivers import NetDriver
from verisim.netoracle import ReferenceNetworkOracle


def test_scaled_net_config_sizes() -> None:
    cfg = scaled_net_config(12, 4)
    assert len(cfg.hosts) == 12
    assert cfg.hosts[0] == "h0" and cfg.hosts[-1] == "h11"
    assert len(cfg.ports) == 4
    assert cfg.ports[:3] == (22, 80, 443)  # defaults first


def test_scaled_net_config_extends_ports_beyond_pool() -> None:
    cfg = scaled_net_config(3, 20)
    assert len(cfg.ports) == 20
    assert len(set(cfg.ports)) == 20  # all distinct
    assert cfg.ports[-1] == 10000 + (20 - 12) - 1


def test_scaled_net_config_validates() -> None:
    for bad in (1, 0, -1):
        try:
            scaled_net_config(bad, 3)
            raise AssertionError("expected ValueError for n_hosts < 2")
        except ValueError:
            pass


def test_scaled_world_drives_the_oracle() -> None:
    """A scaled world is fully usable: drivers sample it and the oracle steps it."""
    cfg = scaled_net_config(16, 5)
    driver = NetDriver(name="weighted", config=cfg, rng=random.Random(0))
    oracle = ReferenceNetworkOracle()
    state = NetworkState.initial(cfg.hosts)
    for _ in range(20):
        state = oracle.step(state, driver.sample(state)).state
    assert len(state.hosts) == 16


def test_summarize_mean_and_single_value_ci() -> None:
    s = summarize(5, "m", "gap", [0.2, 0.4, 0.6])
    assert math.isclose(s.mean, 0.4, abs_tol=1e-9)
    assert s.ci_lo <= s.mean <= s.ci_hi
    assert s.n == 3
    one = summarize(5, "m", "gap", [0.7])
    assert one.ci_lo == one.ci_hi == 0.7  # CI collapses to the single value


def test_summarize_is_deterministic() -> None:
    vals = [0.1, 0.3, 0.25, 0.5, 0.42]
    a = summarize(5, "m", "gap", vals)
    b = summarize(5, "m", "gap", vals)
    assert (a.mean, a.ci_lo, a.ci_hi) == (b.mean, b.ci_lo, b.ci_hi)


def test_disjoint_from_zero() -> None:
    assert disjoint_from_zero(GapStat(5, "m", "g", 0.3, 0.1, 0.5, 4))  # all positive
    assert disjoint_from_zero(GapStat(5, "m", "g", -0.3, -0.5, -0.1, 4))  # all negative
    assert not disjoint_from_zero(GapStat(5, "m", "g", 0.1, -0.2, 0.4, 4))  # straddles 0
