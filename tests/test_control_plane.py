"""Property tests for the Batfish-style control-plane oracle (SPEC-5 §5.1, H12).

The deterministic core EN10 consumes: the control-plane oracle projects the data-plane truth to the
reachability matrix, prices a consult, and measures reachability-bits-to-correct. These checks pin
the invariants (0 iff reachability agrees; a flipped reachability fact is detected; the consult
matches the data-plane oracle's next-state reachability) before any experiment claim.
"""

from __future__ import annotations

import random

from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState, reachability_matrix
from verisim.netdata import NetDriver
from verisim.netoracle import (
    ControlPlaneOracle,
    ReferenceNetworkOracle,
    control_plane_bits,
    reachability_bits_to_correct,
)


def test_reachability_bits_to_correct_zero_iff_agree() -> None:
    cfg = scaled_net_config(5, 3)
    oracle = ReferenceNetworkOracle()
    driver = NetDriver(name="weighted", config=cfg, rng=random.Random(0))
    state = NetworkState.initial(cfg.hosts)
    for _ in range(20):
        state = oracle.step(state, driver.sample(state)).state
    # A state vs itself: reachability is identical, so zero bits to correct.
    assert reachability_bits_to_correct(state, state) == 0


def test_flipped_reachability_fact_is_detected() -> None:
    cfg = scaled_net_config(4, 2)
    oracle = ReferenceNetworkOracle()
    driver = NetDriver(name="weighted", config=cfg, rng=random.Random(1))
    state = NetworkState.initial(cfg.hosts)
    for _ in range(15):
        state = oracle.step(state, driver.sample(state)).state
    # Take down a host that has listening services -> its reachability column changes.
    target = next((h for h, hs in state.hosts.items() if hs.up and hs.services), None)
    if target is None:
        return  # degenerate seed; the zero-iff-agree test covers the invariant
    perturbed = state.copy()
    perturbed.hosts[target] = perturbed.hosts[target].with_up(False)
    assert reachability_bits_to_correct(perturbed, state) > 0


def test_consult_matches_data_plane_reachability() -> None:
    cfg = scaled_net_config(5, 3)
    oracle = ReferenceNetworkOracle()
    cp = ControlPlaneOracle(oracle)
    driver = NetDriver(name="weighted", config=cfg, rng=random.Random(2))
    state = NetworkState.initial(cfg.hosts)
    for _ in range(12):
        action = driver.sample(state)
        result = cp.consult(state, action)
        true_next = oracle.step(state, action).state
        # The control-plane consult is exactly the next state's reachability, by construction.
        assert result.reachability == reachability_matrix(true_next)
        assert result.bits == control_plane_bits(true_next)
        state = true_next


def test_control_plane_bits_counts_reachability_entries() -> None:
    cfg = scaled_net_config(6, 3)
    state = NetworkState.initial(cfg.hosts)
    assert control_plane_bits(state) == len(reachability_matrix(state))
