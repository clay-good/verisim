"""NW3: graph divergence, reachability-faithfulness, and net bits-to-correct (SPEC-5 §9)."""

import random

from verisim.net import NetworkState, parse_net_action
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netdata import NetDriver
from verisim.netdelta.edits import FlowOpen, LinkAdd, NetDelta, SvcUp
from verisim.netmetrics import (
    bits_to_correct,
    correction_symbols,
    delta_exact,
    delta_exact_rate,
    divergence,
    reachability_faithfulness,
)
from verisim.netoracle import ReferenceNetworkOracle

ORACLE = ReferenceNetworkOracle()
HOSTS = DEFAULT_NET_CONFIG.hosts


def run(cmds: list[str]) -> NetworkState:
    state = NetworkState.initial(HOSTS)
    for cmd in cmds:
        state = ORACLE.step(state, parse_net_action(cmd)).state
    return state


def test_divergence_zero_iff_identical():
    a = run(["link_up h0 h1", "svc_up h1 80"])
    b = run(["link_up h0 h1", "svc_up h1 80"])
    assert divergence(a, b) == 0.0
    c = run(["link_up h0 h1", "svc_up h1 80", "svc_up h1 443"])
    assert 0.0 < divergence(a, c) <= 1.0


def test_reachability_faithfulness():
    a = run(["link_up h0 h1", "svc_up h1 80"])  # h0 can reach h1:80
    b = run(["link_up h0 h1", "svc_up h1 80"])
    assert reachability_faithfulness(a, b) == 1.0
    # a state where the firewall blocks the same service: reachability differs.
    blocked = run(["link_up h0 h1", "svc_up h1 80", "fw_deny h1 h0"])
    assert reachability_faithfulness(a, blocked) < 1.0


def test_bits_to_correct_zero_iff_equal_and_monotone():
    true: NetDelta = [LinkAdd("h0", "h1"), SvcUp("h1", 80)]
    assert bits_to_correct(true, list(true)) == 0.0
    near = [LinkAdd("h0", "h1")]  # missing one edit
    assert bits_to_correct([], true) > bits_to_correct(near, true) > 0.0
    # an invented edit counts as residual.
    extra: NetDelta = [LinkAdd("h0", "h1"), SvcUp("h1", 80), FlowOpen("h0", "h1", 80)]
    assert correction_symbols(extra, true) > 0


def test_delta_exact_matches_bits_to_correct_gate():
    true: NetDelta = [LinkAdd("h0", "h1"), SvcUp("h1", 80)]
    # exact iff bits-to-correct is 0; order-independent (multiset), like the bits gate.
    assert delta_exact(list(reversed(true)), true) is True
    assert bits_to_correct(list(reversed(true)), true) == 0.0
    wrong = [LinkAdd("h0", "h1")]
    assert delta_exact(wrong, true) is False
    assert bits_to_correct(wrong, true) > 0.0


def test_delta_exact_rate():
    true: NetDelta = [SvcUp("h1", 80)]
    pairs: list[tuple[NetDelta, NetDelta]] = [
        (list(true), true), ([LinkAdd("h0", "h1")], true), (list(true), true)
    ]
    assert delta_exact_rate(pairs) == 2 / 3
    assert delta_exact_rate([]) == 1.0  # vacuous: no steps to be wrong on


def test_divergence_in_unit_interval_over_trajectories():
    driver = NetDriver("adversarial", DEFAULT_NET_CONFIG, random.Random(0))
    prev = NetworkState.initial(HOSTS)
    state = prev
    for _ in range(40):
        state = ORACLE.step(state, driver.sample(state)).state
        d = divergence(prev, state)
        assert 0.0 <= d <= 1.0
        f = reachability_faithfulness(prev, state)
        assert 0.0 <= f <= 1.0
        prev = state
