"""OG2: oracle hard-negatives + counterfactual branches (SPEC-8 §4.3).

Property tests for the deterministic negative/counterfactual factory: every one-edit-wrong
successor is ``≠ O(s, a)``, every counterfactual equals ``O(s, a')`` by construction, and the
counterfactual branches span the whole action grammar.
"""

from __future__ import annotations

import random

from verisim.net import NetworkState, parse_net_action
from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netdata import (
    NetDriver,
    counterfactual_branches,
    enumerate_actions,
    one_edit_negatives,
)
from verisim.netdata.negatives import is_hard_negative
from verisim.netmetrics import divergence
from verisim.netoracle import ReferenceNetworkOracle

ORACLE = ReferenceNetworkOracle()
CONFIG = DEFAULT_NET_CONFIG
HOSTS = CONFIG.hosts

# the 11 command names of the SPEC-5 v0 grammar (§3.2).
ALL_COMMANDS = {
    "host_up", "host_down", "link_up", "link_down", "svc_up", "svc_down",
    "fw_deny", "fw_allow", "connect", "close", "advance",
}


def _states(seed: int, n: int) -> list[tuple[NetworkState, NetAction]]:
    driver = NetDriver("weighted", CONFIG, random.Random(seed))
    state = NetworkState.initial(HOSTS)
    out: list[tuple[NetworkState, NetAction]] = []
    for _ in range(n):
        action = driver.sample(state)
        out.append((state, action))
        state = ORACLE.step(state, action).state
    return out


def test_every_negative_differs_from_true_successor():
    for state, action in _states(seed=1, n=15):
        true_next = ORACLE.step(state, action).state
        negatives = one_edit_negatives(true_next, CONFIG)
        assert negatives  # non-empty
        for neg in negatives:
            assert divergence(neg, true_next) > 0.0
            assert is_hard_negative(neg, true_next)


def test_negatives_are_distinct_one_edit_neighbors():
    state, action = _states(seed=2, n=8)[-1]
    true_next = ORACLE.step(state, action).state
    negatives = one_edit_negatives(true_next, CONFIG)
    # canonical enumeration yields distinct neighbors (no duplicate emitted state).
    seen: set[object] = set()
    for neg in negatives:
        key = (
            tuple(sorted((h, hs.up, hs.services, hs.fw_deny) for h, hs in neg.hosts.items())),
            tuple(sorted(neg.links)),
            tuple(sorted(neg.flows)),
        )
        assert key not in seen
        seen.add(key)


def test_negative_sampling_is_deterministic_and_bounded():
    state, action = _states(seed=3, n=5)[-1]
    true_next = ORACLE.step(state, action).state
    full = one_edit_negatives(true_next, CONFIG)
    sub_a = one_edit_negatives(true_next, CONFIG, limit=5, rng=random.Random(0))
    sub_b = one_edit_negatives(true_next, CONFIG, limit=5, rng=random.Random(0))
    assert len(sub_a) == 5
    assert [divergence(s, true_next) for s in sub_a] == [divergence(s, true_next) for s in sub_b]
    assert len(full) >= 5


def test_counterfactuals_equal_oracle_and_span_the_grammar():
    state, _action = _states(seed=4, n=6)[-1]
    branches = counterfactual_branches(state, ORACLE, CONFIG)
    covered = set()
    for alt, succ in branches:
        # each counterfactual successor is exactly the oracle's truth for that branch.
        assert divergence(succ, ORACLE.step(state, alt).state) == 0.0
        covered.add(alt.name)
    assert covered == ALL_COMMANDS  # coverage spans the action grammar


def test_enumerate_actions_is_one_per_command():
    actions = enumerate_actions(CONFIG)
    assert {a.name for a in actions} == ALL_COMMANDS
    assert len(actions) == len(ALL_COMMANDS)
    # all parse back identically (round-trip through the grammar).
    for a in actions:
        assert parse_net_action(a.raw).name == a.name
