"""OG1: oracle-grounded targets + the decidable/residual partition (SPEC-8 §3, §4.1-4.2).

Property tests for the deterministic target machinery that must ship before the EN8 GPU runs:
the mask partitions the next-state facts (union is s', intersection empty), the divergence target
equals ``netmetrics.divergence`` by construction, and full observation degenerates to all-decidable.
"""

from __future__ import annotations

import random

from verisim.net import NetworkState, parse_net_action
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netdata import NetDriver, oracle_targets
from verisim.netdata.grounding import fact_hosts, is_decidable
from verisim.netmetrics import divergence, net_facts
from verisim.netoracle import ReferenceNetworkOracle

ORACLE = ReferenceNetworkOracle()
CONFIG = DEFAULT_NET_CONFIG
HOSTS = CONFIG.hosts


def _rollout_steps(seed: int, n: int) -> list[tuple[NetworkState, object]]:
    driver = NetDriver("weighted", CONFIG, random.Random(seed))
    state = NetworkState.initial(HOSTS)
    steps: list[tuple[NetworkState, object]] = []
    for _ in range(n):
        action = driver.sample(state)
        steps.append((state, action))
        state = ORACLE.step(state, action).state
    return steps


def test_partition_is_exact_and_disjoint():
    """D and R partition net_facts(s') (union all facts, disjoint) for every observation."""
    observations: list[frozenset[str] | None] = [
        None, frozenset(), frozenset({"h0"}), frozenset({"h0", "h1"}), frozenset(HOSTS)
    ]
    for state, action in _rollout_steps(seed=1, n=30):
        all_facts = net_facts(ORACLE.step(state, action).state)  # type: ignore[arg-type]
        for observed in observations:
            t = oracle_targets(state, action, ORACLE, observed)  # type: ignore[arg-type]
            assert t.decidable | t.residual == all_facts
            assert t.decidable & t.residual == frozenset()


def test_full_observation_makes_everything_decidable():
    """Full observation: ``R = ∅`` (the degenerate fully-observed case, SPEC-8 §3)."""
    for state, action in _rollout_steps(seed=2, n=20):
        t = oracle_targets(state, action, ORACLE, observed_hosts=None)  # type: ignore[arg-type]
        assert t.residual == frozenset()
        assert t.decidable == net_facts(t.next_state)


def test_empty_observation_leaves_only_global_facts_decidable():
    """With nothing observed, only the host-free clock/exit facts are decidable; the rest is R."""
    state, action = _rollout_steps(seed=3, n=5)[-1]
    t = oracle_targets(state, action, ORACLE, observed_hosts=set())  # type: ignore[arg-type]
    assert all(fact_hosts(f) == frozenset() for f in t.decidable)
    assert all(fact_hosts(f) != frozenset() for f in t.residual)


def test_divergence_target_equals_netmetrics_by_construction():
    state, action = _rollout_steps(seed=4, n=10)[-1]
    t = oracle_targets(state, action, ORACLE)  # type: ignore[arg-type]
    # against the true next-state the target is 0; against a perturbed state it equals netmetrics d.
    assert t.divergence_to(t.next_state) == 0.0
    other = ORACLE.step(state, parse_net_action("advance")).state
    assert t.divergence_to(other) == divergence(t.next_state, other)


def test_is_decidable_respects_referenced_hosts():
    # a firewall fact references two hosts; decidable only if both are observed.
    fw = ("fw", "h1", "h0")
    assert is_decidable(fw, frozenset({"h0", "h1"})) is True
    assert is_decidable(fw, frozenset({"h1"})) is False
    assert is_decidable(("\x00clock", 3), frozenset()) is True  # global fact, always decidable
