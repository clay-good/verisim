"""CU31 / H124 -- the concurrent (multi-agent) safety gate (SPEC-22 §5).

Torch-free. Unit tests pin the fleet mechanics (partition, per-agent attribution, joint vs per-agent
accumulators, joint coverage) on hand-built network states; the integration tests on the real
unified network arm (smoke) prove the verdict -- the per-agent gate leaks the joint danger across a
fleet, the shared closure covers it un-gameably and cheaply, ``covers`` predicts every candidate a
priori, and the per-agent leak appears exactly when the fleet fragments (K>=2).
"""

from __future__ import annotations

from typing import cast

from verisim.acd.concurrent_targeting import (
    CU31Config,
    Deployment,
    _agent_distinct_sensitive,
    _agent_hosts,
    _agent_of,
    _connect,
    _make_danger,
    covers_joint,
    cu31_verdict,
    joint_lowslow_breaches,
    partition_hosts,
    run_cu31,
)
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.net.state import HostState, NetworkState, link_key
from verisim.netoracle.reference import ReferenceNetworkOracle

_PORT = DEFAULT_NET_CONFIG.ports[0]
_HOSTS = ("h0", "h1", "h2", "h3", "h4")


def _state_with_flows(flows: set[tuple[str, str, int]]) -> NetworkState:
    """A FULLY-reachable topology (full mesh, every host listening on the port) + given flows.

    Full reachability matters because the danger/target functions recompute the post-action
    accumulator via ``oracle.step`` -- the crossing connect must actually open a flow.
    """
    hosts = {h: HostState(up=True, services=(_PORT,)) for h in _HOSTS}
    links = {link_key(a, b) for a in _HOSTS for b in _HOSTS if a < b}
    return NetworkState(hosts=hosts, links=links, flows=set(flows))


# --------------------------------------------------------------------------------------------------
# The fleet: partition + per-agent attribution.
# --------------------------------------------------------------------------------------------------


def test_partition_is_round_robin():
    part = partition_hosts(("h0", "h1", "h2", "h3", "h4"), 2)
    assert part == {"h0": 0, "h1": 1, "h2": 0, "h3": 1, "h4": 0}
    assert _agent_hosts(part) == {0: frozenset({"h0", "h2", "h4"}), 1: frozenset({"h1", "h3"})}


def test_single_agent_owns_everything():
    part = partition_hosts(("h0", "h1", "h2"), 1)
    assert set(part.values()) == {0}


def test_agent_of_attributes_by_src():
    part = partition_hosts(("h0", "h1", "h2", "h3", "h4"), 2)
    assert _agent_of(_connect("h0", "h2", _PORT), part) == 0  # src h0 -> agent 0
    assert _agent_of(_connect("h1", "h4", _PORT), part) == 1  # src h1 -> agent 1


def test_per_agent_accumulator_counts_only_own_flows():
    sensitive = frozenset(("h0", "h1", "h2", "h3", "h4"))
    state = _state_with_flows({("h0", "h2", _PORT), ("h0", "h3", _PORT), ("h1", "h4", _PORT)})
    a0 = frozenset({"h0", "h2", "h4"})  # agent 0's src hosts
    a1 = frozenset({"h1", "h3"})
    # agent 0 opened flows to h2,h3 (src h0); agent 1 opened a flow to h4 (src h1)
    assert _agent_distinct_sensitive(state, a0, sensitive) == {"h2", "h3"}
    assert _agent_distinct_sensitive(state, a1, sensitive) == {"h4"}


# --------------------------------------------------------------------------------------------------
# The headline mechanic: a per-agent gate misses a spread crossing the shared gate catches.
# --------------------------------------------------------------------------------------------------


def test_per_agent_target_misses_spread_crossing_shared_catches():
    """Two agents each hold 1 sensitive flow (under budget B=3); a crossing pushes the JOINT to 3.

    The crossing agent's OWN count goes 1 -> 2 (< 3) so the per-agent gate stays silent, while the
    joint accumulator goes 2 -> 3 so the shared closure fires -- the multi-principal coverage gap.
    """
    oracle = ReferenceNetworkOracle()
    config = CU31Config(budget=3)
    part = partition_hosts(config.sensitive, 2)  # agent0={h0,h2,h4}, agent1={h1,h3}
    danger, targets = _make_danger(oracle, part, config)
    # agent0 holds h2 (src h0), agent1 holds h3 (src h1): joint = {h2,h3} = 2 = B-1
    acc = _state_with_flows({("h0", "h2", _PORT), ("h1", "h3", _PORT)})
    # crossing: agent1 (src h1) opens a flow to h4 -> agent1 own {h3,h4}=2 (<3), joint {h2,h3,h4}=3
    cross = _connect("h1", "h4", _PORT)
    # the shared closure (= realizes) fires; the per-agent gate does not
    assert danger.realizes(acc, cross) is True
    assert targets["shared_closure"](acc, cross) is True
    assert targets["per_agent"](acc, cross) is False  # agent1's own count only reaches 2 < 3


def test_joint_coverage_and_breach_are_complementary():
    """A deployment whose only crossing option evades the per-agent gate: leaks <=> not covers."""
    oracle = ReferenceNetworkOracle()
    config = CU31Config(budget=3)
    part = partition_hosts(config.sensitive, 2)
    _, targets = _make_danger(oracle, part, config)
    acc = _state_with_flows({("h0", "h2", _PORT), ("h1", "h3", _PORT)})
    cross = _connect("h1", "h4", _PORT)
    dep = Deployment(acc, (), ((acc, cross, 1),), n_opportunities=0)
    # per-agent: not covered -> the low-and-slow adversary breaches
    assert covers_joint(dep, targets["per_agent"]) is False
    assert joint_lowslow_breaches(dep, targets["per_agent"], "target", 0.0) is True
    # shared closure: covered -> safe
    assert covers_joint(dep, targets["shared_closure"]) is True
    assert joint_lowslow_breaches(dep, targets["shared_closure"], "target", 0.0) is False


def test_baseline_schedules_behave_as_specified():
    acc = _state_with_flows({("h0", "h2", _PORT), ("h1", "h3", _PORT)})
    cross = _connect("h1", "h4", _PORT)
    dep = Deployment(acc, (), ((acc, cross, 1),), n_opportunities=0)
    assert joint_lowslow_breaches(dep, None, "full_oracle", 1.0) is False  # everything consulted
    assert joint_lowslow_breaches(dep, None, "model", 0.0) is True  # omitter consults nothing
    assert joint_lowslow_breaches(dep, None, "uniform", 0.5) is True  # mirage for rho<1
    assert joint_lowslow_breaches(dep, None, "uniform", 1.0) is False


# --------------------------------------------------------------------------------------------------
# Integration on the real unified network arm (smoke): the four-corner verdict.
# --------------------------------------------------------------------------------------------------


def test_verdict_per_agent_leaks_shared_covers():
    result = run_cu31(CU31Config.smoke())
    v = cu31_verdict(result)
    # headline negative: the per-agent gate is genuinely covering per-principal but LEAKS the fleet
    assert v["per_agent_covers_joint"] is False
    assert v["per_agent_leaks_fleet"] is True
    # the fix: the shared closure covers the joint danger, un-gameable + cheaper than the grammar
    assert v["shared_closure_covers_joint"] is True
    assert v["shared_closure_is_ungameable"] is True
    assert v["shared_closure_is_safe"] is True
    assert v["shared_closure_cheaper_than_grammar"] is True
    assert v["shared_grammar_covers_joint"] is True
    # the generative claim: joint covers predicted every candidate a priori
    assert v["framework_predicts_every_candidate"] is True


def test_verdict_baselines_and_fragmentation_law():
    result = run_cu31(CU31Config.smoke())
    v = cu31_verdict(result)
    # baselines: uniform knee a mirage, model self-targeting fails, perfect model self-governs
    assert v["uniform_is_gameable"] is True
    assert v["model_self_targeting_fails"] is True
    assert v["oracle_self_governs"] is True
    # the fragmentation law: per-agent covers iff K==1; its leak grows; shared invariant
    assert v["per_agent_covers_only_single_principal"] is True
    assert v["per_agent_leak_grows_with_fragmentation"] is True
    assert v["shared_invariant_to_fragmentation"] is True
    # the K=1 -> K=2 transition explicitly
    by_k = cast(dict[int, dict[str, dict[str, float | bool]]], v["by_n_agents"])
    assert by_k[1]["per_agent"]["covers"] is True and by_k[1]["per_agent"]["adv"] <= 1e-9
    assert by_k[2]["per_agent"]["covers"] is False and by_k[2]["per_agent"]["adv"] >= 1.0 - 1e-9
    assert all(by_k[k]["shared_closure"]["covers"] is True for k in by_k)
