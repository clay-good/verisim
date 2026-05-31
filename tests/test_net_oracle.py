"""NW0 semantics: reachability, firewall, async flow drop, purity/determinism (SPEC-5 §5)."""

import pytest

from verisim.net import (
    NetParseError,
    NetworkState,
    can_reach,
    parse_net_action,
    reachability_matrix,
)
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netoracle import ReferenceNetworkOracle

ORACLE = ReferenceNetworkOracle()
HOSTS = DEFAULT_NET_CONFIG.hosts


def run(cmds: list[str]) -> NetworkState:
    """Step a fresh network through a command script; return the final state."""
    state = NetworkState.initial(HOSTS)
    for cmd in cmds:
        state = ORACLE.step(state, parse_net_action(cmd)).state
    return state


def last(cmds: list[str]) -> int:
    state = NetworkState.initial(HOSTS)
    exit_code = 0
    for cmd in cmds:
        result = ORACLE.step(state, parse_net_action(cmd))
        state, exit_code = result.state, result.exit_code
    return exit_code


def test_connect_succeeds_over_a_path():
    # h0 -- h1 -- h2, service on h2:80 -> h0 can reach it.
    state = run(["link_up h0 h1", "link_up h1 h2", "svc_up h2 80", "connect h0 h2 80"])
    assert ("h0", "h2", 80) in state.flows
    assert can_reach(state, "h0", "h2", 80)


def test_connect_refused_without_path():
    # service exists but no link path -> unreachable, connection refused (exit 1), no flow.
    assert last(["svc_up h2 80", "connect h0 h2 80"]) == 1
    state = run(["svc_up h2 80", "connect h0 h2 80"])
    assert ("h0", "h2", 80) not in state.flows


def test_connect_refused_without_service():
    assert last(["link_up h0 h1", "connect h0 h1 80"]) == 1


def test_firewall_blocks_reachable_service():
    base = ["link_up h0 h1", "svc_up h1 80"]
    assert last([*base, "connect h0 h1 80"]) == 0  # allowed
    assert last([*base, "fw_deny h1 h0", "connect h0 h1 80"]) == 1  # blocked
    assert last([*base, "fw_deny h1 h0", "fw_allow h1 h0", "connect h0 h1 80"]) == 0  # unblocked


def test_host_down_breaks_reachability():
    assert last(["link_up h0 h1", "svc_up h1 80", "host_down h1", "connect h0 h1 80"]) == 1


def test_advance_drops_stale_flow():
    # establish a flow, break the path, then advance -> the flow is dropped (delayed effect).
    cmds = ["link_up h0 h1", "svc_up h1 80", "connect h0 h1 80", "link_down h0 h1"]
    before = run(cmds)
    assert ("h0", "h1", 80) in before.flows  # not dropped immediately
    after = run([*cmds, "advance"])
    assert ("h0", "h1", 80) not in after.flows  # dropped on advance
    assert after.clock == 1


def test_invalid_args_exit_2():
    assert last(["link_up h0 h0"]) == 2  # self-link
    assert last(["connect h0 zzz 80"]) == 2  # unknown host


def test_oracle_is_pure_and_deterministic():
    state = run(["link_up h0 h1", "svc_up h1 80"])
    snapshot = (dict(state.hosts), set(state.links), set(state.flows))
    action = parse_net_action("connect h0 h1 80")
    r1 = ORACLE.step(state, action)
    r2 = ORACLE.step(state, action)
    # same input -> same output...
    assert r1.delta == r2.delta and r1.exit_code == r2.exit_code
    # ...and the input state is not mutated.
    assert (dict(state.hosts), set(state.links), set(state.flows)) == snapshot


def test_idempotent_config_ops_are_noops():
    state = run(["host_up h0"])  # already up
    assert state == NetworkState.initial(HOSTS).copy() or state.last_exit == 0


def test_reachability_matrix_shape():
    state = run(["link_up h0 h1", "svc_up h1 80", "svc_up h1 443"])
    matrix = reachability_matrix(state)
    # one entry per (src host) x (listening service); h0->h1:80 reachable.
    assert matrix[("h0", "h1", 80)] is True
    assert ("h0", "h1", 80) in matrix


def test_parser_rejects_garbage():
    with pytest.raises(NetParseError):
        parse_net_action("frobnicate h0")
