"""Tests for the torch-free graph featurization (SPEC-5 §6.1, the NW8 deterministic core).

These pin the exact information the message-passing/RSSM graph arm sees, with no torch and no
GPU -- the NW0-NW3 discipline applied to the graph arm: prove the deterministic featurization
before any learned weight touches it.
"""

from __future__ import annotations

import math

from verisim.net.action import parse_net_action
from verisim.net.config import NetConfig
from verisim.net.state import HostState, NetworkState, link_key
from verisim.netmodel.graph import (
    ACTION_NAMES,
    build_graph,
    feature_dims,
    symlog,
)

CFG = NetConfig()  # hosts h0..h4, ports 22/80/443


def _state() -> NetworkState:
    return NetworkState(
        hosts={
            "h0": HostState(up=True, services=(22, 80), fw_deny=("h3",)),
            "h1": HostState(up=False, services=(), fw_deny=()),
            "h2": HostState(up=True, services=(443,), fw_deny=("h0", "h1")),
            "h3": HostState(up=True, services=(), fw_deny=()),
            "h4": HostState(up=True, services=(22,), fw_deny=()),
        },
        links={link_key("h0", "h2"), link_key("h2", "h4")},
        flows={("h0", "h2", 443)},
        clock=7,
        last_exit=1,
    )


def test_determinism() -> None:
    a = parse_net_action("svc_up h1 80")
    g1 = build_graph(_state(), a, CFG)
    g2 = build_graph(_state(), a, CFG)
    assert g1 == g2


def test_node_count_and_order() -> None:
    g = build_graph(_state(), None, CFG)
    assert g.host_ids == CFG.hosts
    assert g.host_index == {"h0": 0, "h1": 1, "h2": 2, "h3": 3, "h4": 4}
    assert len(g.node_features) == len(CFG.hosts)
    assert all(len(f) == g.dims.node for f in g.node_features)


def test_up_flag() -> None:
    g = build_graph(_state(), None, CFG)
    assert g.node_features[0][0] == 1.0  # h0 up
    assert g.node_features[1][0] == 0.0  # h1 down


def test_service_indicators() -> None:
    g = build_graph(_state(), None, CFG)
    # layout: [up, svc(22), svc(80), svc(443), fw_deny x5, role x3]
    h0 = g.node_features[0]
    assert (h0[1], h0[2], h0[3]) == (1.0, 1.0, 0.0)  # h0 listens 22,80 not 443
    h2 = g.node_features[2]
    assert (h2[1], h2[2], h2[3]) == (0.0, 0.0, 1.0)  # h2 listens 443


def test_fw_deny_indicators() -> None:
    g = build_graph(_state(), None, CFG)
    base = 1 + len(CFG.ports)  # start of the fw-deny block
    # h2 denies h0 and h1 (sources indexed in host order h0..h4)
    h2 = g.node_features[2]
    deny = h2[base : base + len(CFG.hosts)]
    assert deny == (1.0, 1.0, 0.0, 0.0, 0.0)
    # h0 denies h3
    h0 = g.node_features[0]
    assert h0[base : base + len(CFG.hosts)] == (0.0, 0.0, 0.0, 1.0, 0.0)


def test_link_edges_canonical_and_up_only() -> None:
    g = build_graph(_state(), None, CFG)
    # links h0-h2 and h2-h4 -> index pairs (0,2),(2,4), sorted, each once.
    assert g.link_edges == ((0, 2), (2, 4))


def test_flow_edges_directed_with_port() -> None:
    g = build_graph(_state(), None, CFG)
    # flow (h0->h2, 443): src 0, dst 2, port index 2 (443 is ports[2])
    assert g.flow_edges == ((0, 2, 2),)


def test_action_arg_roles() -> None:
    # link_up h1 h3 -> h1 is arg0, h3 is arg1
    g = build_graph(_state(), parse_net_action("link_up h1 h3"), CFG)
    role_start = g.dims.node - 3
    h1 = g.node_features[1]
    h3 = g.node_features[3]
    assert h1[role_start:] == (1.0, 0.0, 0.0)  # arg position 0
    assert h3[role_start:] == (0.0, 1.0, 0.0)  # arg position 1
    # an untouched host has no role
    assert g.node_features[4][role_start:] == (0.0, 0.0, 0.0)


def test_connect_arg_roles_three_positions() -> None:
    # connect h0 h2 80 -> h0 arg0, h2 arg1 (80 is the port, not a host arg-role)
    g = build_graph(_state(), parse_net_action("connect h0 h2 80"), CFG)
    role_start = g.dims.node - 3
    assert g.node_features[0][role_start:] == (1.0, 0.0, 0.0)
    assert g.node_features[2][role_start:] == (0.0, 1.0, 0.0)


def test_graph_features_action_onehot_and_port_and_clock_exit() -> None:
    g = build_graph(_state(), parse_net_action("svc_up h1 80"), CFG)
    gf = g.graph_features
    n_act = len(ACTION_NAMES)
    # action-type one-hot: svc_up is index 4
    assert gf[: n_act].index(1.0) == ACTION_NAMES.index("svc_up")
    assert sum(gf[:n_act]) == 1.0
    # port one-hot: 80 is ports[1]
    port_block = gf[n_act : n_act + len(CFG.ports)]
    assert port_block == (0.0, 1.0, 0.0)
    # clock symlog
    assert math.isclose(gf[n_act + len(CFG.ports)], symlog(7.0))
    # last-exit one-hot for exit=1
    exit_block = gf[n_act + len(CFG.ports) + 1 :]
    assert exit_block == (0.0, 1.0, 0.0)


def test_action_none_zeros_action_blocks() -> None:
    g = build_graph(_state(), None, CFG)
    n_act = len(ACTION_NAMES)
    assert sum(g.graph_features[: n_act + len(CFG.ports)]) == 0.0  # no action -> no onehot/port
    role_start = g.dims.node - 3
    assert all(f[role_start:] == (0.0, 0.0, 0.0) for f in g.node_features)


def test_advance_has_no_host_roles_or_port() -> None:
    g = build_graph(_state(), parse_net_action("advance"), CFG)
    n_act = len(ACTION_NAMES)
    assert g.graph_features[:n_act].index(1.0) == ACTION_NAMES.index("advance")
    assert g.graph_features[n_act : n_act + len(CFG.ports)] == (0.0, 0.0, 0.0)
    role_start = g.dims.node - 3
    assert all(f[role_start:] == (0.0, 0.0, 0.0) for f in g.node_features)


def test_feature_dims_match() -> None:
    dims = feature_dims(CFG)
    assert dims.node == 1 + 3 + 5 + 3  # up + ports + fw-deny-hosts + roles
    assert dims.graph == len(ACTION_NAMES) + 3 + 1 + 3
    g = build_graph(_state(), None, CFG)
    assert g.dims == dims


def test_symlog_monotone_and_zero() -> None:
    assert symlog(0.0) == 0.0
    assert symlog(1.0) > 0 and symlog(-1.0) < 0
    assert symlog(1000.0) > symlog(10.0)
    # compresses: a 100x input is far less than 100x output
    assert symlog(1000.0) < 10 * symlog(10.0)
