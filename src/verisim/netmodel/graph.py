"""Torch-free featurization of a ``NetworkState`` into a typed graph (SPEC-5 §6.1).

The message-passing/RSSM graph arm of ``M_θ`` (the NW8 arm, SPEC-5 §6.1-6.2) operates on
the network *as a graph* rather than on its serialized token stream (the flat NW4 arm). This
module is the deterministic, dependency-free bridge between the two: it turns a
:class:`~verisim.net.state.NetworkState` and the conditioning
:class:`~verisim.net.action.NetAction` into a :class:`NetGraph` of plain Python numbers --
host nodes with feature vectors, link edges, flow edges, and graph-level action/clock
features -- which the torch GNN then consumes.

Keeping the featurization torch-free and here (not inside the model) follows the repo's
NW0-NW3 discipline: the deterministic core ships and is tested with **no GPU and no torch**
before any learned weights touch it (SPEC-5 §13). It also fixes, in one canonical place, the
exact information the graph arm sees -- which is the **same closed world** the flat arm
serializes (:class:`~verisim.net.config.NetConfig`'s finite host/port pools), so the
graph-vs-flat comparison (H11, EN4) is fair: neither arm is handed information the other lacks.

Determinism: host nodes are indexed in the config's canonical host order; every feature is a
pure function of ``(state, action, config)``. ``symlog`` (DreamerV3, SPEC-5 §2.4 / §6.1)
compresses the one unbounded quantity (the clock) so a fixed-hyperparameter model handles
scales spanning orders of magnitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from verisim.net.action import NetAction
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState, link_key

# The §3.2 command grammar, in a fixed canonical order. The action-type one-hot indexes here.
ACTION_NAMES: tuple[str, ...] = (
    "host_up", "host_down", "link_up", "link_down", "svc_up", "svc_down",
    "fw_deny", "fw_allow", "connect", "close", "advance",
)
_ACTION_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}


def symlog(x: float) -> float:
    """Symmetric log (DreamerV3): ``sign(x)*log(1+|x|)``. Identity-ish near 0, log-ish far."""
    return math.copysign(math.log1p(abs(x)), x)


@dataclass(frozen=True)
class FeatureDims:
    """The exact widths of each feature block, so the torch model can size itself (§6.1)."""

    n_hosts: int
    n_ports: int

    @property
    def node(self) -> int:
        # up(1) + service-per-port(P) + fw-deny-per-source-host(H) + action-arg-role(3)
        return 1 + self.n_ports + self.n_hosts + 3

    @property
    def graph(self) -> int:
        # action-type one-hot(len ACTION_NAMES) + action-port one-hot(P) + clock symlog(1)
        #   + last-exit one-hot(3)
        return len(ACTION_NAMES) + self.n_ports + 1 + 3


def feature_dims(config: NetConfig) -> FeatureDims:
    return FeatureDims(n_hosts=len(config.hosts), n_ports=len(config.ports))


@dataclass(frozen=True)
class NetGraph:
    """A network state featurized as a graph (SPEC-5 §6.1). All values are plain floats/ints.

    - ``host_ids`` / ``host_index``: the canonical node order (config host order).
    - ``node_features[i]``: feature vector for host ``host_ids[i]`` (width ``dims.node``).
    - ``link_edges``: undirected up-links as sorted ``(i, j)`` index pairs (``i < j``); the GNN
      symmetrizes. A link is present iff it is up (down links are absent from the state set).
    - ``flow_edges``: established flows as ``(src_i, dst_i, port_index)`` directed triples.
    - ``graph_features``: action + clock + last-exit conditioning (width ``dims.graph``).
    - ``dims``: the block widths, for model sizing.
    """

    host_ids: tuple[str, ...]
    host_index: dict[str, int]
    node_features: tuple[tuple[float, ...], ...]
    link_edges: tuple[tuple[int, int], ...]
    flow_edges: tuple[tuple[int, int, int], ...]
    graph_features: tuple[float, ...]
    dims: FeatureDims


def build_graph(
    state: NetworkState, action: NetAction | None, config: NetConfig
) -> NetGraph:
    """Featurize ``(state, action)`` into a :class:`NetGraph` over ``config``'s closed world.

    ``action`` may be ``None`` (e.g. for encoding a state with no pending command); then the
    action-type one-hot, port one-hot, and per-node arg-role indicators are all zero.
    """
    dims = feature_dims(config)
    hosts = config.hosts
    host_index = {h: i for i, h in enumerate(hosts)}
    port_index = {p: i for i, p in enumerate(config.ports)}

    # Which host(s) the action references, by positional arg role (arg0/arg1/arg2).
    arg_role: dict[str, set[int]] = {h: set() for h in hosts}
    action_port_idx: int | None = None
    if action is not None:
        for pos, arg in enumerate(action.args[:3]):
            if arg in arg_role:  # a host id (ports are also args but not host ids)
                arg_role[arg].add(pos)
        # svc_*/connect/close carry a port in the last positional slot.
        if action.name in ("svc_up", "svc_down", "connect", "close"):
            try:
                action_port_idx = port_index.get(action.port)
            except (ValueError, IndexError):
                action_port_idx = None

    # --- per-host node features -------------------------------------------------
    node_features: list[tuple[float, ...]] = []
    for h in hosts:
        hs = state.hosts.get(h)
        feats: list[float] = []
        feats.append(1.0 if (hs is not None and hs.up) else 0.0)
        listening = set(hs.services) if hs is not None else set()
        feats.extend(1.0 if p in listening else 0.0 for p in config.ports)
        denied = set(hs.fw_deny) if hs is not None else set()
        feats.extend(1.0 if src in denied else 0.0 for src in hosts)
        roles = arg_role[h]
        feats.extend(1.0 if r in roles else 0.0 for r in (0, 1, 2))
        assert len(feats) == dims.node
        node_features.append(tuple(feats))

    # --- link edges (undirected, present iff up) --------------------------------
    link_edges = tuple(
        sorted(
            (host_index[a], host_index[b])
            for (a, b) in (link_key(x, y) for (x, y) in state.links)
            if a in host_index and b in host_index
        )
    )

    # --- flow edges (directed src->dst with port) -------------------------------
    flow_edges = tuple(
        sorted(
            (host_index[s], host_index[d], port_index[p])
            for (s, d, p) in state.flows
            if s in host_index and d in host_index and p in port_index
        )
    )

    # --- graph-level action / clock / last-exit features ------------------------
    gfeats: list[float] = [0.0] * len(ACTION_NAMES)
    if action is not None and action.name in _ACTION_INDEX:
        gfeats[_ACTION_INDEX[action.name]] = 1.0
    port_onehot = [0.0] * dims.n_ports
    if action_port_idx is not None:
        port_onehot[action_port_idx] = 1.0
    gfeats.extend(port_onehot)
    gfeats.append(symlog(float(state.clock)))
    exit_onehot = [0.0, 0.0, 0.0]
    if 0 <= state.last_exit <= 2:
        exit_onehot[state.last_exit] = 1.0
    gfeats.extend(exit_onehot)
    assert len(gfeats) == dims.graph

    return NetGraph(
        host_ids=hosts,
        host_index=host_index,
        node_features=tuple(node_features),
        link_edges=link_edges,
        flow_edges=flow_edges,
        graph_features=tuple(gfeats),
        dims=dims,
    )
