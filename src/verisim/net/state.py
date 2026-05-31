"""Network state `S` for the SPEC-5 v0 reachability/connectivity world (SPEC-5 §3.1).

The smallest *network* world with the hard property (compounding state, long-range
dependencies, combinatorial reachability) and a free, deterministic oracle: a graph of
hosts joined by links, each host running services behind a simple firewall, with
established flows between them. This is to SPEC-5 what the shell/filesystem ``State`` is
to SPEC-2 -- deliberately minimal (open question SPEC-5 §17.1), but unmistakably a network.

What is modeled (the pinned subset):
  - **hosts** -- up/down, a set of listening service ports, and a firewall block-list of
    source hosts whose incoming traffic is denied;
  - **links** -- undirected host-host edges, up/down (the topology);
  - **flows** -- established connections ``(src, dst, port)``;
  - a virtual **clock** and the **last** action's result (observation).

The operationally meaningful quantity (SPEC-5 §3.1, §9.2) is the **reachability matrix**
``R[src][(dst, port)]`` -- can ``src`` reach the service ``(dst, port)``? -- which is a pure
function of the state and is what a defender/SRE actually relies on. It is computed here
(not stored) and is the basis of reachability-faithfulness.

State is immutable by convention -- the oracle and ``apply`` return a fresh ``NetworkState``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HostState:
    """One host. ``services``/``fw_deny`` are kept sorted so equality is canonical."""

    up: bool = True
    services: tuple[int, ...] = ()  # sorted listening ports
    fw_deny: tuple[str, ...] = ()  # sorted source host ids whose incoming traffic is blocked

    def with_service(self, port: int, listening: bool) -> HostState:
        ports = set(self.services)
        ports.add(port) if listening else ports.discard(port)
        return HostState(self.up, tuple(sorted(ports)), self.fw_deny)

    def with_fw(self, src: str, deny: bool) -> HostState:
        blocked = set(self.fw_deny)
        blocked.add(src) if deny else blocked.discard(src)
        return HostState(self.up, self.services, tuple(sorted(blocked)))

    def with_up(self, up: bool) -> HostState:
        return HostState(up, self.services, self.fw_deny)


Link = tuple[str, str]  # canonical: sorted (a, b) with a < b
Flow = tuple[str, str, int]  # (src, dst, port)


def link_key(a: str, b: str) -> Link:
    """Canonical undirected link key (sorted), so (a,b) and (b,a) are the same link."""
    return (a, b) if a <= b else (b, a)


@dataclass
class NetworkState:
    """A typed network graph. Canonicalizable, serializable, immutable by convention."""

    hosts: dict[str, HostState]
    links: set[Link] = field(default_factory=set)
    flows: set[Flow] = field(default_factory=set)
    clock: int = 0
    last_exit: int = 0

    @staticmethod
    def initial(host_ids: tuple[str, ...]) -> NetworkState:
        """All hosts up, no links, no services, no flows -- the empty network."""
        return NetworkState(hosts={h: HostState() for h in host_ids})

    def copy(self) -> NetworkState:
        return NetworkState(
            hosts=dict(self.hosts),
            links=set(self.links),
            flows=set(self.flows),
            clock=self.clock,
            last_exit=self.last_exit,
        )


# --- derived: reachability (pure functions of state) -------------------------


def connected_hosts(state: NetworkState, src: str) -> set[str]:
    """Hosts reachable from ``src`` over **up** links between **up** hosts (BFS)."""
    if src not in state.hosts or not state.hosts[src].up:
        return set()
    adj: dict[str, list[str]] = {h: [] for h in state.hosts}
    for a, b in state.links:
        if state.hosts.get(a) and state.hosts.get(b) and state.hosts[a].up and state.hosts[b].up:
            adj[a].append(b)
            adj[b].append(a)
    seen = {src}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def can_reach(state: NetworkState, src: str, dst: str, port: int) -> bool:
    """Can ``src`` reach service ``(dst, port)``? Path + host-up + service + firewall."""
    if src == dst:
        # loopback to a local service is allowed iff the host is up and listening.
        return src in state.hosts and state.hosts[src].up and port in state.hosts[src].services
    dst_host = state.hosts.get(dst)
    if dst_host is None or not dst_host.up or port not in dst_host.services:
        return False
    if src in dst_host.fw_deny:
        return False
    return dst in connected_hosts(state, src)


def services(state: NetworkState) -> list[tuple[str, int]]:
    """All listening ``(host, port)`` services, sorted (the columns of ``R``)."""
    out = [(h, p) for h, hs in state.hosts.items() for p in hs.services]
    return sorted(out)


def reachability_matrix(state: NetworkState) -> dict[tuple[str, str, int], bool]:
    """``R[(src, dst, port)]`` over all src hosts x listening services (SPEC-5 §3.1)."""
    svc = services(state)
    return {
        (src, dst, port): can_reach(state, src, dst, port)
        for src in sorted(state.hosts)
        for (dst, port) in svc
    }
