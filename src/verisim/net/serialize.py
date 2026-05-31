"""Canonical serialization of network states (SPEC-5 §3.1).

One and only one serialized form per semantic state (sorted hosts/links/flows), so two
equal states serialize identically -- the prerequisite for the divergence metric and
exact-match scoring, exactly as v0 canonicalizes the filesystem. Round-trip is identity.
"""

from __future__ import annotations

from typing import Any

from .state import HostState, NetworkState, link_key


def to_canonical(state: NetworkState) -> dict[str, Any]:
    """Deterministic dict form: sorted hosts, links, and flows."""
    return {
        "hosts": {
            h: {
                "up": state.hosts[h].up,
                "services": list(state.hosts[h].services),
                "fw_deny": list(state.hosts[h].fw_deny),
            }
            for h in sorted(state.hosts)
        },
        "links": sorted([list(link) for link in state.links]),
        "flows": sorted([[s, d, p] for (s, d, p) in state.flows]),
        "clock": state.clock,
        "last_exit": state.last_exit,
    }


def from_canonical(d: dict[str, Any]) -> NetworkState:
    """Inverse of :func:`to_canonical`."""
    hosts = {
        h: HostState(
            up=bool(v["up"]),
            services=tuple(int(p) for p in v["services"]),
            fw_deny=tuple(str(s) for s in v["fw_deny"]),
        )
        for h, v in d["hosts"].items()
    }
    links = {link_key(a, b) for a, b in d["links"]}
    flows = {(str(s), str(dst), int(p)) for s, dst, p in d["flows"]}
    return NetworkState(
        hosts=hosts, links=links, flows=flows, clock=int(d["clock"]), last_exit=int(d["last_exit"])
    )
