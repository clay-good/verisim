"""The SPEC-5 v0 network world: typed-graph state, action grammar, serialization (NW0)."""

from .action import NetAction, NetParseError, parse_net_action
from .config import DEFAULT_NET_CONFIG, NetConfig
from .serialize import from_canonical, to_canonical
from .state import (
    Flow,
    HostState,
    Link,
    NetworkState,
    can_reach,
    connected_hosts,
    link_key,
    reachability_matrix,
    services,
)

__all__ = [
    "DEFAULT_NET_CONFIG",
    "Flow",
    "HostState",
    "Link",
    "NetAction",
    "NetConfig",
    "NetParseError",
    "NetworkState",
    "can_reach",
    "connected_hosts",
    "from_canonical",
    "link_key",
    "parse_net_action",
    "reachability_matrix",
    "services",
    "to_canonical",
]
