"""Network environment vocabulary config (SPEC-5 §3).

Fixes the finite vocabularies the drivers draw from -- host ids and service ports --
so the action/state space is finite and learnable, exactly as v0's :class:`EnvConfig`
fixes content/mode/name pools. The §3.4 difficulty dials (host/link count, topology
family) live on the driver, not here, mirroring v0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class NetConfig:
    name: str = "net-v0"
    hosts: tuple[str, ...] = ("h0", "h1", "h2", "h3", "h4")
    ports: tuple[int, ...] = (22, 80, 443)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "hosts": list(self.hosts), "ports": list(self.ports)}

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


DEFAULT_NET_CONFIG = NetConfig()


# Canonical service-port pool the world factory draws from, in priority order (the
# defaults first). Extended deterministically if more ports are requested than listed.
_PORT_POOL: tuple[int, ...] = (22, 80, 443, 53, 8080, 3306, 5432, 6379, 9200, 25, 110, 143)


def scaled_net_config(n_hosts: int, n_ports: int = 3, *, name: str | None = None) -> NetConfig:
    """A :class:`NetConfig` of ``n_hosts`` hosts and ``n_ports`` ports (SPEC-8 §7.1 scale axis).

    Hosts are ``h0…h{n-1}``; ports are drawn from :data:`_PORT_POOL` (the defaults first), extended
    with synthetic ``10000+k`` ports if more are requested than the pool lists. The world-size lever
    the OG5/OG6 scale-up sweeps over - every downstream component (drivers, oracle, graph
    featurization, vocab) already keys off ``config.hosts`` / ``config.ports``, so nothing changes.
    """
    if n_hosts < 2:
        raise ValueError(f"n_hosts must be >= 2 (need two hosts for a link), got {n_hosts}")
    if n_ports < 1:
        raise ValueError(f"n_ports must be >= 1, got {n_ports}")
    hosts = tuple(f"h{i}" for i in range(n_hosts))
    if n_ports <= len(_PORT_POOL):
        ports = _PORT_POOL[:n_ports]
    else:
        extra = tuple(10000 + k for k in range(n_ports - len(_PORT_POOL)))
        ports = _PORT_POOL + extra
    return NetConfig(name=name or f"net-{n_hosts}h{n_ports}p", hosts=hosts, ports=ports)
