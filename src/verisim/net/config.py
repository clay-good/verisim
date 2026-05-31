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
