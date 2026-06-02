"""Host workload vocabulary config (SPEC-6 §3.2, HC2).

The finite vocabulary the workload drivers (:mod:`verisim.hostdata.drivers`) draw their syscall
arguments from: the **path pool** for ``open``, the **content tokens** for ``write``, and the **uid
pool** for ``setuid`` (the privilege axis). The analogue of v0's
:class:`~verisim.env.config.EnvConfig` -- the host studies process/fd/credential *structure and
consequence*, so the argument space is fixed and small, not arbitrary-byte (SPEC-6 §2.3).

Paths are **top-level** (parent ``/``) so a ``write`` through a freshly ``open``-ed fd creates the
file on the embedded v0 filesystem (the v0 ``write`` is create-on-write when the parent dir exists;
the HC0 grammar has no ``mkdir`` yet). The §2.4 difficulty dial -- composition depth and the
interleaving-entropy / chaos knob -- arrives with the scheduler in a later HC increment; for now
difficulty is carried by the driver weighting, exactly as in v0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class HostConfig:
    """The finite workload vocabulary. Embedded (with its hash) in dataset manifests."""

    name: str = "host-v0"
    paths: tuple[str, ...] = ("/passwd", "/cfg", "/log", "/scratch", "/data")
    content_tokens: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "omega")
    uids: tuple[int, ...] = (0, 1000, 1001)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "paths": list(self.paths),
            "content_tokens": list(self.content_tokens),
            "uids": list(self.uids),
        }

    def config_hash(self) -> str:
        """Stable hash identifying this config; embedded in dataset manifests (SPEC-6 §3.1)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


DEFAULT_HOST_CONFIG = HostConfig()
