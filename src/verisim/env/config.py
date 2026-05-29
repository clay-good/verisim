"""Environment vocabulary config (SPEC-2 §2.2, §4).

v0 fixes content, mode, env-key, and name vocabularies so the action/content
space is finite and learnable -- v0 studies *structure and consequence*, not
arbitrary byte modeling (SPEC-2 §17.1). The drivers (``data/drivers.py``) draw
their arguments from these pools.

The full §2.4 "difficulty dial" (max tree depth/breadth as explicit knobs) is
deferred to M6, where it is tuned empirically so the pure-neural rollout drifts
within tested horizons. v0's difficulty is carried by the driver weighting; we
do not add unused config fields ahead of that need.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    name: str = "v0"
    content_tokens: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "omega")
    modes: tuple[int, ...] = (0o644, 0o600, 0o755, 0o700)
    env_keys: tuple[str, ...] = ("PATH", "HOME", "LANG", "TERM")
    name_pool: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "content_tokens": list(self.content_tokens),
            "modes": list(self.modes),
            "env_keys": list(self.env_keys),
            "name_pool": list(self.name_pool),
        }

    def config_hash(self) -> str:
        """Stable hash identifying this config; embedded in dataset manifests."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


DEFAULT_CONFIG = EnvConfig()
