"""Host run-record schema (SPEC-6 §9, HC3; the v0 §7.3 / SPEC-5 schema, composed).

Every host rollout emits one structured run-record: resolved config, seed, ε, the **composed**
per-step divergence trajectory *and* the **per-subsystem** trajectories (the §5.4 / §9.1
decomposition), and the consultation schedule actually used. Figures are produced *only* from these
records (the SPEC-2 §7.3 discipline) so every figure is reproducible from a record plus a plotting
script. The composed-host ``H_ε(ρ)`` curve (EH1, HC6) and the composition-law measurement (H13) are
read off these records; HC3 defines the schema before the loop (HC5) populates it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verisim.metrics.horizon import faithful_horizon


@dataclass
class HostRunRecord:
    """One host rollout: composed + per-subsystem divergence trajectories and the schedule."""

    config: dict[str, Any]
    seed: int
    epsilon: float
    divergences: list[float]  # composed per-step (SPEC-6 §9.1)
    subsystem_divergences: dict[str, list[float]] = field(default_factory=dict)
    consultation_schedule: list[bool] = field(default_factory=list)

    @property
    def faithful_horizon(self) -> int:
        """The composed faithful horizon ``H_ε`` -- the headline (SPEC-6 §9.3)."""
        return faithful_horizon(self.divergences, self.epsilon)

    @property
    def subsystem_horizons(self) -> dict[str, int]:
        """The per-subsystem component horizons ``H_ε^i`` (the §9.2 diagnostic input)."""
        return {
            sub: faithful_horizon(traj, self.epsilon)
            for sub, traj in self.subsystem_divergences.items()
        }

    @property
    def oracle_calls(self) -> int:
        return sum(self.consultation_schedule)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["faithful_horizon"] = self.faithful_horizon
        d["subsystem_horizons"] = self.subsystem_horizons
        d["oracle_calls"] = self.oracle_calls
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> HostRunRecord:
        d = json.loads(text)
        return HostRunRecord(
            config=d["config"],
            seed=d["seed"],
            epsilon=d["epsilon"],
            divergences=list(d["divergences"]),
            subsystem_divergences={
                k: list(v) for k, v in d.get("subsystem_divergences", {}).items()
            },
            consultation_schedule=list(d.get("consultation_schedule", [])),
        )


def write_host_records(records: Sequence[HostRunRecord], path: str | Path) -> Path:
    """Write host run-records as JSONL (one per line) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(record.to_json() for record in records) + "\n")
    return out


def read_host_records(path: str | Path) -> list[HostRunRecord]:
    """Read a JSONL host run-record file written by :func:`write_host_records`."""
    text = Path(path).read_text().strip()
    return [HostRunRecord.from_json(line) for line in text.splitlines() if line]


__all__ = ["HostRunRecord", "read_host_records", "write_host_records"]
