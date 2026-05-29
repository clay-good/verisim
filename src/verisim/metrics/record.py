"""Run-record schema (SPEC-2 §7.3).

Every rollout emits one structured run-record: full resolved config, seed, the
per-step divergence trajectory, the consultation schedule actually used, and the
resulting ``H_ε``. Figures are produced *only* from these records (SPEC-2 §7.3,
§12) so every figure is reproducible from a record plus a plotting script. The
propose-verify-correct loop (M5) is what populates these; M3 defines the schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .horizon import faithful_horizon


@dataclass
class RunRecord:
    config: dict[str, Any]
    seed: int
    epsilon: float
    divergences: list[float]
    consultation_schedule: list[bool] = field(default_factory=list)

    @property
    def faithful_horizon(self) -> int:
        return faithful_horizon(self.divergences, self.epsilon)

    @property
    def oracle_calls(self) -> int:
        return sum(self.consultation_schedule)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["faithful_horizon"] = self.faithful_horizon
        d["oracle_calls"] = self.oracle_calls
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> RunRecord:
        d = json.loads(text)
        return RunRecord(
            config=d["config"],
            seed=d["seed"],
            epsilon=d["epsilon"],
            divergences=list(d["divergences"]),
            consultation_schedule=list(d.get("consultation_schedule", [])),
        )
