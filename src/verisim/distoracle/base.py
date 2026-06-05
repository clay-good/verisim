"""The distributed oracle protocol + step result (SPEC-7 §5, DS0 increment 1).

DS0 increment 1 ships **Tier-A** (the from-scratch deterministic reference DES,
:class:`~verisim.distoracle.reference.ReferenceDistOracle`). The cheap tiers (consistency-cycle,
metamorphic) and Tier-B (a wrapped real DST runtime) are later DS increments; the *tiered* oracle is
SPEC-7's payload (§5) but the deterministic core ships and is fully tested first, exactly as v0/net/
host did. The step result mirrors every prior world: the true next state, the **delta** that
produced it (``apply(state, delta) == state`` by construction -- the M1-analogue invariant), and the
client-visible result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from verisim.dist.action import DistAction
from verisim.dist.delta import DistDelta
from verisim.dist.state import DistributedState


@dataclass(frozen=True)
class DistStepResult:
    """One distributed transition: the true next state, the delta that produced it, and the
    client-visible ``(status, value)`` result. ``apply(state, delta) == state`` holds by
    construction (the M1-analogue invariant, DS1)."""

    state: DistributedState
    delta: DistDelta
    status: str
    value: str


class DistOracle(Protocol):
    """A deterministic interpreter of the distributed action grammar over the cluster state."""

    def step(self, state: DistributedState, action: DistAction) -> DistStepResult: ...
