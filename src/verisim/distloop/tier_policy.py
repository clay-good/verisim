"""The tier policy ``π_w`` -- *which oracle tier to spend* on a consultation (SPEC-7 §8.2; DS5).

SPEC-7's central new axis. v0/net/host had only ``π_c`` (*when* to consult); the host added ``π_w``
as *which subsystem*; the distributed world generalizes ``π_w`` to *which **tier*** of the tiered
oracle (§5) -- because bit-exact truth is intractable, the loop must choose the price point paid on
each consult. The cheapest tier that still catches the model's error is the efficient choice,
and the H17 question (DS6) is whether spending cheap tiers buys more faithful horizon per
oracle-dollar than always paying for bit-exact truth.

DS5 ships the dependency-free baselines:

  - ``FixedTierPolicy(tier)`` -- always consult the same tier. ``bit_exact`` is the full-truth
    headline (directly comparable to every prior world's ``H_ε(ρ)``); the cheaper tiers are the H17
    comparison arms.
  - ``EscalatingTierPolicy`` -- consult the **cheapest tier that refutes** (the DD-D1 policy): try
    tiers cheapest-first and stop at the first that catches the error (or pay bit-exact to be sure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from verisim.distoracle.tiers import TIER_COSTS


@runtime_checkable
class TierPolicy(Protocol):
    """Chooses which oracle tier to consult at a given step (``escalate`` = cheapest-refutation)."""

    def tier(self, step: int) -> str: ...

    @property
    def escalate(self) -> bool: ...


@dataclass(frozen=True)
class FixedTierPolicy:
    """Always consult the same named tier (e.g. ``bit_exact`` for the full-truth headline)."""

    name: str = "bit_exact"

    def __post_init__(self) -> None:
        if self.name not in TIER_COSTS:
            raise ValueError(f"unknown tier {self.name!r}; choose from {sorted(TIER_COSTS)}")

    def tier(self, step: int) -> str:
        return self.name

    @property
    def escalate(self) -> bool:
        return False


@dataclass(frozen=True)
class EscalatingTierPolicy:
    """Consult the **cheapest tier that refutes** (DD-D1): cheapest-first, stop at first catch."""

    def tier(self, step: int) -> str:
        return "metamorphic"  # the starting tier; the runner escalates from here when escalate=True

    @property
    def escalate(self) -> bool:
        return True
