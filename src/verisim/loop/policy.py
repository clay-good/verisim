"""Consultation policies ``π_c`` (SPEC-2 §6.1).

A policy decides, per step, whether to *propose* spending an oracle consultation.
The runner enforces the budget ``ρ`` on top of the policy's proposals, so "budget
never exceeded" holds regardless of the policy (SPEC-2 §16 loop invariant).

v0/M5 ships the ``fixed`` (every-``k``-steps) policy only -- the dumb baseline of
RQ2. The drift- and uncertainty-triggered policies (the interesting half of H2)
are M7, once a calibrated uncertainty signal exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class ConsultationPolicy(Protocol):
    def should_consult(self, step: int) -> bool: ...


@dataclass(frozen=True)
class FixedInterval:
    """Consult every ``k`` steps (``k = 1`` is every step). The §6.1 baseline."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"FixedInterval k must be >= 1, got {self.k}")

    def should_consult(self, step: int) -> bool:
        return step % self.k == 0


@dataclass(frozen=True)
class Never:
    """Never consult -- the ``ρ = 0`` pure-model policy (baseline b0)."""

    def should_consult(self, step: int) -> bool:
        return False


def fixed_interval_for_rho(rho: float) -> ConsultationPolicy:
    """A ``fixed`` policy whose natural consultation rate is ``rho``.

    ``rho = 0`` -> never consult; ``rho = 1`` -> every step; otherwise consult
    every ``round(1/rho)`` steps. The runner still caps total calls at the budget
    ``floor(rho * T)`` (see :func:`verisim.loop.runner.budget_for_rho`), so the
    policy's rate and the hard budget agree.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if rho == 0.0:
        return Never()
    return FixedInterval(k=max(1, round(1.0 / rho)))
