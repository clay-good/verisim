"""Subsystem policies ``π_w`` -- *which* subsystem's truth to buy (SPEC-6 §8.2). The new axis.

Given that the consultation policy ``π_c`` (when, §8.1; reused from v0's
:mod:`verisim.loop.policy`) has decided to spend a cheap probe, the subsystem policy decides *which*
subsystem to verify -- the process table, the fd table, the embedded filesystem, or the global
last-exit. This is active oracle-selection: it generalizes NW5's probe policy ``π_o`` (which *host*)
to *which truth-source to buy*, and it is what makes H13's per-subsystem budgeting measurable
(§8.2). It has no v0 analogue.

  - ``FixedSubsystem`` -- always verify one named subsystem (the single-subsystem ablation arm).
  - ``RoundRobinSubsystem`` -- cycle through the subsystems deterministically (uniform coverage).
  - ``UncertaintySubsystem`` -- the **information-gain** policy (§8.2): spend the consult on the
    subsystem whose predicted delta the model is *least certain* about, read from the factored arm's
    per-subsystem decode entropy (§5.4). Falls back to round-robin when no uncertainty is supplied
    (a baseline model exposes none), so it degrades gracefully.

``select`` takes an optional ``uncertainty`` map (subsystem -> the model's per-subsystem
uncertainty for this step); the static baselines ignore it, the smart policy reads it. The runner
supplies it only when the proposer exposes one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from verisim.host.state import HostState
from verisim.hostmetrics.divergence import SUBSYSTEMS


@runtime_checkable
class SubsystemPolicy(Protocol):
    def select(
        self, belief: HostState, uncertainty: Mapping[str, float] | None = None
    ) -> str: ...


@dataclass
class FixedSubsystem:
    """Always verify ``subsystem`` (the single-subsystem ablation arm, §9.2)."""

    subsystem: str

    def __post_init__(self) -> None:
        if self.subsystem not in SUBSYSTEMS:
            raise ValueError(f"unknown subsystem {self.subsystem!r}; choose from {SUBSYSTEMS}")

    def select(self, belief: HostState, uncertainty: Mapping[str, float] | None = None) -> str:
        return self.subsystem


@dataclass
class RoundRobinSubsystem:
    """Cycle through ``subsystems`` deterministically -- uniform coverage, no randomness."""

    subsystems: Sequence[str] = SUBSYSTEMS
    _i: int = field(default=0)

    def select(self, belief: HostState, uncertainty: Mapping[str, float] | None = None) -> str:
        subsystem = self.subsystems[self._i % len(self.subsystems)]
        self._i += 1
        return subsystem


@dataclass
class UncertaintySubsystem:
    """Verify the subsystem the model is **least certain** about (the §8.2 information-gain choice).

    Reads the per-subsystem decode entropy the factored arm exposes and picks the argmax (ties
    broken by canonical subsystem order, so it is deterministic). When ``uncertainty`` is ``None``
    (the proposer exposes none, or all subsystems tie at 0) it falls back to a round-robin sweep, so
    the policy is always well-defined.
    """

    subsystems: Sequence[str] = SUBSYSTEMS
    _i: int = field(default=0)

    def select(self, belief: HostState, uncertainty: Mapping[str, float] | None = None) -> str:
        if uncertainty:
            # max() returns the first argmax on ties, so equal-uncertainty subsystems break in
            # canonical order -- deterministic. A positive max means real information to act on.
            best = max(self.subsystems, key=lambda s: uncertainty.get(s, 0.0))
            if uncertainty.get(best, 0.0) > 0.0:
                return best
        subsystem = self.subsystems[self._i % len(self.subsystems)]
        self._i += 1
        return subsystem
