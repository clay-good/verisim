"""Subsystem policies ``π_w`` -- *which* subsystem's truth to buy (SPEC-6 §8.2). The new axis.

Given that the consultation policy ``π_c`` (when, §8.1; reused from v0's
:mod:`verisim.loop.policy`) has decided to spend a cheap probe, the subsystem policy decides *which*
subsystem to verify -- the process table, the fd table, the embedded filesystem, or the global
last-exit. This is active oracle-selection: it generalizes NW5's probe policy ``π_o`` (which *host*)
to *which truth-source to buy*, and it is what makes H13's per-subsystem budgeting measurable
(§8.2). It has no v0 analogue.

HC5 ships the dependency-free baselines and the protocol; the uncertainty- /
information-gain-targeted policy that beats them -- spend on the subsystem whose predicted delta is
least certain and most consequential -- needs a model that localizes its per-subsystem uncertainty
(§5.4) and lands with the
factored arm + EH3 (a later increment):

  - ``FixedSubsystem`` -- always verify one named subsystem (the single-subsystem ablation arm).
  - ``RoundRobinSubsystem`` -- cycle through the subsystems deterministically (uniform coverage).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from verisim.host.state import HostState
from verisim.hostmetrics.divergence import SUBSYSTEMS


@runtime_checkable
class SubsystemPolicy(Protocol):
    def select(self, belief: HostState) -> str: ...


@dataclass
class FixedSubsystem:
    """Always verify ``subsystem`` (the single-subsystem ablation arm, §9.2)."""

    subsystem: str

    def __post_init__(self) -> None:
        if self.subsystem not in SUBSYSTEMS:
            raise ValueError(f"unknown subsystem {self.subsystem!r}; choose from {SUBSYSTEMS}")

    def select(self, belief: HostState) -> str:
        return self.subsystem


@dataclass
class RoundRobinSubsystem:
    """Cycle through ``subsystems`` deterministically -- uniform coverage, no randomness."""

    subsystems: Sequence[str] = SUBSYSTEMS
    _i: int = field(default=0)

    def select(self, belief: HostState) -> str:
        subsystem = self.subsystems[self._i % len(self.subsystems)]
        self._i += 1
        return subsystem
