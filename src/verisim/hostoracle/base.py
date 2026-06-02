"""The host oracle protocol + step result (SPEC-6 §5, HC0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from verisim.host.action import HostAction
from verisim.host.delta import HostDelta
from verisim.host.state import HostState

EXIT_OK = 0
EXIT_ERR = 1


@dataclass(frozen=True)
class HostStepResult:
    """One host transition: the true next bundle state, the **bundle delta** that produced it, and
    the observation (exit code + stdout). ``apply(state, delta) == state`` holds by construction
    (the M1-analogue invariant, HC1)."""

    state: HostState
    delta: HostDelta
    exit_code: int
    stdout: str


class HostOracle(Protocol):
    """A deterministic interpreter of the syscall grammar over the bundle state."""

    def step(self, state: HostState, action: HostAction) -> HostStepResult: ...
