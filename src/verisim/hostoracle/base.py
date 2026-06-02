"""The host oracle protocol + step result (SPEC-6 §5, HC0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from verisim.host.action import HostAction
from verisim.host.state import HostState

EXIT_OK = 0
EXIT_ERR = 1


@dataclass(frozen=True)
class HostStepResult:
    """One host transition: the true next bundle state + the observation (exit code + stdout).

    The compositional bundle ``Delta`` and the ``apply == oracle`` invariant are HC1 (the next
    increment); HC0 returns the next state directly, as the oracle's executable truth.
    """

    state: HostState
    exit_code: int
    stdout: str


class HostOracle(Protocol):
    """A deterministic interpreter of the syscall grammar over the bundle state."""

    def step(self, state: HostState, action: HostAction) -> HostStepResult: ...
