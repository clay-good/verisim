"""Network oracle protocol and step result (SPEC-5 §5).

Mirrors v0's ``Oracle`` protocol (SPEC-2 §3.2) so a Tier-B system oracle (real Linux
network namespaces) can later drop in behind the same interface without the loop/metrics
knowing which oracle they hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.edits import NetDelta


@dataclass(frozen=True)
class NetStepResult:
    """The oracle's verdict on one transition: the true next state, its delta, and exit code.

    ``apply(state, delta) == next_state`` by construction (the NW1 invariant)."""

    state: NetworkState
    delta: NetDelta
    exit_code: int


class NetOracle(Protocol):
    def step(self, state: NetworkState, action: NetAction) -> NetStepResult: ...
