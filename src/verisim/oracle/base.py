"""The ``Oracle`` interface (SPEC-2 §3.2).

This is one of v0's two sanctioned forward-looking abstractions (SPEC-2 §14): a
single protocol so Phase 1's ``SandboxOracle`` (a real namespaced shell) drops in
unchanged. Nothing in the codebase should branch on *which* oracle it holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from verisim.delta.edits import Delta
from verisim.env.action import Action
from verisim.env.state import State


@dataclass(frozen=True)
class StepResult:
    """Outcome of one oracle step: the true next state, the delta that produced
    it, and the observation (exit code + stdout)."""

    state: State
    delta: Delta
    exit_code: int
    stdout: str


@dataclass(frozen=True)
class DeterminismReport:
    """Which nondeterminism sources are sealed (SPEC-2 §3.2). The reference
    oracle seals all of them by construction; the system oracle will not."""

    clock_sealed: bool
    rng_sealed: bool
    concurrency_sealed: bool
    env_leakage_sealed: bool
    notes: str = ""


@runtime_checkable
class Oracle(Protocol):
    def step(self, state: State, action: Action) -> StepResult: ...

    def reset(self, state: State) -> State:
        """Return a restorable snapshot of ``state`` (a deep-enough copy)."""
        ...

    def determinism_report(self) -> DeterminismReport: ...


# Exit codes (SPEC-2 §2.2 failure modes). v0 uses a coarse OK/ERR split; the
# specific failure *condition* per command is documented in docs/semantics.md.
EXIT_OK = 0
EXIT_ERR = 1
