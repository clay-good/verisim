"""Platform-honest tracers behind one interface (OpenSpec ``add-sandbox-trace-oracle``).

A :class:`Tracer` observes one oracle step and emits a :class:`RuntimeTrace`. The interface is a
``begin``/``finish`` pair around the step so a future privileged tracer (a syscall-level ``full``
tier) can start capture before the execution and assemble events after — the start/stop contract
the spec names.

Only the **degraded** tier is wired here, and that is a deliberate, disclosed choice, not an
omission:

  - The tracer is a *pure decorator* over the ``Oracle`` protocol (the proposal's "wrapper, not
    rewrite" constraint), so it cannot inject ``ptrace``/DTrace into the subprocess
    ``SandboxOracle`` spawns *inside* its own ``step`` — a ``full``-tier syscall tracer would have
    to instrument that spawn, which is out of scope for a decorator.
  - On the macOS development host, privileged tracing of an arbitrary process (DTrace / the
    ``ptrace``-equivalent) requires disabling SIP or holding entitlements the dev environment
    lacks (the proposal's stated risk).

So the spec's mandated **floor** is what ships: the degraded tracer records exec + file delta +
binds from the signals a decorator *can* observe (the action and the oracle's structural delta),
and tags every trace ``degraded`` so nothing downstream over-trusts it. eBPF is never a default
(the spec forbids it); :func:`select_tracer` returns the degraded tier at runtime, and
:func:`full_tracing_available` reports — honestly — that the privileged tier is not wired.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

from verisim.delta.edits import Chmod, Create, Delete, Modify, Move
from verisim.env.action import Action
from verisim.env.state import State
from verisim.oracle.base import StepResult

from .model import (
    FIDELITY_DEGRADED,
    MUT_CHMOD,
    MUT_CREATED,
    MUT_DELETED,
    MUT_MODIFIED,
    TRACE_SCHEMA_VERSION,
    ExecEvent,
    FileMutation,
    RuntimeTrace,
)


@runtime_checkable
class Tracer(Protocol):
    """Observe one oracle step and emit a :class:`RuntimeTrace`.

    ``fidelity`` is the tier this tracer produces. ``begin`` is called immediately before the step
    (a no-op for the degraded tier; where a live capture would start for a future ``full`` tier),
    ``finish`` immediately after, given the step's inputs/outputs and timing.
    """

    fidelity: str

    def begin(self) -> None: ...

    def finish(
        self,
        *,
        action: Action,
        before: State,
        result: StepResult,
        fixture_source_sha: str | None,
        elapsed_s: float,
    ) -> RuntimeTrace: ...


def _file_mutations_from_delta(result: StepResult) -> tuple[FileMutation, ...]:
    """Project the oracle's structural delta into file mutations — *reused*, not recomputed.

    The delta is the oracle's own truth representation (SPEC-2 §5.1); cwd/env/result edits are not
    file-tree mutations and are skipped. A ``Move`` (not produced by ``SandboxOracle``, which diffs
    as delete+create, but possible from other oracles) is recorded as a delete of the source and a
    create of the destination.
    """
    muts: list[FileMutation] = []
    for edit in result.delta:
        if isinstance(edit, Create):
            muts.append(FileMutation(path=edit.path, kind=MUT_CREATED))
        elif isinstance(edit, Delete):
            muts.append(FileMutation(path=edit.path, kind=MUT_DELETED))
        elif isinstance(edit, Modify):
            muts.append(FileMutation(path=edit.path, kind=MUT_MODIFIED))
        elif isinstance(edit, Chmod):
            muts.append(FileMutation(path=edit.path, kind=MUT_CHMOD, mode=edit.mode))
        elif isinstance(edit, Move):
            muts.append(FileMutation(path=edit.src, kind=MUT_DELETED))
            muts.append(FileMutation(path=edit.dst, kind=MUT_CREATED))
    return tuple(muts)


class DegradedTracer:
    """The always-available floor: exec + file delta + binds from a decorator's observable signals.

    Builds the trace post-hoc from the action and the oracle's :class:`StepResult` — purely
    observational, so a traced step is bit-identical to an untraced one. Net events are empty: the
    v0 grammar has no network action and the sandbox blocks egress by allowlist.
    """

    fidelity = FIDELITY_DEGRADED

    def begin(self) -> None:
        """No live capture at the degraded tier — the trace is assembled in :meth:`finish`."""

    def finish(
        self,
        *,
        action: Action,
        before: State,
        result: StepResult,
        fixture_source_sha: str | None,
        elapsed_s: float,
    ) -> RuntimeTrace:
        exec_event = ExecEvent(
            command=action.name,
            args=tuple(action.args),
            exit_code=result.exit_code,
        )
        return RuntimeTrace(
            schema_version=TRACE_SCHEMA_VERSION,
            fidelity=self.fidelity,
            action_name=action.name,
            action_args=tuple(action.args),
            fixture_source_sha=fixture_source_sha,
            exit_code=result.exit_code,
            exec_events=(exec_event,),
            file_mutations=_file_mutations_from_delta(result),
            net_events=(),
            elapsed_s=elapsed_s,
        )


def full_tracing_available() -> bool:
    """Whether a privileged ``full``-tier syscall tracer is wired and usable here.

    Always ``False`` in the shipped prototype: full-fidelity tracing would require instrumenting the
    subprocess ``SandboxOracle`` spawns inside its own ``step`` (impossible for a pure decorator)
    and, on macOS, privileged process tracing the SIP-locked dev host does not grant. The check is
    explicit so :func:`select_tracer` and callers branch on a named capability, not an assumption.
    """
    return False


def tracing_capability_note() -> str:
    """A human-readable reason the degraded tier is in force (for trace/figure disclosure)."""
    host = "macOS" if sys.platform == "darwin" else sys.platform
    return (
        f"degraded tier on {host}: full-fidelity syscall tracing is not wired — a pure Oracle "
        "decorator cannot instrument the sandbox's internal subprocess spawn, and privileged "
        "process tracing is unavailable on the dev host. Degraded floor: exec + file delta + binds."
    )


def select_tracer() -> Tracer:
    """Select a platform-appropriate tracer at runtime (the ``ExplicitTracerFidelity`` requirement).

    Returns the highest *implemented and usable* tier. Today that is always the degraded tracer
    (see :func:`full_tracing_available`); eBPF is never selected by default (the spec forbids it).
    """
    return DegradedTracer()
