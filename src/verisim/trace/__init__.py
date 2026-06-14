"""Sandbox runtime-trace oracle (OpenSpec ``add-sandbox-trace-oracle``).

The third link in the Verisim ↔ OpenLore prototype chain (findings §3): capture what the code
*actually does* at runtime by tracing the real :class:`~verisim.oracle.sandbox.SandboxOracle`
execution — exec, file mutations, binds — as typed :class:`RuntimeTrace` records, so dynamic
reality can later correct (Change 4) the static call graph (Change 2). Tracing is an additive,
removable decorator (:class:`TracingOracle`) that never perturbs the oracle's result or its
``DeterminismSeal``; every trace self-reports its fidelity tier so a degraded trace is never
mistaken for an authoritative one.

See :mod:`verisim.trace.model` for the types, :mod:`verisim.trace.tracer` for the platform-honest
tracers, and :mod:`verisim.trace.oracle` for the decorator.
"""

from __future__ import annotations

from .model import (
    FIDELITY_DEGRADED,
    FIDELITY_FULL,
    MUT_CHMOD,
    MUT_CREATED,
    MUT_DELETED,
    MUT_MODIFIED,
    TRACE_SCHEMA_VERSION,
    ExecEvent,
    FileMutation,
    NetEvent,
    RuntimeTrace,
)
from .oracle import TraceBudgetExceeded, TraceError, TracingOracle, write_trace
from .tracer import (
    DegradedTracer,
    Tracer,
    full_tracing_available,
    select_tracer,
    tracing_capability_note,
)

__all__ = [
    "FIDELITY_DEGRADED",
    "FIDELITY_FULL",
    "MUT_CHMOD",
    "MUT_CREATED",
    "MUT_DELETED",
    "MUT_MODIFIED",
    "TRACE_SCHEMA_VERSION",
    "DegradedTracer",
    "ExecEvent",
    "FileMutation",
    "NetEvent",
    "RuntimeTrace",
    "TraceBudgetExceeded",
    "TraceError",
    "Tracer",
    "TracingOracle",
    "full_tracing_available",
    "select_tracer",
    "tracing_capability_note",
    "write_trace",
]
