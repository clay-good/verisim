"""The typed runtime-trace model (OpenSpec ``add-sandbox-trace-oracle``).

A :class:`RuntimeTrace` is the faithful, reproducible record of *one* real
:class:`~verisim.oracle.base.StepResult`'s observable effects — the dynamic facts static analysis
cannot see (what actually exec'd, which files actually mutated, which binds were attempted). It is
the dynamic half of the static↔dynamic correction the prototype exists to make (findings §3): the
static :class:`~verisim.bridge.graph.CodeGraph` says what *could* happen; a trace says what *did*.

Each trace carries its **fidelity tier** ([`FIDELITY_FULL`]/[`FIDELITY_DEGRADED`]) so a downstream
consumer (Change 4) never treats a degraded trace as authoritative, and a stable link back to the
originating action and the fixture's source sha so every artifact is attributable to a known source
state. Traces are typed and versioned (``schema_version``) so a later reader can detect a format it
does not understand rather than mis-parse it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# Fidelity tiers (the ``ExplicitTracerFidelity`` requirement). ``full`` means a privileged
# syscall-level tracer observed the execution; ``degraded`` means the always-available floor
# (exec + file delta + binds from observable signals) — useful but not authoritative.
FIDELITY_FULL = "full"
FIDELITY_DEGRADED = "degraded"

# The trace artifact format version. Bumped if the on-disk shape changes, so a reader fails closed
# on an unknown version rather than mis-parsing (the bridge schema-guard discipline, write side).
# v2 added ``syscall_events`` (the full-tier syscall stream).
TRACE_SCHEMA_VERSION = 2

# File-mutation kinds, drawn from the oracle's structural delta (reused, not recomputed).
MUT_CREATED = "created"
MUT_MODIFIED = "modified"
MUT_DELETED = "deleted"
MUT_CHMOD = "chmod"


@dataclass(frozen=True, slots=True)
class ExecEvent:
    """A process execution. At the degraded tier this is the v0 action that ran and its coarse
    exit code (the kernel-level argv is a ``full``-tier detail the floor does not observe)."""

    command: str
    args: tuple[str, ...]
    exit_code: int


@dataclass(frozen=True, slots=True)
class FileMutation:
    """A file-tree mutation under the throwaway root, reused from the oracle's structural delta.

    ``mode`` is set only for a ``chmod`` mutation (the new octal mode), ``None`` otherwise.
    """

    path: str
    kind: str
    mode: int | None = None


@dataclass(frozen=True, slots=True)
class NetEvent:
    """A network bind/connect attempt. The v0 filesystem grammar exposes no network surface and the
    sandbox blocks egress by grammar allowlist, so this is empty for v0 actions — recorded honestly
    rather than omitted, so a future grammar with a network action has a place to land."""

    kind: str  # "bind" | "connect"
    target: str


@dataclass(frozen=True, slots=True)
class SyscallEvent:
    """One observed syscall from the `full`-tier (`strace`) tracer: its name, raw argument text, and
    return value. Empty at the degraded tier (no privileged tracer ran)."""

    name: str
    args: str
    result: str


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    """One step's observable runtime effects, tagged with its fidelity tier and provenance."""

    schema_version: int
    fidelity: str
    action_name: str
    action_args: tuple[str, ...]
    fixture_source_sha: str | None
    exit_code: int
    exec_events: tuple[ExecEvent, ...]
    file_mutations: tuple[FileMutation, ...]
    net_events: tuple[NetEvent, ...]
    elapsed_s: float
    syscall_events: tuple[SyscallEvent, ...] = ()

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, stable separators) — stable across dict ordering.

        ``elapsed_s`` (wall-clock, machine-dependent) is included for traceability; structural
        comparisons should key on the typed event fields, not the JSON whole.
        """
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def is_degraded(self) -> bool:
        return self.fidelity == FIDELITY_DEGRADED
