"""Platform-honest tracers behind one interface (OpenSpec ``add-sandbox-trace-oracle``).

A :class:`Tracer` observes one oracle step and emits a :class:`RuntimeTrace`. The interface is a
``begin``/``finish`` pair around the step: ``begin`` starts capture before the execution, ``finish``
assembles the events after — the start/stop contract the spec names.

Two tiers ship, selected at runtime by :func:`select_tracer`:

  - :class:`StraceTracer` — the **full** tier. A ptrace-based ``strace`` wraps the *real* sandbox
    subprocess via the oracle's ``exec_wrapper`` seam, so the trace carries the actual argv and the
    syscall stream the command issued. Selected only where ``strace`` is present and permitted
    (Linux) **and** the wrapped oracle exposes the seam — never as an empty-stream pretender.
  - :class:`DegradedTracer` — the always-available **floor**. A pure post-hoc reader of the action
    and the oracle's structural delta (exec + file delta + binds), tagged ``degraded`` so nothing
    downstream over-trusts it. This is what runs where privileged tracing is unavailable — e.g. the
    macOS dev host, where DTrace needs SIP-disable/entitlements, and any host without ``strace``.

eBPF is never a default (the spec forbids it); :func:`full_tracing_available` is the honest probe
the selection branches on, so a ``full`` tracer is never chosen where it could not actually run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from verisim.delta.edits import Chmod, Create, Delete, Modify, Move
from verisim.env.action import Action
from verisim.env.state import State
from verisim.oracle.base import StepResult

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
    SyscallEvent,
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


# The focused syscall set the full tier records: exec, the file-mutating calls, and the net calls
# (so an unexpected egress attempt would be *seen*, not assumed-absent). Filtering keeps the log
# tiny — well within the seal's RLIMIT_FSIZE — and the parse cheap.
_STRACE_SYSCALLS = (
    "execve,execveat,openat,open,creat,write,unlink,unlinkat,rename,renameat,renameat2,"
    "mkdir,mkdirat,rmdir,chmod,fchmod,fchmodat,socket,connect,bind"
)

# A complete strace line: an optional ``[pid N] `` / ``N `` prefix, then ``name(args) = result``.
# Unfinished/resumed/signal/attach lines are skipped (we parse only complete syscalls).
_STRACE_LINE = re.compile(r"^(?:\[pid\s+\d+\]\s*|\d+\s+)?(\w+)\((.*)\)\s*=\s*(.+?)\s*$")
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _parse_strace(text: str) -> tuple[SyscallEvent, ...]:
    """Parse strace ``-o`` output into typed :class:`SyscallEvent`s (complete lines only)."""
    events: list[SyscallEvent] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or "<unfinished" in s or "resumed>" in s or s.startswith(("+++", "---")):
            continue
        m = _STRACE_LINE.match(s)
        if m is None:
            continue
        events.append(SyscallEvent(name=m.group(1), args=m.group(2), result=m.group(3)))
    return tuple(events)


def _exec_events_from_syscalls(
    syscalls: tuple[SyscallEvent, ...], action: Action, result: StepResult
) -> tuple[ExecEvent, ...]:
    """Real exec events from ``execve``/``execveat`` syscalls (with the actual argv).

    Falls back to the action-level exec event (the degraded shape) if the trace captured no
    ``execve`` — so the spec's "trace contains the exec event" invariant always holds.
    """
    execs: list[ExecEvent] = []
    for sc in syscalls:
        if sc.name not in ("execve", "execveat"):
            continue
        quoted = _QUOTED.findall(sc.args)
        if not quoted:
            continue
        program = quoted[0]
        bracket = sc.args.find("[")
        end = sc.args.find("]", bracket)
        argv = _QUOTED.findall(sc.args[bracket:end]) if bracket != -1 and end != -1 else []
        args = tuple(argv[1:]) if argv else ()
        execs.append(ExecEvent(command=program, args=args, exit_code=result.exit_code))
    if execs:
        return tuple(execs)
    return (ExecEvent(command=action.name, args=tuple(action.args), exit_code=result.exit_code),)


def _net_events_from_syscalls(syscalls: tuple[SyscallEvent, ...]) -> tuple[NetEvent, ...]:
    """Net events from observed ``connect``/``bind`` syscalls (expected empty for the v0 grammar —
    an observed egress attempt would be a real finding, not assumed away)."""
    return tuple(
        NetEvent(kind=sc.name, target=sc.args) for sc in syscalls if sc.name in ("connect", "bind")
    )


class StraceTracer:
    """The ``full`` tier: a ptrace-based syscall tracer (``strace``) wrapping the real subprocess.

    Installed into a :class:`~verisim.oracle.sandbox.SandboxOracle` via its ``exec_wrapper`` seam:
    :meth:`exec_wrapper` prepends ``strace -f -qq -e trace=… -o <log> --`` to the *already-confined*
    rendered argv (a constant trusted prefix — the grammar allowlist is untouched), and
    :meth:`finish` parses the log into the real exec/syscall/net events. File mutations still come
    from the oracle's canonical structural delta; the syscall stream is the full tier's addition.

    Confinement note: the strace log is harness observability written to a Verisim-owned temp file
    (like capturing stdout), not a write by the sandboxed command — the v0 command stays confined to
    its throwaway tree, as the snapshot/delta proves.
    """

    fidelity = FIDELITY_FULL

    def __init__(self, *, strace_bin: str = "strace", syscalls: str = _STRACE_SYSCALLS) -> None:
        self._strace = shutil.which(strace_bin) or strace_bin
        self._syscalls = syscalls
        self._out: str | None = None

    def begin(self) -> None:
        """Allocate a fresh log path for the upcoming step (consumed by :meth:`exec_wrapper`)."""
        fd, path = tempfile.mkstemp(prefix="verisim-strace-", suffix=".log")
        os.close(fd)
        self._out = path

    def exec_wrapper(self, argv: list[str]) -> list[str]:
        """Prepend the strace instrumentation prefix to the rendered argv (constant, trusted)."""
        if self._out is None:
            raise RuntimeError("StraceTracer.begin() must run before exec_wrapper()")
        return [
            self._strace,
            "-f",
            "-qq",
            "-e",
            "trace=" + self._syscalls,
            "-o",
            self._out,
            "--",
            *argv,
        ]

    def finish(
        self,
        *,
        action: Action,
        before: State,
        result: StepResult,
        fixture_source_sha: str | None,
        elapsed_s: float,
    ) -> RuntimeTrace:
        syscalls = self._read_and_clear()
        return RuntimeTrace(
            schema_version=TRACE_SCHEMA_VERSION,
            fidelity=self.fidelity,
            action_name=action.name,
            action_args=tuple(action.args),
            fixture_source_sha=fixture_source_sha,
            exit_code=result.exit_code,
            exec_events=_exec_events_from_syscalls(syscalls, action, result),
            file_mutations=_file_mutations_from_delta(result),
            net_events=_net_events_from_syscalls(syscalls),
            elapsed_s=elapsed_s,
            syscall_events=syscalls,
        )

    def _read_and_clear(self) -> tuple[SyscallEvent, ...]:
        if self._out is None:
            return ()
        path = Path(self._out)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        finally:
            path.unlink(missing_ok=True)
            self._out = None
        return _parse_strace(text)


def _strace_usable() -> bool:
    """Whether ``strace`` is present and actually permitted to trace here (a real probe).

    ``strace`` is Linux/ptrace-only, so this is ``False`` off Linux (macOS needs privileged DTrace,
    which the SIP-locked dev host does not grant). On Linux it confirms the binary exists *and* a
    trivial trace of ``/bin/true`` succeeds — so a restrictive ``ptrace_scope`` degrades cleanly.
    """
    if sys.platform != "linux":
        return False
    strace = shutil.which("strace")
    if strace is None or not os.path.exists("/bin/true"):
        return False
    try:
        proc = subprocess.run(
            [strace, "-f", "-qq", "-o", os.devnull, "/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _has_exec_seam(oracle: object | None) -> bool:
    """Whether ``oracle`` exposes the exec-instrumentation seam the full tier needs."""
    return oracle is not None and hasattr(oracle, "exec_wrapper")


def full_tracing_available(oracle: object | None = None) -> bool:
    """Whether the ``full`` tier can actually run — a permitted ``strace`` **and**, if an oracle is
    given, an exec-instrumentation seam to install it into. Branch on this named capability rather
    than assuming a tier; ``select_tracer`` uses it to never pick a ``full`` tracer that would emit
    an empty syscall stream while claiming ``full`` fidelity."""
    if not _strace_usable():
        return False
    return oracle is None or _has_exec_seam(oracle)


def tracing_capability_note(oracle: object | None = None) -> str:
    """A human-readable line stating which tier is in force and why (for disclosure)."""
    if full_tracing_available(oracle):
        return "full tier: ptrace-based strace wraps the real sandbox subprocess (exec + syscalls)."
    host = "macOS" if sys.platform == "darwin" else sys.platform
    if sys.platform != "linux":
        why = (
            f"privileged syscall tracing unavailable on {host} (no permitted strace; DTrace gated)"
        )
    elif not _strace_usable():
        why = "strace is absent or not permitted (ptrace_scope) on this Linux host"
    else:
        why = "the wrapped oracle exposes no exec-instrumentation seam"
    return f"degraded tier: {why}. Floor recorded: exec + file delta + binds."


def select_tracer(oracle: object | None = None) -> Tracer:
    """Select a platform-appropriate tracer at runtime (the ``ExplicitTracerFidelity`` requirement).

    Returns the ``full`` (strace) tracer when it can actually run — a permitted ``strace`` and an
    oracle exposing the exec-instrumentation seam — else the always-available degraded floor. eBPF
    is never selected by default (the spec forbids it).
    """
    if full_tracing_available(oracle):
        return StraceTracer()
    return DegradedTracer()
