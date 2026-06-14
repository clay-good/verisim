"""Sandbox runtime-trace oracle tests (OpenSpec ``add-sandbox-trace-oracle``).

Pin the three requirements of the trace-oracle spec delta: RuntimeTraceCapture (a real file write
and exec are recorded, linked to the action and fixture sha), DeterminismPreservedUnderTracing (a
traced step is bit-identical to an untraced one), and ExplicitTracerFidelity (the degraded tier is
useful and labeled). The trace machinery is oracle-agnostic, so the model/budget/artifact units run
against the always-available :class:`ReferenceOracle` (hermetic, no shell); the
"real sandbox execution surface" scenarios run against :class:`SandboxOracle` and skip, disclosed,
when no shell is present (the SPEC-11 macOS-first / Linux-CI principle).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from verisim.env.action import Action, parse_action
from verisim.env.state import State
from verisim.oracle.base import DeterminismReport, Oracle, StepResult
from verisim.oracle.differential import canonical_world
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable
from verisim.trace import (
    FIDELITY_DEGRADED,
    FIDELITY_FULL,
    MUT_CREATED,
    TRACE_SCHEMA_VERSION,
    DegradedTracer,
    RuntimeTrace,
    StraceTracer,
    TraceBudgetExceeded,
    TraceError,
    TracingOracle,
    full_tracing_available,
    select_tracer,
    tracing_capability_note,
    write_trace,
)
from verisim.trace.tracer import _parse_strace

try:
    _SHELL: SandboxOracle | None = SandboxOracle()
except SystemOracleUnavailable:  # pragma: no cover - only on a host with no shell
    _SHELL = None

requires_shell = pytest.mark.skipif(_SHELL is None, reason="no real shell (disclosed)")


# --- RuntimeTraceCapture -----------------------------------------------------------------------


@requires_shell
def test_file_write_and_exec_are_recorded_in_the_sandbox() -> None:
    """Scenario: a fixture action that writes a file and runs in the sandbox is recorded."""
    oracle = TracingOracle(SandboxOracle(), fixture_source_sha="deadbeef")
    action = parse_action("write /hi.txt hello")
    oracle.step(State.empty(), action)

    assert len(oracle.traces) == 1
    trace = oracle.traces[0]
    # The exec event is recorded and the trace is linked to the action (tier-agnostically: at the
    # degraded tier the exec command is the action name, at the full tier it is the real argv).
    assert trace.action_name == "write"
    assert trace.action_args == ("/hi.txt", "hello")
    assert trace.exec_events
    # The file mutation, reused from the oracle's structural delta.
    assert any(m.path == "/hi.txt" and m.kind == MUT_CREATED for m in trace.file_mutations)
    # Linked to the fixture source sha.
    assert trace.fixture_source_sha == "deadbeef"


def test_trace_capture_works_over_any_oracle() -> None:
    """The decorator is oracle-agnostic; the reference oracle exercises the same capture path."""
    oracle = TracingOracle(ReferenceOracle(), fixture_source_sha="abc123")
    oracle.step(State.empty(), parse_action("mkdir /d"))
    trace = oracle.traces[0]
    assert trace.action_name == "mkdir"
    assert any(m.path == "/d" and m.kind == MUT_CREATED for m in trace.file_mutations)
    assert trace.exec_events[0].exit_code == 0


def test_net_events_empty_for_v0_grammar() -> None:
    """The v0 filesystem grammar has no network surface; binds are honestly empty, not omitted."""
    oracle = TracingOracle(ReferenceOracle())
    oracle.step(State.empty(), parse_action("write /a x"))
    assert oracle.traces[0].net_events == ()


# --- DeterminismPreservedUnderTracing --------------------------------------------------------


@requires_shell
def test_traced_and_untraced_steps_agree() -> None:
    """Scenario: a step run with and without tracing yields an identical canonical StepResult."""
    plain = SandboxOracle()
    traced = TracingOracle(SandboxOracle())
    state = State.empty()
    for raw in ("mkdir /d", "write /d/f.txt content", "chmod 600 /d/f.txt", "rm /d/f.txt"):
        action = parse_action(raw)
        r_plain = plain.step(state, action)
        r_traced = traced.step(state, action)
        assert canonical_world(r_plain.state) == canonical_world(r_traced.state), raw
        assert r_plain.exit_code == r_traced.exit_code, raw
        assert r_plain.stdout == r_traced.stdout, raw
        state = r_plain.state


def test_traced_result_is_inner_result_verbatim() -> None:
    """Observational purity: the wrapper returns the inner oracle's StepResult unchanged."""

    sentinel: list[StepResult] = []

    class _Recording:
        version = "rec-1"

        def step(self, state: State, action: Action) -> StepResult:
            r = ReferenceOracle().step(state, action)
            sentinel.append(r)
            return r

        def reset(self, state: State) -> State:
            return state.copy()

        def determinism_report(self) -> DeterminismReport:
            return ReferenceOracle().determinism_report()

    inner = _Recording()
    oracle = TracingOracle(inner)
    out = oracle.step(State.empty(), parse_action("mkdir /d"))
    assert out is sentinel[0]  # identical object, not a copy


# --- ExplicitTracerFidelity ------------------------------------------------------------------


def test_degraded_tier_is_useful_and_labeled() -> None:
    """Scenario: where privileged tracing is unavailable, the trace is degraded yet still useful."""
    oracle = TracingOracle(ReferenceOracle())  # ReferenceOracle has no exec seam → degraded
    oracle.step(State.empty(), parse_action("write /a hello"))
    trace = oracle.traces[0]
    assert trace.fidelity == FIDELITY_DEGRADED
    assert trace.is_degraded()
    assert trace.exec_events  # exec recorded
    assert trace.file_mutations  # file delta recorded
    assert trace.net_events == ()  # binds recorded (empty for this grammar)
    assert trace.syscall_events == ()  # no privileged tracer ran


def test_full_tier_is_chosen_only_when_it_can_actually_work() -> None:
    """Scenario: the full tier is never chosen where it would emit an empty syscall stream.

    An oracle with no exec seam (ReferenceOracle) always degrades; selecting over a seam-ful oracle
    yields the full tier iff strace is actually usable here (true only on a permitted-ptrace Linux).
    """
    assert isinstance(select_tracer(ReferenceOracle()), DegradedTracer)
    assert full_tracing_available(ReferenceOracle()) is False  # no seam

    seamful = SandboxOracle() if _SHELL is not None else None
    if seamful is not None:
        expect_full = full_tracing_available(seamful)
        chosen = select_tracer(seamful)
        assert isinstance(chosen, StraceTracer) == expect_full
        assert ("full tier" in tracing_capability_note(seamful)) == expect_full


def test_strace_output_parses_into_typed_events() -> None:
    """The strace parser pins the real output format (pid prefixes, unfinished/resumed skipped)."""
    sample = (
        'execve("/bin/mkdir", ["mkdir", "/tmp/x/d"], 0x7ffd /* 12 vars */) = 0\n'
        '[pid  4711] openat(AT_FDCWD, "/tmp/x/d", O_RDONLY|O_DIRECTORY) = 3\n'
        '4711  write(1, "ok", 2)            = 2\n'
        "strace: Process 4711 attached\n"
        "connect(5, {sa_family=AF_INET, sin_port=htons(80)}, 16) <unfinished ...>\n"
        "+++ exited with 0 +++\n"
    )
    events = _parse_strace(sample)
    names = [e.name for e in events]
    assert names == ["execve", "openat", "write"]  # attach line + unfinished + +++ all skipped
    assert events[0].result == "0"


# --- full tier: exec-wrapper seam + strace integration ---------------------------------------


@requires_shell
def test_exec_wrapper_seam_is_applied_and_preserves_the_result() -> None:
    """The SandboxOracle exec seam wraps the real argv without changing the step's result."""
    seen: list[list[str]] = []

    def wrapper(argv: list[str]) -> list[str]:
        seen.append(list(argv))
        return ["/usr/bin/env", *argv]  # exec the rendered command through `env`

    oracle = SandboxOracle(exec_wrapper=wrapper)
    result = oracle.step(State.empty(), parse_action("mkdir /d"))
    assert seen and seen[0][0] == "mkdir"  # the original rendered argv was passed through
    assert "/d" in result.state.fs  # the wrapped command still produced the real effect


def _make_fake_strace(dirpath: Path) -> Path:
    """A stand-in for `strace -o`: writes a canned syscall log, then execs the real command.

    Lets the full-tier wiring (wrapper → subprocess → log file → parse → trace) be exercised on
    macOS, where real strace does not exist — the parser's format fidelity is pinned separately by
    ``test_strace_output_parses_into_typed_events`` and the real binary by the Linux-gated e2e.
    """
    script = dirpath / "fake-strace"
    script.write_text(
        "#!/bin/sh\n"
        'out=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    -o) shift; out="$1" ;;\n'
        "    --) shift; break ;;\n"
        "  esac\n"
        "  shift\n"
        "done\n"
        "# $@ is now the real command\n"
        'printf \'execve("%s", ["%s"], 0x0) = 0\\nopenat(AT_FDCWD, "x", O_WRONLY) = 3\\n\' '
        '"$1" "$1" > "$out"\n'
        'exec "$@"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@requires_shell
def test_full_tier_wiring_end_to_end_with_a_fake_strace(tmp_path: Path) -> None:
    """The whole full-tier chain on macOS via a fake strace: wrapper installed, log parsed, result
    preserved, trace tagged `full` with a real syscall stream."""
    fake = _make_fake_strace(tmp_path)
    tracer = StraceTracer(strace_bin=str(fake))
    plain = SandboxOracle()
    traced = TracingOracle(SandboxOracle(), tracer=tracer, fixture_source_sha="feed")

    action = parse_action("mkdir /d")
    r_plain = plain.step(State.empty(), action)
    r_traced = traced.step(State.empty(), action)

    # Determinism preserved: the wrapped command produced the identical result.
    assert canonical_world(r_traced.state) == canonical_world(r_plain.state)
    trace = traced.traces[0]
    assert trace.fidelity == FIDELITY_FULL
    assert any(sc.name == "execve" for sc in trace.syscall_events)
    assert trace.exec_events[0].command == "mkdir"  # real argv from the execve line
    assert any(m.path == "/d" and m.kind == MUT_CREATED for m in trace.file_mutations)
    assert trace.fixture_source_sha == "feed"


@pytest.mark.skipif(
    _SHELL is None or not full_tracing_available(_SHELL),
    reason="real strace not available/permitted (Linux + ptrace only)",
)
def test_full_tier_with_real_strace(tmp_path: Path) -> None:
    """On a permitted-ptrace Linux host, real strace records the true syscall stream, and the
    traced result still matches the untraced one."""
    plain = SandboxOracle()
    traced = TracingOracle(SandboxOracle(), tracer=StraceTracer())
    action = parse_action("write /f.txt hello")
    r_plain = plain.step(State.empty(), action)
    r_traced = traced.step(State.empty(), action)

    assert canonical_world(r_traced.state) == canonical_world(r_plain.state)
    trace = traced.traces[0]
    assert trace.fidelity == FIDELITY_FULL
    assert trace.syscall_events  # real syscalls captured
    assert any(sc.name in ("execve", "execveat") for sc in trace.syscall_events)
    # the write went to a real file syscall
    assert any(sc.name in ("openat", "open", "write", "creat") for sc in trace.syscall_events)


# --- trace model + artifacts -----------------------------------------------------------------


def test_runtime_trace_json_round_trips() -> None:
    oracle = TracingOracle(ReferenceOracle(), fixture_source_sha="sha1")
    oracle.step(State.empty(), parse_action("write /a hello"))
    trace = oracle.traces[0]
    data = json.loads(trace.to_json())
    assert data["schema_version"] == TRACE_SCHEMA_VERSION
    assert data["fidelity"] == FIDELITY_DEGRADED
    assert data["action_name"] == "write"
    assert data["fixture_source_sha"] == "sha1"
    assert data["file_mutations"][0]["path"] == "/a"


def test_traces_written_to_scratch_are_typed_and_versioned(tmp_path: Path) -> None:
    out = tmp_path / "traces"
    oracle = TracingOracle(ReferenceOracle(), artifact_dir=out)
    oracle.step(State.empty(), parse_action("mkdir /d"))
    oracle.step(State.empty(), parse_action("write /a x"))
    files = sorted(out.glob("trace-*.json"))
    assert len(files) == 2
    first = json.loads(files[0].read_text(encoding="utf-8"))
    assert first["schema_version"] == TRACE_SCHEMA_VERSION
    assert first["action_name"] == "mkdir"


def test_write_trace_refuses_a_source_root() -> None:
    """Traces are scratch-only — writing under the source-roots allowlist is refused."""
    from verisim.fixture import DEFAULT_SOURCE_ROOT

    trace = RuntimeTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        fidelity=FIDELITY_DEGRADED,
        action_name="mkdir",
        action_args=("/d",),
        fixture_source_sha=None,
        exit_code=0,
        exec_events=(),
        file_mutations=(),
        net_events=(),
        elapsed_s=0.0,
    )
    with pytest.raises(TraceError, match="scratch-only"):
        write_trace(trace, DEFAULT_SOURCE_ROOT / "verisim" / "traces", index=0)


# --- budget ----------------------------------------------------------------------------------


def test_tracing_overhead_budget_fails_loudly() -> None:
    """A step whose tracing overruns its budget fails loudly rather than eating the step budget."""

    class _SlowTracer:
        fidelity = FIDELITY_DEGRADED

        def __init__(self) -> None:
            self._inner = DegradedTracer()

        def begin(self) -> None:
            pass

        def finish(
            self,
            *,
            action: Action,
            before: State,
            result: StepResult,
            fixture_source_sha: str | None,
            elapsed_s: float,
        ) -> RuntimeTrace:
            time.sleep(0.05)
            return self._inner.finish(
                action=action,
                before=before,
                result=result,
                fixture_source_sha=fixture_source_sha,
                elapsed_s=elapsed_s,
            )

    oracle = TracingOracle(ReferenceOracle(), tracer=_SlowTracer(), overhead_budget_s=0.01)
    with pytest.raises(TraceBudgetExceeded):
        oracle.step(State.empty(), parse_action("mkdir /d"))


# --- decorator fidelity (drop-in) ------------------------------------------------------------


@requires_shell
def test_wrapper_delegates_protocol_and_extras() -> None:
    inner = SandboxOracle()
    oracle = TracingOracle(inner)
    assert isinstance(oracle, Oracle)
    # delegated Oracle protocol
    s = State.empty()
    assert oracle.reset(s).fs == s.fs
    assert oracle.determinism_report().clock_sealed
    # delegated SandboxOracle extra (via __getattr__)
    assert oracle.hermeticity().fs_confined
    assert oracle.version == inner.version
