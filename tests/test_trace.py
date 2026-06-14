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
    MUT_CREATED,
    TRACE_SCHEMA_VERSION,
    DegradedTracer,
    RuntimeTrace,
    TraceBudgetExceeded,
    TraceError,
    TracingOracle,
    full_tracing_available,
    select_tracer,
    write_trace,
)

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
    # The exec event, linked to the action.
    assert trace.action_name == "write"
    assert trace.action_args == ("/hi.txt", "hello")
    assert any(e.command == "write" for e in trace.exec_events)
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
    assert full_tracing_available() is False
    assert isinstance(select_tracer(), DegradedTracer)

    oracle = TracingOracle(ReferenceOracle())  # defaults to the selected (degraded) tracer
    oracle.step(State.empty(), parse_action("write /a hello"))
    trace = oracle.traces[0]
    assert trace.fidelity == FIDELITY_DEGRADED
    assert trace.is_degraded()
    assert trace.exec_events  # exec recorded
    assert trace.file_mutations  # file delta recorded
    assert trace.net_events == ()  # binds recorded (empty for this grammar)


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
