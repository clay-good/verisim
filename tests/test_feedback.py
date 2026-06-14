"""Contract-mediated synthesized-edge feedback tests (OpenSpec ``add-synthesized-edge-feedback``).

Pin the four requirements of the feedback spec delta: RuntimeDiscrepancyDetection (a runtime-only
dynamic dispatch becomes exactly one candidate; an already-known edge is not re-proposed),
ContractConformantFeedbackPayload (versioned, labeled, evidence-bearing, idempotent, no DB write),
NodeResolutionFailClosed (an unanchorable invocation is dropped and counted), and PayloadValidation
(the local stand-in accepts a well-formed payload and rejects malformed/mislabeled/over-claiming
ones).

The detector is pure logic over a :class:`RuntimeTrace` and a :class:`CodeGraph`, so the tests
construct both directly — no shell, no real OpenLore database — exercising the static↔dynamic diff
deterministically on any platform.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verisim.bridge import (
    EDGE_CONFIDENCE,
    FEEDBACK_SCHEMA_VERSION,
    FINDING_KIND,
    SYNTHESIZED_BY,
    CallEdge,
    CodeGraph,
    CodeNode,
    FeedbackError,
    FeedbackValidationError,
    LayerInvariant,
    build_feedback_payload,
    detect_candidates,
    detect_findings,
    validate_payload,
    write_feedback,
)
from verisim.trace.model import (
    FIDELITY_DEGRADED,
    FIDELITY_FULL,
    TRACE_SCHEMA_VERSION,
    ExecEvent,
    RuntimeTrace,
)

# --- builders ----------------------------------------------------------------------------------


def _node(
    node_id: str, name: str, file_path: str, *, entry: bool = False, start: int = 0
) -> CodeNode:
    return CodeNode(
        id=node_id,
        name=name,
        file_path=file_path,
        class_name=None,
        language="python",
        is_async=False,
        fan_in=0,
        fan_out=0,
        is_external=False,
        external_kind=None,
        is_hub=False,
        is_entry_point=entry,
        start_index=start,
        end_index=start + 10,
        signature=None,
        docstring=None,
    )


def _edge(caller_id: str, callee_id: str, callee_name: str) -> CallEdge:
    return CallEdge(
        caller_id=caller_id,
        caller_file=caller_id.split("::")[0],
        callee_id=callee_id,
        callee_file=callee_id.split("::")[0],
        callee_name=callee_name,
        line=1,
        confidence="import",
        kind="calls",
        call_type="direct",
        synthesized_by=None,
    )


# A small fixture graph: ``main`` (entrypoint) statically calls ``helper`` but NOT ``run`` — ``run``
# is only ever reached at runtime via dynamic dispatch (the blind spot).
MAIN = _node("app/main.py::main", "main", "app/main.py", entry=True)
RUN = _node("app/plugin.py::run", "run", "app/plugin.py", start=5)
HELPER = _node("app/util.py::helper", "helper", "app/util.py", start=2)


def _graph(*, edges: tuple[CallEdge, ...] = ()) -> CodeGraph:
    return CodeGraph(
        nodes=(MAIN, RUN, HELPER),
        edges=edges,
        classes=(),
        cfg=(),
        schema_version=7,
        db_path="/tmp/fake/call-graph.db",
        db_content_hash="dbhash-abc",
        source_path="/tmp/fake/repo",
        source_tree_hash="srchash",
        fingerprint=None,
    )


def _trace(
    *exec_events: ExecEvent,
    fidelity: str = FIDELITY_FULL,
    action: str = "run-app",
    sha: str | None = "feed1234",
) -> RuntimeTrace:
    return RuntimeTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        fidelity=fidelity,
        action_name=action,
        action_args=(),
        fixture_source_sha=sha,
        exit_code=0,
        exec_events=exec_events,
        file_mutations=(),
        net_events=(),
        elapsed_s=0.01,
        syscall_events=(),
    )


def _exec(command: str, *args: str) -> ExecEvent:
    return ExecEvent(command=command, args=tuple(args), exit_code=0)


# --- RuntimeDiscrepancyDetection ---------------------------------------------------------------


def test_runtime_only_dynamic_dispatch_becomes_one_candidate() -> None:
    """Scenario: a fixture call that fires at runtime with no static edge yields one candidate."""
    graph = _graph(edges=(_edge(MAIN.id, HELPER.id, "helper"),))  # main→helper static, main→run not
    trace = _trace(_exec("app/main.py"), _exec("app/plugin.py"))

    candidates, dropped = detect_candidates(trace, graph)

    assert dropped == 0
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.caller_id == MAIN.id
    assert cand.callee_id == RUN.id
    assert cand.callee_name == "run"
    # Correct provenance + evidence.
    assert cand.confidence == EDGE_CONFIDENCE == "synthesized"
    assert cand.synthesized_by == SYNTHESIZED_BY == "verisim-runtime"
    assert cand.kind == "calls"
    assert cand.line is None and cand.call_type is None  # no source-level call site
    assert cand.evidence.trace_action == "run-app"
    assert cand.evidence.fixture_source_sha == "feed1234"
    assert cand.evidence.fidelity == FIDELITY_FULL


def test_already_known_edge_is_not_re_proposed() -> None:
    """Scenario: a runtime call already in the static graph produces no candidate (idempotency)."""
    graph = _graph(edges=(_edge(MAIN.id, RUN.id, "run"),))  # main→run already known
    trace = _trace(_exec("app/main.py"), _exec("app/plugin.py"))
    candidates, dropped = detect_candidates(trace, graph)
    assert candidates == [] and dropped == 0


def test_pre_existing_synthesized_edge_is_not_re_proposed() -> None:
    """An edge OpenLore *already synthesized* must never be re-proposed."""
    synth = CallEdge(
        caller_id=MAIN.id,
        caller_file="app/main.py",
        callee_id=RUN.id,
        callee_file="app/plugin.py",
        callee_name="run",
        line=None,
        confidence="synthesized",
        kind="calls",
        call_type=None,
        synthesized_by="event-channel",
    )
    graph = _graph(edges=(synth,))
    candidates, _ = detect_candidates(_trace(_exec("app/main.py"), _exec("app/plugin.py")), graph)
    assert candidates == []


def test_interpreter_invocation_resolves_the_script_not_the_interpreter() -> None:
    """An interpreter run (python3 app/plugin.py) resolves to the script in argv, not python3."""
    graph = _graph()
    trace = _trace(_exec("app/main.py"), _exec("/usr/bin/python3", "app/plugin.py"))
    candidates, dropped = detect_candidates(trace, graph)
    assert dropped == 0
    assert [c.callee_id for c in candidates] == [RUN.id]


def test_absolute_program_paths_normalize_against_fixture_root(tmp_path: Path) -> None:
    """Full-tier traces record absolute paths; they normalize to repo-relative node file paths."""
    graph = _graph()
    trace = _trace(
        _exec(str(tmp_path / "app/main.py")),
        _exec(str(tmp_path / "app/plugin.py")),
    )
    candidates, dropped = detect_candidates(trace, graph, fixture_root=tmp_path)
    assert dropped == 0
    assert [c.callee_id for c in candidates] == [RUN.id]


def test_degraded_trace_yields_no_candidates() -> None:
    """A degraded trace observed no syscall stream — it never proposes an edge."""
    graph = _graph()
    trace = _trace(_exec("app/main.py"), _exec("app/plugin.py"), fidelity=FIDELITY_DEGRADED)
    candidates, dropped = detect_candidates(trace, graph)
    assert candidates == [] and dropped == 0


# --- NodeResolutionFailClosed ------------------------------------------------------------------


def test_unresolvable_invocation_is_dropped_and_counted() -> None:
    """Scenario: a runtime call whose endpoint maps to no node is dropped, not emitted; counted."""
    graph = _graph()
    trace = _trace(_exec("app/main.py"), _exec("app/ghost.py"))  # ghost.py has no node
    candidates, dropped = detect_candidates(trace, graph)
    assert candidates == []
    assert dropped == 1


def test_external_node_is_not_a_fixture_invocation() -> None:
    """An external/stdlib leaf is never a fixture call target — resolution fails closed."""
    external = CodeNode(
        id="os::system",
        name="system",
        file_path="app/plugin.py",  # same file_path, but external → must not match
        class_name=None,
        language="python",
        is_async=False,
        fan_in=0,
        fan_out=0,
        is_external=True,
        external_kind="stdlib",
        is_hub=False,
        is_entry_point=False,
        start_index=0,
        end_index=1,
        signature=None,
        docstring=None,
    )
    graph = CodeGraph(
        nodes=(MAIN, external),
        edges=(),
        classes=(),
        cfg=(),
        schema_version=7,
        db_path="x",
        db_content_hash="h",
        source_path=None,
        source_tree_hash=None,
        fingerprint=None,
    )
    # main resolves (entry); plugin.py only has an external node → unresolvable callee → dropped.
    candidates, dropped = detect_candidates(
        _trace(_exec("app/main.py"), _exec("app/plugin.py")), graph
    )
    assert candidates == [] and dropped == 1


# --- ContractConformantFeedbackPayload ---------------------------------------------------------


def test_payload_is_versioned_labeled_and_evidence_bearing() -> None:
    """Scenario: every candidate is synthesized, verisim-runtime, and evidence-bearing."""
    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"), _exec("app/plugin.py"))], graph)
    assert payload.version == FEEDBACK_SCHEMA_VERSION
    assert payload.fixture_source_sha == "feed1234"
    assert payload.generated_against_db_hash == "dbhash-abc"
    assert payload.generated_against_schema_version == 7
    data = json.loads(payload.to_json())
    assert data["version"] == "verisim-feedback-v1"
    edge = data["edges"][0]
    assert edge["callerId"] == MAIN.id and edge["calleeId"] == RUN.id
    assert edge["confidence"] == "synthesized"
    assert edge["synthesizedBy"] == "verisim-runtime"
    assert edge["evidence"]["fixtureSourceSha"] == "feed1234"
    assert edge["evidence"]["fidelity"] == "full"


def test_payload_is_idempotent_and_dedupes_across_traces() -> None:
    """Scenario: the payload is byte-stable for fixed traces; the same edge appears once."""
    graph = _graph()
    traces = [
        _trace(_exec("app/main.py"), _exec("app/plugin.py"), action="t1"),
        _trace(_exec("app/main.py"), _exec("app/plugin.py"), action="t2"),  # same edge again
    ]
    p1 = build_feedback_payload(traces, graph)
    p2 = build_feedback_payload(traces, graph)
    assert p1.to_json() == p2.to_json()  # idempotent
    assert len(p1.edges) == 1  # cross-trace dedup


def test_build_does_not_write_the_database(tmp_path: Path) -> None:
    """Scenario: producing feedback leaves the fixture database content hash unchanged."""
    db = tmp_path / "call-graph.db"
    db.write_bytes(b"pretend-sqlite-bytes")
    before = hashlib.sha256(db.read_bytes()).hexdigest()

    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"), _exec("app/plugin.py"))], graph)
    out = write_feedback(payload, tmp_path / "feedback" / "fb.json")

    assert hashlib.sha256(db.read_bytes()).hexdigest() == before  # DB untouched
    assert out.exists()
    assert json.loads(out.read_text())["version"] == "verisim-feedback-v1"


def test_write_feedback_refuses_a_source_root() -> None:
    """Feedback is scratch-only — writing under the source-roots allowlist is refused."""
    from verisim.fixture import DEFAULT_SOURCE_ROOT

    payload = build_feedback_payload([_trace(_exec("app/main.py"))], _graph())
    with pytest.raises(FeedbackError, match="scratch-only"):
        write_feedback(payload, DEFAULT_SOURCE_ROOT / "verisim" / "fb.json")


# --- PayloadValidation -------------------------------------------------------------------------


def test_validator_accepts_a_well_formed_payload() -> None:
    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"), _exec("app/plugin.py"))], graph)
    validate_payload(payload, graph)  # must not raise
    validate_payload(payload.to_json(), graph)  # also accepts JSON text
    validate_payload(payload.to_dict(), graph)  # and a dict


def test_validator_rejects_an_over_claiming_unknown_node() -> None:
    """Scenario: a payload referencing a node not in the fixture graph fails with unknown-node."""
    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"), _exec("app/plugin.py"))], graph)
    data = payload.to_dict()
    data["edges"][0]["calleeId"] = "app/nowhere.py::ghost"
    with pytest.raises(FeedbackValidationError, match="unknown-node"):
        validate_payload(data, graph)


def test_validator_rejects_a_mislabeled_edge() -> None:
    """An edge claiming a direct-resolution confidence (not synthesized) is rejected."""
    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"), _exec("app/plugin.py"))], graph)
    data = payload.to_dict()
    data["edges"][0]["confidence"] = "import"
    with pytest.raises(FeedbackValidationError, match="must be 'synthesized'"):
        validate_payload(data, graph)

    data2 = payload.to_dict()
    data2["edges"][0]["synthesizedBy"] = "some-ast-rule"
    with pytest.raises(FeedbackValidationError, match="synthesizedBy"):
        validate_payload(data2, graph)


def test_validator_rejects_bad_version_and_malformed_json() -> None:
    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"))], graph)
    data = payload.to_dict()
    data["version"] = "verisim-feedback-v2"
    with pytest.raises(FeedbackValidationError, match="unsupported payload version"):
        validate_payload(data, graph)

    with pytest.raises(FeedbackValidationError, match="not valid JSON"):
        validate_payload("{not json", graph)


def test_validator_rejects_missing_fields() -> None:
    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"), _exec("app/plugin.py"))], graph)
    data = payload.to_dict()
    del data["edges"][0]["evidence"]
    with pytest.raises(FeedbackValidationError, match="missing required fields"):
        validate_payload(data, graph)


# --- RuntimeInvariantFindings (secondary architectural-invariant path) --------------------------

# An invariant forbidding ``app/main.py`` from invoking ``app/plugin.py`` — the runtime path
# ``main`` → ``run`` (which lives in ``app/plugin.py``) crosses it.
NO_MAIN_TO_PLUGIN = LayerInvariant(
    name="no_main_to_plugin",
    forbidden_caller_prefix="app/main.py",
    forbidden_callee_prefix="app/plugin.py",
    description="main must not reach the plugin layer at runtime",
)


def test_runtime_path_violating_an_invariant_becomes_one_finding() -> None:
    """Scenario: a runtime path that crosses a forbidden layer boundary yields one finding."""
    graph = _graph()
    findings = detect_findings(
        _trace(_exec("app/main.py"), _exec("app/plugin.py")), graph, [NO_MAIN_TO_PLUGIN]
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.invariant == "no_main_to_plugin"
    assert f.caller_id == MAIN.id and f.callee_id == RUN.id
    assert f.callee_name == "run"
    assert f.caller_file == "app/main.py" and f.callee_file == "app/plugin.py"
    assert f.kind == FINDING_KIND == "layer-violation"
    assert f.synthesized_by == SYNTHESIZED_BY == "verisim-runtime"
    assert f.evidence.trace_action == "run-app"
    assert f.evidence.fixture_source_sha == "feed1234"
    assert f.evidence.fidelity == FIDELITY_FULL


def test_non_violating_runtime_path_yields_no_finding() -> None:
    """A runtime path that does not cross the forbidden boundary produces no finding."""
    graph = _graph()
    # main → helper (app/util.py) is not forbidden by the no_main_to_plugin rule.
    findings = detect_findings(
        _trace(_exec("app/main.py"), _exec("app/util.py")), graph, [NO_MAIN_TO_PLUGIN]
    )
    assert findings == []


def test_finding_is_reported_even_when_edge_is_already_static() -> None:
    """A finding is about the runtime *path*, not edge novelty — reported even for a known edge.

    This is what distinguishes a finding from a missed-edge candidate: the same path that produces
    *no* candidate (because the edge is already in the static graph) still produces a finding.
    """
    graph = _graph(edges=(_edge(MAIN.id, RUN.id, "run"),))  # main→run known statically
    trace = _trace(_exec("app/main.py"), _exec("app/plugin.py"))
    candidates, _ = detect_candidates(trace, graph)
    assert candidates == []  # not re-proposed as an edge
    findings = detect_findings(trace, graph, [NO_MAIN_TO_PLUGIN])
    assert [f.callee_id for f in findings] == [RUN.id]  # but still a violation


def test_no_invariants_and_degraded_trace_yield_no_findings() -> None:
    """No declared invariants, or a degraded trace, yields no findings."""
    graph = _graph()
    assert detect_findings(_trace(_exec("app/main.py"), _exec("app/plugin.py")), graph, []) == []
    degraded = _trace(_exec("app/main.py"), _exec("app/plugin.py"), fidelity=FIDELITY_DEGRADED)
    assert detect_findings(degraded, graph, [NO_MAIN_TO_PLUGIN]) == []


def test_payload_findings_are_populated_deduped_and_idempotent() -> None:
    """Scenario: declared invariants populate ``findings[]``, deduped across traces, byte-stable."""
    graph = _graph()
    traces = [
        _trace(_exec("app/main.py"), _exec("app/plugin.py"), action="t1"),
        _trace(_exec("app/main.py"), _exec("app/plugin.py"), action="t2"),  # same violating path
    ]
    p1 = build_feedback_payload(traces, graph, invariants=[NO_MAIN_TO_PLUGIN])
    p2 = build_feedback_payload(traces, graph, invariants=[NO_MAIN_TO_PLUGIN])
    assert p1.to_json() == p2.to_json()  # idempotent
    assert len(p1.findings) == 1  # cross-trace dedup
    finding = json.loads(p1.to_json())["findings"][0]
    assert finding["invariant"] == "no_main_to_plugin"
    assert finding["calleeId"] == RUN.id
    assert finding["kind"] == "layer-violation"
    assert finding["synthesizedBy"] == "verisim-runtime"
    assert finding["evidence"]["fidelity"] == "full"


def test_payload_findings_empty_without_invariants() -> None:
    """With no declared invariants the slot stays empty — the missed-edge payload alone."""
    graph = _graph()
    payload = build_feedback_payload([_trace(_exec("app/main.py"), _exec("app/plugin.py"))], graph)
    assert payload.findings == ()
    assert json.loads(payload.to_json())["findings"] == []


def test_validator_accepts_and_guards_findings() -> None:
    """The validator accepts a well-formed findings payload and rejects over-claiming ones."""
    graph = _graph()
    payload = build_feedback_payload(
        [_trace(_exec("app/main.py"), _exec("app/plugin.py"))],
        graph,
        invariants=[NO_MAIN_TO_PLUGIN],
    )
    validate_payload(payload, graph)  # well-formed with findings → must not raise

    # an over-claiming finding referencing a node not in the fixture graph
    data = payload.to_dict()
    data["findings"][0]["calleeId"] = "app/nowhere.py::ghost"
    with pytest.raises(FeedbackValidationError, match="unknown-node"):
        validate_payload(data, graph)

    # a mislabeled finding (wrong provenance)
    data2 = payload.to_dict()
    data2["findings"][0]["synthesizedBy"] = "some-rule"
    with pytest.raises(FeedbackValidationError, match="synthesizedBy"):
        validate_payload(data2, graph)

    # a finding missing a required field
    data3 = payload.to_dict()
    del data3["findings"][0]["evidence"]
    with pytest.raises(FeedbackValidationError, match="missing required fields"):
        validate_payload(data3, graph)
