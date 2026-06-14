"""The headless CD pipeline: intent → simulate → trace → evaluate → feedback → breaker → prepare.

The sixth and final link in the Verisim ↔ OpenLore prototype chain (findings doc §9.1). One local,
network-isolated entry point chains the whole stack against a de-fanged fixture (Change 1) and stops
*before* the one irreversible step — leaving the commit/push/deploy as an explicitly human-confirmed
action that, even confirmed, can only touch the fixture (which cannot reach the original).

The stages run in a fixed order; each writes a typed artifact and appends a :class:`StageResult`:

  1. **intent_graph** — load the read-only :class:`~verisim.bridge.graph.CodeGraph` (Change 2) and
     build the run's intent (verifying steps + declared invariants).
  2. **speculative** — run the loop's speculative-rollout machinery (``loop/speculative.py``) over
     the intent's actions. The prototype wires the oracle itself as the drafter — a perfect,
     offline, deterministic stand-in for M_θ — because this entry point tests *composition* (order,
     gating, artifacts), not the research H_ε quantities, which live in SPEC-13/19.
  3. **trace** — execute the verifying steps through the tracing oracle (Change 3), threading the
     state forward into the real trajectory the breaker watches.
  4. **evaluate** — check the static + runtime-discovered call edges against the declared
     architectural invariants → findings.
  5. **feedback** — emit the ``verisim-feedback-v1`` payload (Change 4) for runtime-found edges.
  6. **breaker** — the oscillation breaker (Change 5) watches the whole trajectory; a ``critical``
     trip halts the run before delivery and surfaces a human-gated rollback recommendation.
  7. **prepare_delivery** — a content-sealed report plus a prepared, *unpushed* commit/patch on the
     fixture. Reachable only after every prior stage passes; a commit happens only with explicit
     confirmation and is never pushed (the de-fang makes a push impossible by construction).

Fail-safe ordering is the spine: a failed or halted stage marks every later stage ``skipped`` and
nothing irreversible is ever reached without passing every prior gate and a human confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verisim.bridge.feedback import build_feedback_payload, validate_payload, write_feedback
from verisim.bridge.graph import (
    BridgeError,
    CodeGraph,
    analysis_db_path,
    load_code_graph,
)
from verisim.env.action import Action, parse_action
from verisim.env.state import State
from verisim.fixture import Fixture, tree_hash
from verisim.loop.speculative import speculative_rollout
from verisim.oracle.base import Oracle
from verisim.oracle.differential import canonical_world
from verisim.safety import (
    TIER_CRITICAL,
    BreakerConfig,
    PlanningTransition,
    SpeculativeRollout,
    TrajectoryBreaker,
)
from verisim.trace.model import RuntimeTrace
from verisim.trace.oracle import TracingOracle
from verisim.trace.tracer import DegradedTracer, Tracer

from .model import (
    REPORT_SCHEMA_VERSION,
    SOURCE_RUNTIME,
    SOURCE_STATIC,
    STAGE_FAILED,
    STAGE_HALTED,
    STAGE_OK,
    STAGE_SKIPPED,
    Intent,
    InvariantFinding,
    PreparedDelivery,
    RunReport,
    StageResult,
    _Accumulator,
)

# Where the run's typed artifacts live: a sibling of ``repo/`` under the fixture root, so they are
# Verisim-owned scratch and never enter the subject tree's hash (mirroring the breaker's snapshot
# discipline).
RUN_DIRNAME = ".verisim-run"


class PipelineError(RuntimeError):
    """A pipeline run could not be set up safely (e.g. an unparsable action in the intent)."""


def _state_label(state: State) -> str:
    """A stable, opaque label for a state — the breaker compares these for equality only."""
    return hashlib.sha256(canonical_world(state).encode("utf-8")).hexdigest()[:16]


def _default_oracle() -> Oracle:
    """The real sandbox oracle when a POSIX shell is present, else the hermetic reference oracle.

    Both are deterministic; the sandbox is the real execution surface the trace stage is meant to
    observe (Change 3), and the reference oracle is the disclosed fallback on a host with no shell.
    """
    from verisim.oracle.reference import ReferenceOracle
    from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

    try:
        return SandboxOracle()
    except SystemOracleUnavailable:  # pragma: no cover - only on a host with no shell
        return ReferenceOracle()


def _parse_actions(intent: Intent) -> list[Action]:
    try:
        return [parse_action(a) for a in intent.actions]
    except Exception as exc:  # ParseError and friends
        raise PipelineError(f"intent has an unparsable action: {exc}") from exc


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in the fixture repo. Never passes ``--no-verify`` — the commit gate is honored."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=check,
        capture_output=True,
        text=True,
    )


# --- stages ------------------------------------------------------------------------------------


def _stage_intent_graph(
    intent: Intent, fixture: Fixture, graph: CodeGraph | None, run_dir: Path
) -> tuple[StageResult, CodeGraph]:
    if graph is None:
        db = analysis_db_path(fixture.repo_path)
        if not db.exists():
            raise BridgeError(
                f"no OpenLore call-graph database for the fixture: {db} "
                "(pass graph=... or run OpenLore analyze on the fixture)"
            )
        graph = load_code_graph(db, source_path=fixture.repo_path)
    artifact = _write_artifact(
        run_dir,
        "01-intent-graph.json",
        {
            "intent": intent.to_dict(),
            "graph": {
                "schemaVersion": graph.schema_version,
                "dbContentHash": graph.db_content_hash,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            },
        },
    )
    summary = {
        "actions": len(intent.actions),
        "invariants": len(intent.invariants),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
    }
    return StageResult("intent_graph", STAGE_OK, summary, artifact), graph


def _stage_speculative(
    actions: list[Action], oracle: Oracle, run_dir: Path
) -> tuple[StageResult, SpeculativeRollout]:
    """Run the speculative-rollout machinery with the oracle as a perfect drafter.

    Also populate the in-memory :class:`SpeculativeRollout` (the breaker's droppable rollout) with
    the imagined state labels — so a ``critical`` trip has something cheap and reversible to drop.
    """

    def draft(state: State, action: Action, _i: int, _variant: int) -> State:
        return oracle.step(state, action).state

    def oracle_step(state: State, action: Action) -> State:
        return oracle.step(state, action).state

    def diverge(truth: State, drafted: State) -> float:
        return 0.0 if canonical_world(truth) == canonical_world(drafted) else 1.0

    rollout = SpeculativeRollout()
    if actions:
        record = speculative_rollout(
            State.empty(),
            actions,
            draft,
            oracle_step,
            diverge,
            k=min(4, len(actions)),
            epsilon=0.0,
        )
        # Re-derive the imagined trajectory labels for the droppable rollout (cheap, deterministic).
        state = State.empty()
        for action in actions:
            state = oracle.step(state, action).state
            rollout.states.append(_state_label(state))
        summary: dict[str, Any] = {
            "oracleCalls": record.oracle_calls,
            "totalSteps": record.total_steps,
            "faithfulSteps": record.faithful_steps,
            "corrections": record.corrections,
        }
        artifact_body: dict[str, Any] = {
            **summary,
            "acceptedPrefixes": list(record.accepted_prefixes),
            "windowLengths": list(record.window_lengths),
            "imaginedStates": list(rollout.states),
        }
    else:
        summary = {"oracleCalls": 0, "totalSteps": 0, "faithfulSteps": 0, "corrections": 0}
        artifact_body = {
            **summary,
            "acceptedPrefixes": [],
            "windowLengths": [],
            "imaginedStates": [],
        }

    artifact = _write_artifact(run_dir, "02-speculative.json", artifact_body)
    return StageResult("speculative", STAGE_OK, summary, artifact), rollout


def _stage_trace(
    actions: list[Action],
    oracle: Oracle,
    tracer: Tracer,
    fixture: Fixture,
    breaker: TrajectoryBreaker,
    run_dir: Path,
) -> tuple[StageResult, list[RuntimeTrace]]:
    """Trace the verifying steps, threading state forward, and feed each transition to the breaker.

    Stops the moment the breaker freezes (a ``critical`` trip) — the loop is halted, the rest of the
    trajectory is never executed. The trace artifact is a *structural* projection (no wall-clock
    ``elapsed_s``, no volatile syscall argument text) so it is byte-deterministic for a fixed run.
    """
    traced = TracingOracle(
        oracle, tracer=tracer, fixture_source_sha=fixture.manifest.source_head_sha
    )
    state = State.empty()
    trace_records: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for action in actions:
        before = state
        result = traced.step(state, action)
        trace = traced.traces[-1]
        files = tuple(m.path for m in trace.file_mutations)
        transition = PlanningTransition(
            from_state=_state_label(before),
            to_state=_state_label(result.state),
            files_modified=files,
        )
        breaker.observe(transition)
        state = result.state
        trace_records.append(_project_trace(trace))
        transitions.append(
            {"from": transition.from_state, "to": transition.to_state, "files": list(files)}
        )
        if breaker.frozen:
            break

    artifact = _write_artifact(
        run_dir,
        "03-trace.json",
        {"traces": trace_records, "transitions": transitions},
    )
    summary = {
        "steps": len(trace_records),
        "transitions": len(transitions),
        "breakerStatus": breaker.status,
    }
    return StageResult("trace", STAGE_OK, summary, artifact), list(traced.traces)


def _stage_evaluate(
    intent: Intent, graph: CodeGraph, traces: list[RuntimeTrace], fixture: Fixture, run_dir: Path
) -> tuple[StageResult, list[InvariantFinding], Any]:
    """Evaluate the declared invariants against static + runtime-discovered edges.

    The runtime edges come from diffing the traces against the graph (Change 4); the payload built
    here is reused (written + validated) by the feedback stage, so the graph is read once.
    """
    payload = build_feedback_payload(
        traces,
        graph,
        fixture_root=fixture.repo_path,
        fixture_source_sha=fixture.manifest.source_head_sha,
    )
    by_id = {n.id: n for n in graph.nodes}
    findings: list[InvariantFinding] = []

    def check(caller_id: str, callee_id: str, source: str) -> None:
        caller, callee = by_id.get(caller_id), by_id.get(callee_id)
        if caller is None or callee is None:
            return
        for inv in intent.invariants:
            if caller.file_path.startswith(
                inv.forbidden_caller_prefix
            ) and callee.file_path.startswith(inv.forbidden_callee_prefix):
                findings.append(
                    InvariantFinding(
                        invariant=inv.name,
                        caller_id=caller_id,
                        callee_id=callee_id,
                        caller_file=caller.file_path,
                        callee_file=callee.file_path,
                        source=source,
                    )
                )

    for edge in graph.edges:
        check(edge.caller_id, edge.callee_id, SOURCE_STATIC)
    for cand in payload.edges:
        check(cand.caller_id, cand.callee_id, SOURCE_RUNTIME)

    findings.sort(key=lambda f: (f.invariant, f.source, f.caller_id, f.callee_id))
    artifact = _write_artifact(
        run_dir, "04-evaluate.json", {"findings": [f.to_dict() for f in findings]}
    )
    summary = {
        "findings": len(findings),
        "static": sum(1 for f in findings if f.source == SOURCE_STATIC),
        "runtime": sum(1 for f in findings if f.source == SOURCE_RUNTIME),
    }
    return StageResult("evaluate", STAGE_OK, summary, artifact), findings, payload


def _stage_feedback(payload: Any, graph: CodeGraph, run_dir: Path) -> StageResult:
    """Emit and validate the ``verisim-feedback-v1`` payload (never writes ``call-graph.db``)."""
    path = write_feedback(payload, run_dir / "05-feedback.json")
    validate_payload(payload, graph)  # the local stand-in for OpenLore's ingest; fails closed
    summary = {"edges": len(payload.edges), "dropped": payload.dropped}
    return StageResult("feedback", STAGE_OK, summary, path.name)


def _stage_breaker(
    breaker: TrajectoryBreaker, run_dir: Path
) -> tuple[StageResult, bool, str | None]:
    """The breaker gate: a ``critical`` status halts the run before delivery."""
    rec = breaker.recommendation
    body: dict[str, Any] = {
        "status": breaker.status,
        "tripped": breaker.tripped,
        "frozen": breaker.frozen,
        "transitions": len(breaker.transitions),
        "rollbackRecommendation": None if rec is None else _recommendation_dict(rec),
    }
    artifact = _write_artifact(run_dir, "06-breaker.json", body)
    summary = {"status": breaker.status, "tripped": breaker.tripped}
    if breaker.status == TIER_CRITICAL:
        reason = "oscillation breaker tripped critical; run halted before delivery"
        return StageResult("breaker", STAGE_HALTED, summary, artifact), True, reason
    return StageResult("breaker", STAGE_OK, summary, artifact), False, None


def _stage_prepare_delivery(
    intent: Intent, fixture: Fixture, confirm: bool, run_dir: Path
) -> tuple[StageResult, PreparedDelivery]:
    """Apply the intent's change, stage it, write a patch, and (only if confirmed) commit it.

    The commit is never pushed: the de-fanged fixture has no remote and a blocking pre-push hook, so
    a push is impossible by construction. ``git commit`` is run plainly — never ``--no-verify`` — so
    the fixture's commit gate (if any) is honored, never bypassed.
    """
    repo = fixture.repo_path
    for rel, content in intent.change:
        dest = repo / rel
        if not dest.resolve().is_relative_to(repo.resolve()):
            raise PipelineError(f"refusing to write a change outside the fixture repo: {rel!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    _git(repo, "add", "-A")
    patch = _git(repo, "diff", "--cached").stdout
    patch_path = run_dir / "07-delivery.patch"
    patch_path.write_text(patch, encoding="utf-8")
    names = _git(repo, "diff", "--cached", "--name-only").stdout.split()

    committed = False
    commit_sha: str | None = None
    if confirm:
        _git(repo, "commit", "-m", f"verisim-cd: {intent.goal}")
        commit_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        committed = True

    delivery = PreparedDelivery(
        prepared=True,
        committed=committed,
        commit_sha=commit_sha,
        patch_artifact=patch_path.name,
        files_changed=tuple(names),
    )
    artifact = _write_artifact(run_dir, "07-delivery.json", delivery.to_dict())
    summary = {"filesChanged": len(names), "committed": committed}
    return StageResult("prepare_delivery", STAGE_OK, summary, artifact), delivery


# --- orchestration -----------------------------------------------------------------------------


def run_pipeline(
    fixture: Fixture,
    intent: Intent,
    *,
    graph: CodeGraph | None = None,
    oracle: Oracle | None = None,
    tracer: Tracer | None = None,
    run_dir: str | Path | None = None,
    confirm_delivery: bool = False,
    breaker_config: BreakerConfig | None = None,
) -> RunReport:
    """Run the headless CD pipeline end to end and return a content-sealed :class:`RunReport`.

    Deterministic for a fixed ``(fixture revision, intent, seed)`` with the default (degraded)
    tracer. ``confirm_delivery`` is the single human gate to the only irreversible step; left
    ``False`` (the default) the run prepares and stops. Never reaches the network and never touches
    the original source — only the de-fanged fixture.
    """
    oracle = oracle or _default_oracle()
    tracer = tracer or DegradedTracer()
    out_dir = Path(run_dir) if run_dir is not None else fixture.root / RUN_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = _parse_actions(intent)
    breaker = TrajectoryBreaker(fixture, breaker_config)

    acc = _Accumulator()
    g: CodeGraph | None = graph
    traces: list[RuntimeTrace] = []
    findings: list[InvariantFinding] = []
    payload: Any = None
    delivery: PreparedDelivery | None = None
    rollback: dict[str, Any] | None = None

    # Each stage runs only if no earlier stage halted/failed; an exception marks the stage FAILED
    # (fail-safe ordering), and the breaker stage may HALT the run before delivery.
    def guarded(name: str, fn: Callable[[], StageResult]) -> StageResult:
        if acc.halted:
            sr = StageResult(name, STAGE_SKIPPED, {}, None)
            acc.stages.append(sr)
            return sr
        try:
            sr = fn()
        except Exception as exc:
            sr = StageResult(name, STAGE_FAILED, {"error": f"{type(exc).__name__}: {exc}"}, None)
            acc.halted = True
            acc.halt_reason = f"stage {name!r} failed: {exc}"
        acc.stages.append(sr)
        return sr

    def s1() -> StageResult:
        nonlocal g
        sr, g = _stage_intent_graph(intent, fixture, graph, out_dir)
        return sr

    def s2() -> StageResult:
        sr, rollout = _stage_speculative(actions, oracle, out_dir)
        breaker.rollout = rollout
        return sr

    def s3() -> StageResult:
        nonlocal traces
        assert tracer is not None
        sr, traces = _stage_trace(actions, oracle, tracer, fixture, breaker, out_dir)
        return sr

    def s4() -> StageResult:
        nonlocal findings, payload
        assert g is not None
        sr, findings, payload = _stage_evaluate(intent, g, traces, fixture, out_dir)
        return sr

    def s5() -> StageResult:
        assert g is not None and payload is not None
        return _stage_feedback(payload, g, out_dir)

    def s6() -> StageResult:
        nonlocal rollback
        sr, halt, reason = _stage_breaker(breaker, out_dir)
        if halt:
            acc.halted = True
            acc.halt_reason = reason
            rec = breaker.recommendation
            rollback = None if rec is None else _recommendation_dict(rec)
        return sr

    def s7() -> StageResult:
        nonlocal delivery
        sr, delivery = _stage_prepare_delivery(intent, fixture, confirm_delivery, out_dir)
        return sr

    guarded("intent_graph", s1)
    guarded("speculative", s2)
    guarded("trace", s3)
    guarded("evaluate", s4)
    guarded("feedback", s5)
    guarded("breaker", s6)
    guarded("prepare_delivery", s7)

    network_isolated = _network_isolated(oracle)
    report = RunReport(
        version=REPORT_SCHEMA_VERSION,
        goal=intent.goal,
        seed=intent.seed,
        fixture_source_sha=fixture.manifest.source_head_sha,
        fixture_repo_tree_hash=tree_hash(fixture.repo_path),
        stages=tuple(acc.stages),
        halted=acc.halted,
        halt_reason=acc.halt_reason,
        breaker_status=breaker.status,
        rollback_recommendation=rollback,
        findings=tuple(findings),
        feedback_edges=0 if payload is None else len(payload.edges),
        feedback_dropped=0 if payload is None else payload.dropped,
        network_isolated=network_isolated,
        prepared_delivery=delivery,
    ).signed()
    (out_dir / "run-report.json").write_text(report.to_json(), encoding="utf-8")
    return report


# --- helpers -----------------------------------------------------------------------------------


def _network_isolated(oracle: Oracle) -> bool:
    """Attest the run required no outbound network: the v0 grammar exposes none and the sandbox
    oracle blocks egress by allowlist. Honors whatever the oracle attests; defaults to ``True``
    for the hermetic reference oracle, which has no network surface at all."""
    herm = getattr(oracle, "hermeticity", None)
    if callable(herm):
        try:
            return bool(herm().network_blocked)
        except Exception:  # pragma: no cover - attestation is best-effort
            return True
    return True


def _project_trace(trace: RuntimeTrace) -> dict[str, Any]:
    """The determinism-stable structural projection of a trace: every typed fact except the
    wall-clock ``elapsed_s`` and the volatile syscall argument/result text (kept out of the
    artifact so a full-tier run stays byte-stable on syscall counts, not on addresses)."""
    return {
        "schemaVersion": trace.schema_version,
        "fidelity": trace.fidelity,
        "action": trace.action_name,
        "args": list(trace.action_args),
        "exitCode": trace.exit_code,
        "exec": [
            {"command": e.command, "args": list(e.args), "exitCode": e.exit_code}
            for e in trace.exec_events
        ],
        "fileMutations": [
            {"path": m.path, "kind": m.kind, "mode": m.mode} for m in trace.file_mutations
        ],
        "netEvents": [{"kind": n.kind, "target": n.target} for n in trace.net_events],
        "syscalls": [s.name for s in trace.syscall_events],
    }


def _recommendation_dict(rec: Any) -> dict[str, Any]:
    return {
        "targetLabel": rec.target_label,
        "targetTreeHash": rec.target_tree_hash,
        "currentTreeHash": rec.current_tree_hash,
        "diffPreview": list(rec.diff_preview),
        "reason": rec.reason,
    }


def _write_artifact(run_dir: Path, name: str, body: dict[str, Any]) -> str:
    """Write a canonical-JSON stage artifact under the run dir; return its relative name."""
    (run_dir / name).write_text(
        json.dumps(body, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return name


# --- console entry point -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``verisim-cd`` — materialize a fixture from a source repo and run the pipeline against it.

    Prepares and stops by default; ``--confirm`` performs the (fixture-only, unpushed) commit.
    Reads no network and writes only Verisim-owned scratch and the de-fanged fixture.
    """
    from verisim.fixture import FixtureConfig, materialize, teardown

    parser = argparse.ArgumentParser(prog="verisim-cd", description=__doc__)
    parser.add_argument("source", help="path to a git repo to materialize as the fixture")
    parser.add_argument("--goal", default="headless CD run", help="the run's intent goal")
    parser.add_argument(
        "--action", action="append", default=[], help="a v0-grammar verifying step (repeatable)"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="commit the prepared change (fixture-only, unpushed)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", action="store_true", help="keep the fixture after the run")
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    cfg = FixtureConfig(source_roots=(source.parent,))
    fixture = materialize(source, cfg)
    try:
        intent = Intent(goal=args.goal, actions=tuple(args.action), seed=args.seed)
        report = run_pipeline(fixture, intent, confirm_delivery=args.confirm)
    finally:
        if not args.keep:
            teardown(fixture)

    print(report.to_json())
    return 1 if report.halted else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
