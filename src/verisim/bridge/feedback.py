"""Contract-mediated synthesized-edge feedback (OpenSpec ``add-synthesized-edge-feedback``).

The fourth and payoff link in the Verisim ↔ OpenLore prototype chain (findings doc §9.2): when
Verisim's runtime traces (Change 3) reveal a call the static call graph (Change 2) missed, emit a
**versioned, ``synthesized_by``-conformant payload** that OpenLore can ingest on its own terms —
never a direct write into OpenLore's database. Runtime reality becomes a new *source* of
synthesized edges, labeled ``synthesizedBy: 'verisim-runtime'`` and distinguished from the
AST-pattern synthesis OpenLore already does.

The honest data path has two halves; this module is the Verisim half and the local stand-in for the
other:

  1. **Verisim side (here).** Diff a :class:`~verisim.trace.model.RuntimeTrace` against the static
     :class:`~verisim.bridge.graph.CodeGraph`, resolve the runtime call sites to OpenLore node ids,
     drop anything that cannot be anchored (precision over recall, drops counted), and emit a
     :class:`FeedbackPayload` (``verisim-feedback-v1``) — idempotent, deduplicated against the
     static graph (including edges OpenLore *already* synthesized); every candidate is
     evidence-bearing.
  2. **OpenLore side (not built here).** An ingest entry point that validates the payload, maps it
     onto known nodes, and writes the edges through OpenLore's *existing* synthesized-edge path so
     every consumer sees them as ``synthesized`` with Verisim provenance — and rejects anything that
     does not resolve to known nodes. :func:`validate_payload` is the local validator that stands in
     for that ingest during the prototype's own tests; the cross-repo interface contract is
     documented in ``docs/openlore-ingest-contract.md``.

What counts as a "runtime call the static graph missed." A trace's exec stream is the dynamic fact
the static AST cannot see: which programs/scripts *actually* ran. When a fixture file is invoked
during a step (a subprocess shell-out, an interpreter run, reflection-based dispatch — the exact
blind spot static resolution has), that invocation is a runtime call edge from the step's
**entrypoint** (the first fixture node observed) to the invoked node. Only **full**-tier traces
yield candidates: a degraded trace observed no syscall stream, so it cannot witness an inter-program
call and is never mistaken for an authoritative one (the trace module's fidelity discipline, write
side). The trace's flat exec list carries no parent-pid linkage, so every later invocation is
attributed conservatively to the step entrypoint rather than guessed into a chain — honest about
what the surface can and cannot establish.

Boundaries this module holds:

  - **Never writes ``call-graph.db``.** It reads the graph and writes a *payload file*; OpenLore (or
    the validator stub) decides whether to ingest. :func:`build_feedback_payload` only reads the
    graph, so a fixture's database content hash is unchanged across feedback production.
  - **Additive + labeled only.** Every candidate is ``confidence: 'synthesized'``,
    ``synthesizedBy: 'verisim-runtime'``; a candidate never modifies or removes a resolved edge.
  - **Evidence-bearing.** Every candidate carries the runtime trace's action and the fixture source
    sha that justify it, so OpenLore (and a human) can audit *why* the edge is claimed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.fixture import DEFAULT_SOURCE_ROOT
from verisim.trace.model import RuntimeTrace

from .graph import CodeGraph, CodeNode

# The feedback payload format version. A reader keys on this and fails closed on an unknown version
# rather than mis-parsing — the bridge schema-guard discipline, carried to the write side.
FEEDBACK_SCHEMA_VERSION = "verisim-feedback-v1"

# The provenance every Verisim-emitted candidate carries. ``confidence`` reuses OpenLore's existing
# ``synthesized`` label so the edge flows through the *same* additive, provenance-labeled path as
# AST-synthesized edges; ``synthesized_by`` distinguishes the runtime source from OpenLore's own
# AST rules. ``kind`` is the OpenLore ``EdgeKind`` for a call relationship.
EDGE_CONFIDENCE = "synthesized"
SYNTHESIZED_BY = "verisim-runtime"
EDGE_KIND = "calls"


class FeedbackError(RuntimeError):
    """A feedback payload could not be produced, written, or validated safely.

    Raised loudly (the bridge/fixture discipline) rather than emitting a partial or wrong payload.
    """


class FeedbackValidationError(FeedbackError):
    """A ``verisim-feedback-v1`` payload failed validation against its schema or a fixture's nodes.

    The local stand-in for OpenLore's ingest rejection: a malformed, mislabeled, or over-claiming
    payload (one referencing a node id the fixture graph does not have) is refused, never ingested.
    """


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    """Why a candidate edge is claimed: the runtime trace that witnessed the invocation.

    Carries the trace's action and the fixture source sha (so the edge is attributable to a known
    source state), the program path the exec stream resolved to, and the trace fidelity — so a
    consumer can see the edge rests on a ``full``-tier observation, never a degraded guess.
    """

    trace_action: str
    fixture_source_sha: str | None
    exec_command: str
    fidelity: str


@dataclass(frozen=True, slots=True)
class CandidateEdge:
    """A candidate synthesized edge proposed from runtime evidence, conforming to OpenLore's
    ``CallEdge`` shape (camelCase in the payload) plus the justifying :class:`RuntimeEvidence`.

    ``line`` is ``None``: a runtime exec has no source-level call-site line (the static analyzer's
    line is exactly what is missing), recorded honestly rather than fabricated. ``call_type`` is
    ``None`` for the same reason — there is no AST call shape behind a subprocess invocation.
    """

    caller_id: str
    callee_id: str
    callee_name: str
    file: str
    line: int | None
    kind: str
    call_type: str | None
    confidence: str
    synthesized_by: str
    evidence: RuntimeEvidence

    @property
    def pair(self) -> tuple[str, str]:
        """The ``(caller_id, callee_id)`` identity used for dedup against the static graph."""
        return (self.caller_id, self.callee_id)

    def to_payload_dict(self) -> dict[str, Any]:
        """The camelCase dict OpenLore's ``CallEdge`` ingest expects, plus an ``evidence`` block."""
        return {
            "callerId": self.caller_id,
            "calleeId": self.callee_id,
            "calleeName": self.callee_name,
            "file": self.file,
            "line": self.line,
            "kind": self.kind,
            "callType": self.call_type,
            "confidence": self.confidence,
            "synthesizedBy": self.synthesized_by,
            "evidence": {
                "traceAction": self.evidence.trace_action,
                "fixtureSourceSha": self.evidence.fixture_source_sha,
                "execCommand": self.evidence.exec_command,
                "fidelity": self.evidence.fidelity,
            },
        }


@dataclass(frozen=True, slots=True)
class FeedbackPayload:
    """A versioned ``verisim-feedback-v1`` payload of candidate synthesized edges.

    ``generated_against`` pins the payload to the exact static graph state it was diffed against
    (the database content hash + schema version), so an ingest can detect a payload computed against
    a different graph. ``dropped`` is the count of runtime invocations that could not be anchored to
    a known node — surfaced, not silently discarded (precision over recall).
    """

    version: str
    fixture_source_sha: str | None
    generated_against_db_hash: str
    generated_against_schema_version: int
    edges: tuple[CandidateEdge, ...]
    dropped: int
    findings: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """The serializable payload dict (camelCase edges, stable key order)."""
        return {
            "version": self.version,
            "fixtureSourceSha": self.fixture_source_sha,
            "generatedAgainst": {
                "dbContentHash": self.generated_against_db_hash,
                "schemaVersion": self.generated_against_schema_version,
            },
            "edges": [e.to_payload_dict() for e in self.edges],
            "findings": list(self.findings),
            "dropped": self.dropped,
        }

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, stable separators) — byte-stable for a fixed input."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# --- node resolution ---------------------------------------------------------------------------


def _normalize_program(program: str, fixture_root: Path | None) -> str:
    """Reduce an exec program/script path to a fixture-relative path when it lives under the root.

    A ``full``-tier trace records absolute program paths; OpenLore node ``file_path`` values are
    repo-relative. If ``program`` is under ``fixture_root`` it is made relative; otherwise it is
    returned unchanged (an already-relative path, or a path outside the fixture that will simply not
    resolve to a node).
    """
    if fixture_root is None:
        return program
    try:
        rel = Path(program).resolve().relative_to(fixture_root.resolve())
    except (ValueError, OSError):
        return program
    return str(rel)


def _resolve_file_to_node(graph: CodeGraph, file_rel: str) -> CodeNode | None:
    """Resolve a fixture-relative file path to its representative node, or ``None``.

    Matches nodes whose ``file_path`` equals ``file_rel`` (internal nodes only — an external/stdlib
    leaf is not a fixture invocation). Among matches, prefers an ``is_entry_point`` node, else the
    node with the smallest ``start_index`` (the file's first definition) — a stable, deterministic
    representative for "this file was invoked". Fails closed (``None``) when nothing matches, so an
    unanchorable invocation is dropped, not forced.
    """
    matches = [n for n in graph.nodes if not n.is_external and n.file_path == file_rel]
    if not matches:
        return None
    entry = [n for n in matches if n.is_entry_point]
    pool = entry if entry else matches
    return min(pool, key=lambda n: n.start_index)


def _resolve_exec_to_node(
    command: str, args: tuple[str, ...], graph: CodeGraph, fixture_root: Path | None
) -> CodeNode | None:
    """Resolve one exec event to the fixture node it invoked, or ``None``.

    Tries the program path first, then each argument in order — so an interpreter invocation
    (``python3 app/main.py``: the fixture file is ``argv[1]``, not the program) resolves to the
    script, not the interpreter. Returns the first token that anchors to a known internal node.
    """
    for token in (command, *args):
        node = _resolve_file_to_node(graph, _normalize_program(token, fixture_root))
        if node is not None:
            return node
    return None


# --- discrepancy detection ---------------------------------------------------------------------


def detect_candidates(
    trace: RuntimeTrace,
    graph: CodeGraph,
    *,
    fixture_root: str | Path | None = None,
) -> tuple[list[CandidateEdge], int]:
    """Diff one runtime trace against the static graph into ``(candidates, dropped)``.

    The step **entrypoint** is the first fixture node the exec stream resolves to; every *later*
    resolved fixture node is a callee the entrypoint invoked at runtime. A candidate edge
    entrypoint→callee is proposed iff the static graph has no such edge already — including edges
    OpenLore already labeled ``synthesized`` (so an edge OpenLore has is never re-proposed). A later
    exec that cannot be anchored to a known node is **dropped and counted**, never emitted.

    Only ``full``-tier traces yield candidates: a ``degraded`` trace observed no syscall stream and
    cannot witness an inter-program call, so it returns ``([], 0)`` rather than a low-confidence
    guess.
    """
    if trace.is_degraded():
        return [], 0

    root = None if fixture_root is None else Path(fixture_root)
    static_pairs = {(e.caller_id, e.callee_id) for e in graph.edges}

    entry: CodeNode | None = None
    candidates: list[CandidateEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    dropped = 0

    for ev in trace.exec_events:
        node = _resolve_exec_to_node(ev.command, ev.args, graph, root)
        if entry is None:
            # The entrypoint itself must anchor; pre-entrypoint launchers (e.g. /bin/sh) simply do
            # not resolve and are not yet candidates.
            if node is not None:
                entry = node
            continue
        if node is None:
            # A post-entrypoint invocation we saw but cannot anchor — dropped, not emitted.
            dropped += 1
            continue
        if node.id == entry.id:
            continue
        pair = (entry.id, node.id)
        if pair in static_pairs or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        candidates.append(
            CandidateEdge(
                caller_id=entry.id,
                callee_id=node.id,
                callee_name=node.name,
                file=entry.file_path,
                line=None,
                kind=EDGE_KIND,
                call_type=None,
                confidence=EDGE_CONFIDENCE,
                synthesized_by=SYNTHESIZED_BY,
                evidence=RuntimeEvidence(
                    trace_action=trace.action_name,
                    fixture_source_sha=trace.fixture_source_sha,
                    exec_command=ev.command,
                    fidelity=trace.fidelity,
                ),
            )
        )
    return candidates, dropped


def build_feedback_payload(
    traces: list[RuntimeTrace] | tuple[RuntimeTrace, ...],
    graph: CodeGraph,
    *,
    fixture_root: str | Path | None = None,
    fixture_source_sha: str | None = None,
) -> FeedbackPayload:
    """Aggregate traces into one idempotent ``verisim-feedback-v1`` payload.

    Candidates from every trace are deduplicated by ``(caller_id, callee_id)`` (first occurrence
    wins, keeping its evidence) and sorted by that pair, so the payload is byte-stable for a fixed
    set of traces and graph. Drops accumulate across traces. This **only reads** the graph — it
    never writes ``call-graph.db``; the database content hash is unchanged across the call.
    """
    by_pair: dict[tuple[str, str], CandidateEdge] = {}
    dropped = 0
    for trace in traces:
        candidates, n_dropped = detect_candidates(trace, graph, fixture_root=fixture_root)
        dropped += n_dropped
        for cand in candidates:
            by_pair.setdefault(cand.pair, cand)

    edges = tuple(by_pair[p] for p in sorted(by_pair))
    sha = fixture_source_sha
    if sha is None:
        for trace in traces:
            if trace.fixture_source_sha is not None:
                sha = trace.fixture_source_sha
                break

    return FeedbackPayload(
        version=FEEDBACK_SCHEMA_VERSION,
        fixture_source_sha=sha,
        generated_against_db_hash=graph.db_content_hash,
        generated_against_schema_version=graph.schema_version,
        edges=edges,
        dropped=dropped,
    )


def write_feedback(payload: FeedbackPayload, path: str | Path) -> Path:
    """Write ``payload`` as canonical JSON to ``path`` (Verisim-owned scratch only).

    Refuses to write inside the source-roots allowlist (the trace/fixture isolation discipline):
    feedback is a Verisim-owned artifact, never written into a repo the user cares about — and
    emphatically never into OpenLore's database. Returns the written path.
    """
    out = Path(path)
    resolved = out.resolve()
    root = DEFAULT_SOURCE_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise FeedbackError(
            f"refusing to write feedback under a source root: {resolved} (payload is scratch-only)"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload.to_json(), encoding="utf-8")
    return out


# --- local validator (OpenLore ingest stand-in) ------------------------------------------------

_REQUIRED_TOP = ("version", "fixtureSourceSha", "generatedAgainst", "edges", "findings", "dropped")
_REQUIRED_EDGE = (
    "callerId",
    "calleeId",
    "calleeName",
    "file",
    "line",
    "kind",
    "callType",
    "confidence",
    "synthesizedBy",
    "evidence",
)


def _as_dict(payload: FeedbackPayload | dict[str, Any] | str) -> dict[str, Any]:
    """Normalize a payload (object / dict / JSON text) to a dict; fail closed on malformed JSON."""
    if isinstance(payload, FeedbackPayload):
        return payload.to_dict()
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise FeedbackValidationError(f"payload is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise FeedbackValidationError("payload JSON is not an object")
        return parsed
    return payload


def validate_payload(payload: FeedbackPayload | dict[str, Any] | str, graph: CodeGraph) -> None:
    """Validate a ``verisim-feedback-v1`` payload against its schema and a fixture's node set.

    The local stand-in for OpenLore's ingest. Checks, failing closed with an explicit
    :class:`FeedbackValidationError` on the first violation:

      - the payload is well-formed (correct version, all required top-level and per-edge fields);
      - every edge is correctly **labeled** — ``confidence == 'synthesized'`` and
        ``synthesizedBy == 'verisim-runtime'`` (an additive, Verisim-provenance edge, never an
        over-claim of a direct resolution);
      - every edge's ``callerId`` and ``calleeId`` resolve to a **known node** in ``graph`` — an
        unknown-node reference is rejected, as OpenLore would refuse an edge it cannot anchor.

    Returns ``None`` on success.
    """
    data = _as_dict(payload)

    version = data.get("version")
    if version != FEEDBACK_SCHEMA_VERSION:
        raise FeedbackValidationError(
            f"unsupported payload version {version!r}; expected {FEEDBACK_SCHEMA_VERSION!r}"
        )
    missing = [k for k in _REQUIRED_TOP if k not in data]
    if missing:
        raise FeedbackValidationError(f"payload missing required fields: {missing}")
    if not isinstance(data["edges"], list):
        raise FeedbackValidationError("payload 'edges' must be a list")

    node_ids = {n.id for n in graph.nodes}
    for i, edge in enumerate(data["edges"]):
        if not isinstance(edge, dict):
            raise FeedbackValidationError(f"edge[{i}] is not an object")
        absent = [k for k in _REQUIRED_EDGE if k not in edge]
        if absent:
            raise FeedbackValidationError(f"edge[{i}] missing required fields: {absent}")
        if edge["confidence"] != EDGE_CONFIDENCE:
            raise FeedbackValidationError(
                f"edge[{i}] confidence {edge['confidence']!r} must be {EDGE_CONFIDENCE!r} "
                "(candidates are always additive synthesized edges)"
            )
        if edge["synthesizedBy"] != SYNTHESIZED_BY:
            raise FeedbackValidationError(
                f"edge[{i}] synthesizedBy {edge['synthesizedBy']!r} must be {SYNTHESIZED_BY!r}"
            )
        for endpoint in ("callerId", "calleeId"):
            if edge[endpoint] not in node_ids:
                raise FeedbackValidationError(
                    f"edge[{i}] {endpoint} {edge[endpoint]!r} is not a known node in the fixture "
                    "graph (unknown-node)"
                )
