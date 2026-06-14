"""Read-only OpenLore call-graph adapter — the typed read model and SQLite loader.

The second link in the Verisim ↔ OpenLore prototype chain (findings doc §9, OpenSpec
``add-openlore-graph-adapter``). OpenLore already produces the artifact the prototype needs: a
deterministic, locally computed **static call graph** of a real codebase, persisted in
``.openlore/analysis/call-graph.db``. The research claim is that *runtime reality can correct this
static picture*; to do that, Verisim must first **read** the static picture faithfully — and
faithfully here means **read-only and lossless**.

This module loads that graph into a typed in-memory :class:`CodeGraph` under three hard contracts
the proposal pins:

  - **One-way / read-only.** The database is OpenLore's regenerable cache, not Verisim's store of
    record. The loader opens it ``mode=ro`` (SQLite never writes the main database file in
    read-only mode) and the database's content hash is **unchanged across any load** — the read
    side of the SPEC-11 hermeticity-by-construction discipline.
  - **Schema-pinned.** Raw SQL is OpenLore's *internal, versioned* schema. The loader checks
    ``schema_version`` against a supported range and **fails closed** (:class:`SchemaVersionError`)
    on a mismatch rather than best-effort-parsing an unknown layout into wrong data.
  - **Provenance-preserving.** Every edge keeps its ``confidence`` label
    (``import``/``type_inference``/``same_file``/``name_only``/``synthesized``/…) and its
    ``synthesized_by`` rule verbatim, so Change 4 can tell a *directly resolved* edge from one
    OpenLore *already synthesized* and never re-propose an edge OpenLore already has.

Staleness is tracked the honest way: a loaded :class:`CodeGraph` records the **source tree hash**
it was read against (reusing the fixture's content-hash machinery), and
:meth:`CodeGraph.is_stale_against` recomputes it — so a regenerated database (or any edit to the
fixture) invalidates the cached graph instead of silently serving stale structure. The OpenLore
``fingerprint.json`` hash is recorded too, for traceability.

The function call graph with per-edge provenance lives **only** in the SQLite ``edges`` table: the
supported MCP surface (``get_call_graph``) returns an *aggregate summary* (hubs, entry points,
counts), not the provenance-bearing edge set — so the SQLite read is the canonical loader, and the
MCP surface is used in :mod:`verisim.bridge.analysis` as an independent **cross-check** on the
aggregates the supported contract does expose (the honest form of the proposal's MCP/SQL parity).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from verisim.fixture import DEFAULT_EXCLUDE, tree_hash

# The OpenLore analysis database, relative to a repository root. OpenLore authors and regenerates
# it on ``analyze``; Verisim only ever reads it.
ANALYSIS_DB_RELPATH = Path(".openlore") / "analysis" / "call-graph.db"

# The OpenLore content fingerprint of the analyzed tree, recorded for traceability.
FINGERPRINT_RELPATH = Path(".openlore") / "analysis" / "fingerprint.json"

# Supported ``schema_version`` range (inclusive). Observed in the wild: v5 (the installed CLI) and
# v7 (older committed analyses). A version outside this range fails closed — the adapter reads a
# known layout or refuses, never a best-effort parse of an unknown one.
MIN_SCHEMA_VERSION = 5
MAX_SCHEMA_VERSION = 7

# The fixture content-hash machinery excludes ``.git`` and caches; for a source fingerprint we must
# also exclude OpenLore's own ``.openlore/`` cache, which is regenerated on every analyze and would
# otherwise make a freshly loaded graph perpetually "stale" against its own source.
_SOURCE_EXCLUDE = DEFAULT_EXCLUDE | {".openlore"}


class BridgeError(RuntimeError):
    """An OpenLore call graph could not be read faithfully.

    Raised loudly (the fixture module's discipline) rather than returning a partial or wrong graph:
    a missing database, an unsupported schema, or a staleness query on a graph with no source
    anchor.
    """


class SchemaVersionError(BridgeError):
    """The database ``schema_version`` is outside the supported range.

    Fails closed with an actionable message (re-run OpenLore analyze, or the adapter needs an
    update) and yields **no** partial graph — an unknown schema is never parsed best-effort.
    """


@dataclass(frozen=True, slots=True)
class CodeNode:
    """A function/method node from OpenLore's ``nodes`` table, fields preserved verbatim."""

    id: str
    name: str
    file_path: str
    class_name: str | None
    language: str
    is_async: bool
    fan_in: int
    fan_out: int
    is_external: bool
    external_kind: str | None
    is_hub: bool
    is_entry_point: bool
    start_index: int
    end_index: int
    signature: str | None
    docstring: str | None


@dataclass(frozen=True, slots=True)
class CallEdge:
    """A call edge from OpenLore's ``edges`` table.

    The provenance fields (``confidence``, ``kind``, ``call_type``, ``synthesized_by``) are kept
    exactly as stored — no normalization — so downstream code can distinguish a directly resolved
    edge from one OpenLore already synthesized (and which rule synthesized it).
    """

    caller_id: str
    caller_file: str
    callee_id: str
    callee_file: str | None
    callee_name: str
    line: int | None
    confidence: str | None
    kind: str | None
    call_type: str | None
    synthesized_by: str | None

    @property
    def is_synthesized(self) -> bool:
        """True iff OpenLore synthesized this edge (vs. resolved it directly)."""
        return self.confidence == "synthesized" or self.synthesized_by is not None


@dataclass(frozen=True, slots=True)
class CodeClass:
    """A class/module from OpenLore's ``classes`` table; JSON list columns parsed to tuples."""

    id: str
    name: str
    file_path: str
    language: str
    parent_classes: tuple[str, ...]
    interfaces: tuple[str, ...]
    method_ids: tuple[str, ...]
    fan_in: int
    fan_out: int
    is_module: bool


@dataclass(frozen=True, slots=True)
class CfgEntry:
    """A control-flow-graph overlay entry (``cfg_overlay`` table); ``cfg`` kept as raw JSON text."""

    function_id: str
    file_path: str
    cfg: str


@dataclass(frozen=True, slots=True)
class CodeGraph:
    """A typed, read-only in-memory OpenLore call graph plus the provenance to re-derive it.

    ``schema_version`` and ``db_content_hash`` pin the read to an exact database state;
    ``source_tree_hash`` (and the OpenLore ``fingerprint``) anchor it to the source it describes,
    so :meth:`is_stale_against` can detect a regenerated database. ``db_file_hashes`` is OpenLore's
    own ``file_hashes`` table verbatim (often empty — OpenLore populates it only on some runs), kept
    for traceability but **not** relied on for staleness.
    """

    nodes: tuple[CodeNode, ...]
    edges: tuple[CallEdge, ...]
    classes: tuple[CodeClass, ...]
    cfg: tuple[CfgEntry, ...]
    schema_version: int
    db_path: str
    db_content_hash: str
    source_path: str | None
    source_tree_hash: str | None
    fingerprint: str | None
    db_file_hashes: Mapping[str, str] = field(default_factory=dict)

    def node_by_id(self, node_id: str) -> CodeNode | None:
        """The node with ``node_id``, or ``None`` (linear scan; graphs here are small)."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def synthesized_edges(self) -> tuple[CallEdge, ...]:
        """Edges OpenLore synthesized — the ones Change 4 must never re-propose."""
        return tuple(e for e in self.edges if e.is_synthesized)

    def entry_point_ids(self) -> frozenset[str]:
        """Ids of nodes OpenLore flagged as entry points (no internal callers)."""
        return frozenset(n.id for n in self.nodes if n.is_entry_point)

    def internal_nodes(self) -> tuple[CodeNode, ...]:
        """Nodes defined *in* the codebase (``is_external`` false) — excludes call targets in
        third-party / stdlib code. OpenLore's supported ``get_call_graph`` summary counts these."""
        return tuple(n for n in self.nodes if not n.is_external)

    def internal_entry_point_ids(self) -> frozenset[str]:
        """Ids of internal nodes flagged as entry points — the set the supported surface reports."""
        return frozenset(n.id for n in self.nodes if n.is_entry_point and not n.is_external)

    def internal_edges_named(self) -> frozenset[tuple[str, str, str, str]]:
        """Internal call edges as ``(caller_file, caller_name, callee_file, callee_name)`` tuples.

        Both endpoints are internal nodes (external call targets excluded), matching the shape of
        OpenLore's real-graph ``get_subgraph`` surface so the SQLite read and the MCP real edges can
        be compared directly (``McpEdge.as_pair``). Deduplicated to a set — call-site multiplicity
        is not part of the identity."""
        by_id = {n.id: n for n in self.nodes}
        internal = {n.id for n in self.nodes if not n.is_external}
        return frozenset(
            (
                by_id[e.caller_id].file_path,
                by_id[e.caller_id].name,
                by_id[e.callee_id].file_path,
                by_id[e.callee_id].name,
            )
            for e in self.edges
            if e.caller_id in internal and e.callee_id in internal
        )

    def is_stale_against(self, repo_path: str | Path) -> bool:
        """True iff the fixture's current source no longer matches what this graph was read at.

        Recomputes the source tree hash (excluding ``.git``, caches, and ``.openlore/``) and
        compares to the recorded one. Any edit to the fixture — including the re-analysis that
        regenerates the database — flips this to ``True``, so a stale graph is reloaded rather than
        served. Raises :class:`BridgeError` if the graph was loaded without a source anchor.
        """
        if self.source_tree_hash is None:
            raise BridgeError(
                "cannot check staleness: this CodeGraph was loaded without a source_path anchor"
            )
        return tree_hash(Path(repo_path), _SOURCE_EXCLUDE) != self.source_tree_hash


def analysis_db_path(repo_path: str | Path) -> Path:
    """The path to a repository's OpenLore call-graph database (may not yet exist)."""
    return Path(repo_path) / ANALYSIS_DB_RELPATH


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` read-only. SQLite never writes the main database file in ``mode=ro``,
    so the file's content hash is unchanged across the load (the read-only contract)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _read_schema_version(con: sqlite3.Connection) -> int:
    """Read and validate ``schema_version``; fail closed on an unsupported (or absent) version."""
    try:
        row = con.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    except sqlite3.OperationalError as exc:  # no schema_version table at all
        raise SchemaVersionError(
            "database has no schema_version table; not an OpenLore call-graph.db "
            "(re-run OpenLore analyze)"
        ) from exc
    if row is None:
        raise SchemaVersionError("database schema_version table is empty; re-run OpenLore analyze")
    version = int(row["version"])
    if not MIN_SCHEMA_VERSION <= version <= MAX_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"unsupported OpenLore schema_version {version}; this adapter supports "
            f"{MIN_SCHEMA_VERSION}-{MAX_SCHEMA_VERSION}. Re-run OpenLore analyze, or update the "
            "adapter."
        )
    return version


def _parse_json_list(raw: str | None) -> tuple[str, ...]:
    """Parse a JSON string-list column (``parent_classes``/``interfaces``/``method_ids``) to tuple.

    OpenLore stores these as JSON text defaulting to ``'[]'``; a malformed or non-list value
    degrades to an empty tuple rather than raising — the column is metadata, not the contract.
    """
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(x) for x in parsed)


class _Row:
    """A column-defensive view of a SQLite row.

    OpenLore's schema evolved within the supported range (e.g. ``edges.synthesized_by`` exists in v7
    but not v5), so reads tolerate a *known* in-range column being absent — degrading to a default
    rather than raising. (An *unknown* schema is the schema guard's job, not this view's.) The
    column set is computed once per table from the cursor description.
    """

    __slots__ = ("_cols", "_row")

    def __init__(self, row: sqlite3.Row, cols: frozenset[str]) -> None:
        self._row = row
        self._cols = cols

    def _raw(self, key: str) -> object:
        return self._row[key] if key in self._cols else None

    def s(self, key: str) -> str:
        v = self._raw(key)
        return "" if v is None else str(v)

    def os(self, key: str) -> str | None:
        v = self._raw(key)
        return None if v is None else str(v)

    def i(self, key: str) -> int:
        v = self._raw(key)
        return 0 if v is None else int(v) if isinstance(v, (int, float, str)) else 0

    def oi(self, key: str) -> int | None:
        v = self._raw(key)
        return None if v is None else int(v) if isinstance(v, (int, float, str)) else None

    def b(self, key: str) -> bool:
        return bool(self._raw(key))


def _rows(con: sqlite3.Connection, table: str) -> list[_Row]:
    # ``table`` is a module-internal literal (never user input).
    cur = con.execute(f"SELECT * FROM {table}")
    cols = frozenset(d[0] for d in cur.description)
    return [_Row(r, cols) for r in cur.fetchall()]


def _load_nodes(con: sqlite3.Connection) -> tuple[CodeNode, ...]:
    return tuple(
        CodeNode(
            id=r.s("id"),
            name=r.s("name"),
            file_path=r.s("file_path"),
            class_name=r.os("class_name"),
            language=r.s("language"),
            is_async=r.b("is_async"),
            fan_in=r.i("fan_in"),
            fan_out=r.i("fan_out"),
            is_external=r.b("is_external"),
            external_kind=r.os("external_kind"),
            is_hub=r.b("is_hub"),
            is_entry_point=r.b("is_entry_point"),
            start_index=r.i("start_index"),
            end_index=r.i("end_index"),
            signature=r.os("signature"),
            docstring=r.os("docstring"),
        )
        for r in _rows(con, "nodes")
    )


def _load_edges(con: sqlite3.Connection) -> tuple[CallEdge, ...]:
    return tuple(
        CallEdge(
            caller_id=r.s("caller_id"),
            caller_file=r.s("caller_file"),
            callee_id=r.s("callee_id"),
            callee_file=r.os("callee_file"),
            callee_name=r.s("callee_name"),
            line=r.oi("line"),
            confidence=r.os("confidence"),
            kind=r.os("kind"),
            call_type=r.os("call_type"),
            synthesized_by=r.os("synthesized_by"),
        )
        for r in _rows(con, "edges")
    )


def _load_classes(con: sqlite3.Connection) -> tuple[CodeClass, ...]:
    return tuple(
        CodeClass(
            id=r.s("id"),
            name=r.s("name"),
            file_path=r.s("file_path"),
            language=r.s("language"),
            parent_classes=_parse_json_list(r.os("parent_classes")),
            interfaces=_parse_json_list(r.os("interfaces")),
            method_ids=_parse_json_list(r.os("method_ids")),
            fan_in=r.i("fan_in"),
            fan_out=r.i("fan_out"),
            is_module=r.b("is_module"),
        )
        for r in _rows(con, "classes")
    )


def _load_cfg(con: sqlite3.Connection) -> tuple[CfgEntry, ...]:
    """Load the optional CFG overlay; absent table → empty (the overlay is optional)."""
    try:
        rows = con.execute("SELECT function_id, file_path, cfg FROM cfg_overlay").fetchall()
    except sqlite3.OperationalError:
        return ()
    return tuple(
        CfgEntry(function_id=r["function_id"], file_path=r["file_path"], cfg=r["cfg"]) for r in rows
    )


def _load_file_hashes(con: sqlite3.Connection) -> dict[str, str]:
    """OpenLore's ``file_hashes`` table verbatim (often empty); absent table → empty dict."""
    try:
        rows = con.execute("SELECT file_path, content_hash FROM file_hashes").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["file_path"]: r["content_hash"] for r in rows}


def _read_fingerprint(repo_path: Path) -> str | None:
    """OpenLore's ``fingerprint.json`` content hash, if present (traceability anchor)."""
    fp = repo_path / FINGERPRINT_RELPATH
    if not fp.exists():
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = data.get("hash")
    return str(value) if value is not None else None


def load_code_graph(
    db_path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> CodeGraph:
    """Load an OpenLore call-graph database into a typed, read-only :class:`CodeGraph`.

    Opens ``db_path`` read-only (the main database file is never written, so its content hash is
    unchanged across the load), validates ``schema_version`` (failing closed on an unsupported
    version), and reads nodes, edges (provenance preserved verbatim), classes, the optional CFG
    overlay, and OpenLore's ``file_hashes``. If ``source_path`` is given (the fixture's ``repo/``
    tree), records the source tree hash and the OpenLore fingerprint so the graph's staleness can
    be checked later.

    Raises :class:`BridgeError` if the database is missing and :class:`SchemaVersionError` if its
    schema is unsupported.
    """
    db = Path(db_path)
    if not db.exists():
        raise BridgeError(f"OpenLore call-graph database not found: {db} (run OpenLore analyze)")

    content_hash_before = _file_sha256(db)
    con = _open_readonly(db)
    try:
        version = _read_schema_version(con)
        nodes = _load_nodes(con)
        edges = _load_edges(con)
        classes = _load_classes(con)
        cfg = _load_cfg(con)
        file_hashes = _load_file_hashes(con)
    finally:
        con.close()

    source_tree_hash: str | None = None
    fingerprint: str | None = None
    if source_path is not None:
        src = Path(source_path)
        source_tree_hash = tree_hash(src, _SOURCE_EXCLUDE)
        fingerprint = _read_fingerprint(src)

    return CodeGraph(
        nodes=nodes,
        edges=edges,
        classes=classes,
        cfg=cfg,
        schema_version=version,
        db_path=str(db),
        db_content_hash=content_hash_before,
        source_path=None if source_path is None else str(source_path),
        source_tree_hash=source_tree_hash,
        fingerprint=fingerprint,
        db_file_hashes=file_hashes,
    )
