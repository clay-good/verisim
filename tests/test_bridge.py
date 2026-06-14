"""Read-only OpenLore call-graph adapter tests (OpenSpec ``add-openlore-graph-adapter``).

Pin the three requirements of the openlore-bridge read-side spec delta. The core
(ReadOnlyCallGraphAccess, SchemaVersionGuard, CacheStalenessAwareness) is exercised against
**synthetic** ``call-graph.db`` files built to OpenLore's exact schema — hermetic, deterministic,
and independent of whether the ``openlore`` CLI is installed, so the suite is macOS-first / Linux-CI
clean. The OpenLore-authored analysis trigger and the supported-surface (MCP) cross-check run
against the real CLI and **skip** when it is absent, so CI never depends on it.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from pathlib import Path

import pytest

from verisim.bridge import (
    MAX_SCHEMA_VERSION,
    BridgeError,
    CodeGraph,
    SchemaVersionError,
    analysis_db_path,
    analyze_fixture,
    call_graph_summary_via_mcp,
    load_code_graph,
    openlore_available,
    subgraph_via_mcp,
)

# --- synthetic-DB helpers (OpenLore schema, no CLI required) ------------------------------------


def _create_schema(con: sqlite3.Connection, *, version: int) -> None:
    con.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, file_path TEXT NOT NULL, class_name TEXT,
            is_async INTEGER NOT NULL DEFAULT 0, language TEXT NOT NULL DEFAULT '',
            start_index INTEGER NOT NULL DEFAULT 0, end_index INTEGER NOT NULL DEFAULT 0,
            fan_in INTEGER NOT NULL DEFAULT 0, fan_out INTEGER NOT NULL DEFAULT 0,
            docstring TEXT, signature TEXT, is_external INTEGER NOT NULL DEFAULT 0,
            external_kind TEXT, is_hub INTEGER NOT NULL DEFAULT 0,
            is_entry_point INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE edges (
            caller_id TEXT NOT NULL, caller_file TEXT NOT NULL, callee_id TEXT NOT NULL,
            callee_file TEXT, callee_name TEXT NOT NULL, line INTEGER, confidence TEXT,
            kind TEXT, call_type TEXT, synthesized_by TEXT
        );
        CREATE TABLE classes (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, file_path TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT '', parent_classes TEXT NOT NULL DEFAULT '[]',
            interfaces TEXT NOT NULL DEFAULT '[]', method_ids TEXT NOT NULL DEFAULT '[]',
            fan_in INTEGER NOT NULL DEFAULT 0, fan_out INTEGER NOT NULL DEFAULT 0,
            is_module INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cfg_overlay (
            function_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, cfg TEXT NOT NULL
        );
        CREATE TABLE file_hashes (
            file_path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, updated_at INTEGER NOT NULL
        );
        """
    )
    con.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def _make_db(
    path: Path,
    *,
    version: int = MAX_SCHEMA_VERSION,
    with_synthesized: bool = True,
    with_file_hashes: bool = True,
    with_cfg: bool = True,
) -> Path:
    """Build a small OpenLore-schema ``call-graph.db`` at ``path`` and return it."""
    con = sqlite3.connect(path)
    try:
        _create_schema(con, version=version)
        con.executemany(
            "INSERT INTO nodes (id, name, file_path, class_name, is_async, language, start_index, "
            "end_index, fan_in, fan_out, docstring, signature, is_external, external_kind, is_hub, "
            "is_entry_point) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    "src/app.py::main",
                    "main",
                    "src/app.py",
                    None,
                    0,
                    "Python",
                    0,
                    5,
                    0,
                    2,
                    "entry",
                    "def main() -> int",
                    0,
                    None,
                    0,
                    1,
                ),
                (
                    "src/app.py::helper",
                    "helper",
                    "src/app.py",
                    None,
                    0,
                    "Python",
                    6,
                    9,
                    1,
                    0,
                    None,
                    None,
                    0,
                    None,
                    0,
                    0,
                ),
                (
                    "src/util.py::Tool.run",
                    "run",
                    "src/util.py",
                    "Tool",
                    1,
                    "Python",
                    0,
                    3,
                    1,
                    0,
                    None,
                    None,
                    0,
                    None,
                    1,
                    0,
                ),
            ],
        )
        edges: list[tuple[str, str, str, str | None, str, int, str, str, str, str | None]] = [
            (
                "src/app.py::main",
                "src/app.py",
                "src/app.py::helper",
                "src/app.py",
                "helper",
                7,
                "same_file",
                "calls",
                "direct",
                None,
            ),
            (
                "src/app.py::main",
                "src/app.py",
                "external::print",
                None,
                "print",
                8,
                "external",
                "calls",
                "direct",
                None,
            ),
        ]
        if with_synthesized:
            edges.append(
                (
                    "src/app.py::main",
                    "src/app.py",
                    "src/util.py::Tool.run",
                    "src/util.py",
                    "run",
                    4,
                    "synthesized",
                    "calls",
                    "direct",
                    "route-handler",
                )
            )
        con.executemany(
            "INSERT INTO edges (caller_id, caller_file, callee_id, callee_file, callee_name, line, "
            "confidence, kind, call_type, synthesized_by) VALUES (?,?,?,?,?,?,?,?,?,?)",
            edges,
        )
        con.execute(
            "INSERT INTO classes (id, name, file_path, language, parent_classes, interfaces, "
            "method_ids, fan_in, fan_out, is_module) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "src/util.py::Tool",
                "Tool",
                "src/util.py",
                "Python",
                '["Base"]',
                "[]",
                '["src/util.py::Tool.run"]',
                1,
                0,
                0,
            ),
        )
        if with_cfg:
            con.execute(
                "INSERT INTO cfg_overlay (function_id, file_path, cfg) VALUES (?,?,?)",
                ("src/app.py::main", "src/app.py", '{"blocks": []}'),
            )
        if with_file_hashes:
            con.execute(
                "INSERT INTO file_hashes (file_path, content_hash, updated_at) VALUES (?,?,?)",
                ("src/app.py", "abc123", 1780456811267),
            )
        con.commit()
    finally:
        con.close()
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_source_tree(root: Path) -> Path:
    """A tiny on-disk source tree (the fixture ``repo/`` stand-in) for staleness checks."""
    repo = root / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def main():\n    return helper()\n", encoding="utf-8")
    (repo / "src" / "util.py").write_text(
        "class Tool:\n    def run(self):\n        ...\n", encoding="utf-8"
    )
    return repo


# --- ReadOnlyCallGraphAccess -------------------------------------------------------------------


def test_loading_does_not_mutate_the_database(tmp_path: Path) -> None:
    """Scenario: loading does not mutate the database (content hash unchanged)."""
    db = _make_db(tmp_path / "call-graph.db")
    before = _sha256(db)
    load_code_graph(db)
    assert _sha256(db) == before


def test_synthesized_edge_provenance_is_preserved(tmp_path: Path) -> None:
    """Scenario: a synthesized edge round-trips with confidence and synthesizedBy intact."""
    db = _make_db(tmp_path / "call-graph.db", with_synthesized=True)
    graph = load_code_graph(db)
    synth = graph.synthesized_edges()
    assert len(synth) == 1
    edge = synth[0]
    assert edge.confidence == "synthesized"
    assert edge.synthesized_by == "route-handler"
    assert edge.is_synthesized


def test_all_edge_provenance_fields_preserved_verbatim(tmp_path: Path) -> None:
    """Provenance is lossless: every confidence/kind/call_type stored is read back exactly."""
    db = _make_db(tmp_path / "call-graph.db")
    graph = load_code_graph(db)
    confidences = {e.confidence for e in graph.edges}
    assert confidences == {"same_file", "external", "synthesized"}
    same_file = next(e for e in graph.edges if e.confidence == "same_file")
    assert (same_file.kind, same_file.call_type, same_file.line) == ("calls", "direct", 7)
    external = next(e for e in graph.edges if e.confidence == "external")
    assert external.callee_file is None  # nullable column preserved as None


def test_nodes_classes_cfg_and_file_hashes_loaded(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "call-graph.db")
    graph = load_code_graph(db)
    assert len(graph.nodes) == 3
    assert graph.node_by_id("src/util.py::Tool.run") is not None
    assert graph.node_by_id("nonexistent") is None
    assert graph.entry_point_ids() == frozenset({"src/app.py::main"})
    tool = next(c for c in graph.classes if c.id == "src/util.py::Tool")
    assert tool.parent_classes == ("Base",)
    assert tool.method_ids == ("src/util.py::Tool.run",)
    assert len(graph.cfg) == 1
    assert graph.db_file_hashes == {"src/app.py": "abc123"}


def test_missing_database_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(BridgeError, match="not found"):
        load_code_graph(tmp_path / "absent.db")


def test_optional_tables_absent_degrade_to_empty(tmp_path: Path) -> None:
    """An OpenLore DB without the CFG overlay / file_hashes still loads (those are optional)."""
    db = _make_db(tmp_path / "call-graph.db", with_cfg=False, with_file_hashes=False)
    graph = load_code_graph(db)
    assert graph.cfg == ()
    assert graph.db_file_hashes == {}


# --- SchemaVersionGuard ------------------------------------------------------------------------


def test_unsupported_schema_above_range_fails_closed(tmp_path: Path) -> None:
    """Scenario: a schema_version above the supported range fails closed with no partial graph."""
    db = _make_db(tmp_path / "call-graph.db", version=MAX_SCHEMA_VERSION + 50)
    with pytest.raises(SchemaVersionError, match="unsupported OpenLore schema_version"):
        load_code_graph(db)


def test_unsupported_schema_below_range_fails_closed(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "call-graph.db", version=1)
    with pytest.raises(SchemaVersionError):
        load_code_graph(db)


def test_non_openlore_db_fails_closed(tmp_path: Path) -> None:
    """A SQLite file with no schema_version table is not an OpenLore graph — fail closed."""
    db = tmp_path / "call-graph.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE unrelated (x INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(SchemaVersionError):
        load_code_graph(db)


# --- CacheStalenessAwareness -------------------------------------------------------------------


def test_fresh_graph_is_not_stale(tmp_path: Path) -> None:
    repo = _make_source_tree(tmp_path)
    db = _make_db(tmp_path / "call-graph.db")
    graph = load_code_graph(db, source_path=repo)
    assert graph.source_tree_hash is not None
    assert graph.is_stale_against(repo) is False


def test_modifying_the_source_invalidates_the_graph(tmp_path: Path) -> None:
    """Scenario: editing the fixture (as a re-analysis would) flips staleness to true."""
    repo = _make_source_tree(tmp_path)
    db = _make_db(tmp_path / "call-graph.db")
    graph = load_code_graph(db, source_path=repo)
    assert graph.is_stale_against(repo) is False
    (repo / "src" / "app.py").write_text("def main():\n    return 99\n", encoding="utf-8")
    assert graph.is_stale_against(repo) is True


def test_staleness_ignores_openlore_cache(tmp_path: Path) -> None:
    """The regenerable ``.openlore/`` cache is excluded from the source fingerprint."""
    repo = _make_source_tree(tmp_path)
    db = _make_db(tmp_path / "call-graph.db")
    graph = load_code_graph(db, source_path=repo)
    (repo / ".openlore" / "analysis").mkdir(parents=True)
    (repo / ".openlore" / "analysis" / "junk.json").write_text("{}", encoding="utf-8")
    assert graph.is_stale_against(repo) is False


def test_staleness_without_source_anchor_raises(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "call-graph.db")
    graph = load_code_graph(db)  # no source_path
    assert graph.source_tree_hash is None
    with pytest.raises(BridgeError, match="without a source_path anchor"):
        graph.is_stale_against(tmp_path)


def test_codegraph_is_frozen(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "call-graph.db")
    graph = load_code_graph(db)
    with pytest.raises((AttributeError, TypeError)):
        graph.schema_version = 99  # type: ignore[misc]


# --- analysis trigger + supported-surface cross-check (real CLI; skipped when absent) ----------

_REQUIRES_OPENLORE = pytest.mark.skipif(not openlore_available(), reason="openlore CLI not on PATH")


def _make_git_repo(root: Path) -> Path:
    repo = root / "subject"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text(
        "def helper():\n    return 1\n\n\ndef main():\n    return helper()\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    return repo


@_REQUIRES_OPENLORE
def test_analyze_fixture_produces_loadable_openlore_db(tmp_path: Path) -> None:
    """Task 1: the analyzer is invoked, the DB is OpenLore-authored, and it loads under schema."""
    repo = _make_git_repo(tmp_path)
    db = analyze_fixture(repo)
    assert db == analysis_db_path(repo)
    assert db.exists()
    graph = load_code_graph(db, source_path=repo)
    names = {n.name for n in graph.nodes}
    assert {"helper", "main"} <= names
    assert isinstance(graph, CodeGraph)


@_REQUIRES_OPENLORE
def test_sql_read_agrees_with_supported_mcp_aggregates(tmp_path: Path) -> None:
    """Task 4 parity (aggregates): the canonical SQLite read matches OpenLore's summary surface.

    ``get_call_graph`` reports a *derived* view, so the clean invariants are the internal-node count
    and the entry-point count (verified empirically); its ``total_edges`` is an expanded count that
    diverges from the raw ``edges`` table and is not asserted.
    """
    repo = _make_git_repo(tmp_path)
    analyze_fixture(repo)
    graph = load_code_graph(analysis_db_path(repo))
    summary = call_graph_summary_via_mcp(repo)
    assert len(graph.internal_nodes()) == summary.total_nodes
    assert len(graph.internal_entry_point_ids()) == summary.entry_point_count


@_REQUIRES_OPENLORE
def test_sql_read_agrees_with_real_mcp_call_graph(tmp_path: Path) -> None:
    """Task 4 parity (real graph): the SQLite read's edges match OpenLore's real ``get_subgraph``.

    The strong parity — not aggregate counts but the *actual* call edges. ``subgraph_via_mcp``
    (re)builds the MCP graph index e2e (a version upgrade resets it) and returns the real edges; the
    SQLite read's internal edges around the same function must equal them exactly.
    """
    repo = _make_git_repo(tmp_path)
    analyze_fixture(repo)
    graph = load_code_graph(analysis_db_path(repo))

    mcp_edges = subgraph_via_mcp(repo, "main", direction="both", max_depth=1)
    mcp_pairs = {e.as_pair() for e in mcp_edges}
    sql_pairs = {p for p in graph.internal_edges_named() if "main" in (p[1], p[3])}

    assert mcp_edges, "get_subgraph returned no real edges (index not built?)"
    assert mcp_pairs == sql_pairs
