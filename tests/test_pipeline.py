"""Headless CD pipeline tests (OpenSpec ``add-headless-cd-pipeline``).

Pin the three requirements of the cd-pipeline spec delta on a real, locally-built git fixture (no
network, no external repo): HeadlessOrderedPipeline (a full run produces a prepared, undelivered
change + a content-sealed report; the per-stage artifacts are deterministic), FailSafeGating (a
``critical`` breaker trip halts before delivery and surfaces a rollback recommendation; delivery
needs explicit confirmation; a failed stage blocks later ones), and LocalNetworkIsolatedExecution
(no outbound network, source untouched, a confirmed commit lands only on the fixture and cannot
push). The pipeline is composed over the always-available :class:`ReferenceOracle` so the suite is
hermetic and cross-POSIX (macOS-first, Linux CI for free); the trace machinery is oracle-agnostic.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from verisim.bridge.graph import CodeGraph, load_code_graph
from verisim.fixture import Fixture, FixtureConfig, materialize, remotes, teardown, tree_hash
from verisim.oracle.reference import ReferenceOracle
from verisim.pipeline import (
    REPORT_SCHEMA_VERSION,
    STAGE_FAILED,
    STAGE_OK,
    STAGE_SKIPPED,
    ArchInvariant,
    Intent,
    run_pipeline,
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _make_source_repo(root: Path) -> Path:
    repo = root / "sample-proj"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (repo / "src" / "util.py").write_text("class Tool:\n    def run(self):\n        ...\n", "utf-8")
    (repo / "README.md").write_text("# sample\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.name", "Source Author")
    _git(repo, "config", "user.email", "author@example.com")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _make_graph_db(path: Path, *, with_layering_edge: bool) -> Path:
    """A small OpenLore-schema ``call-graph.db`` with two files and (optionally) a cross-layer edge.

    The ``main -> Tool.run`` edge crosses ``src/app.py -> src/util.py``, the layering an
    :class:`ArchInvariant` can forbid.
    """
    con = sqlite3.connect(path)
    try:
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
        con.execute("INSERT INTO schema_version (version) VALUES (7)")
        con.executemany(
            "INSERT INTO nodes (id, name, file_path, class_name, is_entry_point) "
            "VALUES (?,?,?,?,?)",
            [
                ("src/app.py::main", "main", "src/app.py", None, 1),
                ("src/util.py::Tool.run", "run", "src/util.py", "Tool", 0),
            ],
        )
        if with_layering_edge:
            con.execute(
                "INSERT INTO edges (caller_id, caller_file, callee_id, callee_file, callee_name, "
                "confidence, kind, call_type) VALUES (?,?,?,?,?,?,?,?)",
                (
                    "src/app.py::main",
                    "src/app.py",
                    "src/util.py::Tool.run",
                    "src/util.py",
                    "run",
                    "same_file",
                    "calls",
                    "direct",
                ),
            )
        con.commit()
    finally:
        con.close()
    return path


@pytest.fixture
def fixture(tmp_path: Path) -> Iterator[Fixture]:
    source = _make_source_repo(tmp_path / "src-root")
    cfg = FixtureConfig(source_roots=(tmp_path / "src-root",), scratch_root=tmp_path / "scratch")
    fx = materialize(source, cfg)
    yield fx
    teardown(fx)


@pytest.fixture
def graph(tmp_path: Path) -> CodeGraph:
    db = _make_graph_db(tmp_path / "call-graph.db", with_layering_edge=False)
    return load_code_graph(db)


def _benign_intent() -> Intent:
    return Intent(
        goal="add a note",
        actions=("write /a.txt hello", "mkdir /d", "touch /d/b.txt"),
        change=(("NOTE.md", "prepared by verisim-cd\n"),),
        seed=7,
    )


# --- HeadlessOrderedPipeline -------------------------------------------------------------------


def test_full_run_prepares_an_undelivered_change(fixture: Fixture, graph: CodeGraph) -> None:
    """Scenario: a full run produces a content-sealed report + a prepared, unpushed commit/patch,
    with no commit/push/deploy performed."""
    head_before = _git(fixture.repo_path, "rev-parse", "HEAD").stdout.strip()
    report = run_pipeline(fixture, _benign_intent(), graph=graph, oracle=ReferenceOracle())

    assert report.version == REPORT_SCHEMA_VERSION
    assert not report.halted
    assert [s.name for s in report.stages] == [
        "intent_graph",
        "speculative",
        "trace",
        "evaluate",
        "feedback",
        "breaker",
        "prepare_delivery",
    ]
    assert all(s.status == STAGE_OK for s in report.stages)

    # A prepared, unpushed change: staged + a patch, but not committed.
    assert report.prepared_delivery is not None
    assert report.prepared_delivery.prepared and not report.prepared_delivery.committed
    assert report.prepared_delivery.commit_sha is None
    assert "NOTE.md" in report.prepared_delivery.files_changed
    # No commit happened: HEAD is unchanged.
    assert _git(fixture.repo_path, "rev-parse", "HEAD").stdout.strip() == head_before
    # The report and a patch artifact were written under Verisim-owned scratch (not the repo tree).
    run_dir = fixture.root / ".verisim-run"
    assert (run_dir / "run-report.json").exists()
    assert (run_dir / "07-delivery.patch").read_text().find("NOTE.md") != -1


def test_report_is_content_sealed(fixture: Fixture, graph: CodeGraph) -> None:
    """The report carries a signature equal to the hash of its own content (tamper-evident)."""
    report = run_pipeline(fixture, _benign_intent(), graph=graph, oracle=ReferenceOracle())
    assert report.signature
    assert report.signature == report.compute_signature()
    # Round-trips through JSON unchanged.
    assert json.loads(report.to_json())["signature"] == report.signature


def test_deterministic_stages(tmp_path: Path) -> None:
    """Scenario: a fixed (fixture revision, intent, seed) yields identical per-stage artifacts.

    The *same* source repo (one commit sha) is materialized into two scratch roots — the fixed
    fixture revision — and run with the same intent; the signatures and artifacts must match.
    """
    source = _make_source_repo(tmp_path / "src-root")
    db = _make_graph_db(tmp_path / "g.db", with_layering_edge=False)
    intent = _benign_intent()
    sigs = []
    artifacts = []
    for i in range(2):
        cfg = FixtureConfig(
            source_roots=(tmp_path / "src-root",), scratch_root=tmp_path / f"scratch-{i}"
        )
        fx = materialize(source, cfg)
        g = load_code_graph(db, source_path=fx.repo_path)
        report = run_pipeline(fx, intent, graph=g, oracle=ReferenceOracle())
        sigs.append(report.signature)
        run_dir = fx.root / ".verisim-run"
        artifacts.append({p.name: p.read_text() for p in sorted(run_dir.glob("0*.json"))})
        teardown(fx)

    assert sigs[0] == sigs[1]
    assert artifacts[0] == artifacts[1]


def test_invariant_violation_is_reported(fixture: Fixture, tmp_path: Path) -> None:
    """A declared layering invariant is checked against the static graph; a violation surfaces."""
    db = _make_graph_db(tmp_path / "layered.db", with_layering_edge=True)
    g = load_code_graph(db)
    intent = Intent(
        goal="check layering",
        actions=("mkdir /x",),
        invariants=(ArchInvariant("no_app_to_util", "src/app", "src/util"),),
    )
    report = run_pipeline(fixture, intent, graph=g, oracle=ReferenceOracle())
    assert len(report.findings) == 1
    assert report.findings[0].invariant == "no_app_to_util"
    assert report.findings[0].source == "static"


# --- FailSafeGating ----------------------------------------------------------------------------


def test_breaker_trip_halts_before_delivery(fixture: Fixture, graph: CodeGraph) -> None:
    """Scenario: a run that reaches the ``critical`` tier halts before delivery and surfaces a
    human-gated rollback recommendation, with no commit/push/deploy."""
    head_before = _git(fixture.repo_path, "rev-parse", "HEAD").stdout.strip()
    # A flip-flop trajectory: create then remove the same file repeatedly → oscillation critical.
    flip = ("write /a x", "rm /a") * 4
    intent = Intent(goal="oscillate", actions=flip, change=(("NOTE.md", "x\n"),))
    report = run_pipeline(fixture, intent, graph=graph, oracle=ReferenceOracle())

    assert report.halted
    assert report.breaker_status == "critical"
    assert report.rollback_recommendation is not None
    # The breaker stage halted and delivery was skipped — never reached.
    by_name = {s.name: s for s in report.stages}
    assert by_name["breaker"].status == "halted"
    assert by_name["prepare_delivery"].status == STAGE_SKIPPED
    assert report.prepared_delivery is None
    # No commit, and the change was never applied to the repo.
    assert _git(fixture.repo_path, "rev-parse", "HEAD").stdout.strip() == head_before
    assert not (fixture.repo_path / "NOTE.md").exists()


def test_a_failed_stage_blocks_later_stages(fixture: Fixture) -> None:
    """Fail-safe: with no graph available, the intent_graph stage fails and every later stage is
    skipped — delivery is never reached."""
    report = run_pipeline(fixture, _benign_intent(), graph=None, oracle=ReferenceOracle())
    assert report.halted
    by_name = {s.name: s for s in report.stages}
    assert by_name["intent_graph"].status == STAGE_FAILED
    assert by_name["prepare_delivery"].status == STAGE_SKIPPED
    assert report.prepared_delivery is None


def test_delivery_requires_explicit_confirmation(fixture: Fixture, graph: CodeGraph) -> None:
    """Scenario: with no confirmation, no commit/push/deploy occurs (the default)."""
    report = run_pipeline(fixture, _benign_intent(), graph=graph, oracle=ReferenceOracle())
    assert report.prepared_delivery is not None and not report.prepared_delivery.committed
    # Exactly one commit exists (the fixture's initial commit).
    log = _git(fixture.repo_path, "log", "--oneline").stdout.splitlines()
    assert len(log) == 1


# --- LocalNetworkIsolatedExecution -------------------------------------------------------------


def test_source_untouched_and_network_isolated(tmp_path: Path) -> None:
    """Scenario: the run requires no outbound network and the original source is byte-identical."""
    source = _make_source_repo(tmp_path / "src-root")
    source_hash_before = tree_hash(source)
    cfg = FixtureConfig(source_roots=(tmp_path / "src-root",), scratch_root=tmp_path / "scratch")
    fx = materialize(source, cfg)
    db = _make_graph_db(tmp_path / "g.db", with_layering_edge=False)
    g = load_code_graph(db)
    try:
        report = run_pipeline(fx, _benign_intent(), graph=g, confirm_delivery=True)
        assert report.network_isolated
    finally:
        teardown(fx)
    # The original source repo is unchanged by the whole run (even the confirmed commit).
    assert tree_hash(source) == source_hash_before


def test_confirmed_commit_lands_only_on_fixture_and_cannot_push(
    fixture: Fixture, graph: CodeGraph
) -> None:
    """Scenario: a confirmed commit lands only on the de-fanged fixture; any push fails because the
    fixture has no remote (and a blocking pre-push hook)."""
    report = run_pipeline(
        fixture, _benign_intent(), graph=graph, oracle=ReferenceOracle(), confirm_delivery=True
    )
    assert report.prepared_delivery is not None and report.prepared_delivery.committed
    # The commit exists on the fixture.
    assert report.prepared_delivery.commit_sha == _git(
        fixture.repo_path, "rev-parse", "HEAD"
    ).stdout.strip()
    assert (fixture.repo_path / "NOTE.md").exists()
    # The fixture cannot reach the original: no remote, and a push fails.
    assert remotes(fixture.repo_path) == []
    push = _git(fixture.repo_path, "push", check=False)
    assert push.returncode != 0


def test_commit_honors_the_gate_no_no_verify(fixture: Fixture, graph: CodeGraph) -> None:
    """The pipeline commits plainly (never ``--no-verify``): a fixture pre-commit hook would run.

    We install a refusing pre-commit hook on the fixture; a confirmed commit must fail because the
    gate is honored, not bypassed."""
    hook = fixture.repo_path / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    report = run_pipeline(
        fixture, _benign_intent(), graph=graph, oracle=ReferenceOracle(), confirm_delivery=True
    )
    by_name = {s.name: s for s in report.stages}
    assert by_name["prepare_delivery"].status == STAGE_FAILED  # the gate refused the commit
    assert report.prepared_delivery is None
