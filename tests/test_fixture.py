"""De-fanged real-codebase fixture tests (OpenSpec ``add-defanged-codebase-fixture``).

Pin the three requirements of the fixture spec delta on a real, locally-built git repo (no
network, no external repo): IsolatedFixtureMaterialization (allowlist enforcement, source left
untouched, deterministic copy), GitDefanging (no remotes, push structurally blocked, manifest
traceable to the source HEAD), and FixtureTeardown (full removal). Every test builds its own
source repo inside the test's allowlist, so the suite is hermetic and cross-POSIX (macOS-first,
Linux CI for free) — it never touches a repo the user cares about.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verisim.fixture import (
    Fixture,
    FixtureConfig,
    FixtureError,
    load_manifest,
    materialize,
    remotes,
    teardown,
    tree_hash,
    validate_source,
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout


def _make_source_repo(root: Path, *, with_remote: bool = True) -> Path:
    """Build a small, real git repo with a couple of files and one commit."""
    repo = root / "sample-proj"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (repo / "README.md").write_text("# sample\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.name", "Source Author")
    _git(repo, "config", "user.email", "author@example.com")
    if with_remote:
        _git(repo, "remote", "add", "origin", "https://example.com/sample-proj.git")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _config_for(allowlist_root: Path, scratch: Path) -> FixtureConfig:
    return FixtureConfig(source_roots=(allowlist_root,), scratch_root=scratch)


# --- IsolatedFixtureMaterialization ------------------------------------------------------------


def test_source_outside_allowlist_rejected(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    other = tmp_path / "elsewhere"
    src_root.mkdir()
    other.mkdir()
    repo = _make_source_repo(other)
    config = _config_for(src_root, tmp_path / "scratch")

    with pytest.raises(FixtureError, match="outside the roots allowlist"):
        materialize(repo, config)
    # No scratch state created for a rejected request.
    assert not (tmp_path / "scratch").exists()


def test_non_git_source_rejected(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    plain = src_root / "not-a-repo"
    plain.mkdir(parents=True)
    (plain / "f.txt").write_text("hi\n", encoding="utf-8")
    config = _config_for(src_root, tmp_path / "scratch")

    with pytest.raises(FixtureError, match="not a git repository"):
        materialize(plain, config)


def test_submodule_source_fails_loudly(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    (repo / ".gitmodules").write_text("[submodule \"x\"]\n", encoding="utf-8")
    config = _config_for(src_root, tmp_path / "scratch")

    with pytest.raises(FixtureError, match="submodules"):
        materialize(repo, config)


def test_lfs_source_fails_loudly(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    (repo / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs\n", encoding="utf-8")
    config = _config_for(src_root, tmp_path / "scratch")

    with pytest.raises(FixtureError, match="Git-LFS"):
        materialize(repo, config)


def test_scratch_inside_source_root_rejected(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    # Scratch nested inside the allowlisted source root → isolation violation.
    config = _config_for(src_root, src_root / "scratch")

    with pytest.raises(FixtureError, match="inside a source root"):
        materialize(repo, config)


def test_materialization_leaves_source_untouched(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root)
    before = tree_hash(repo)

    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))
    # Use the fixture (write into it) — the source must still be untouched afterward.
    (fx.repo_path / "scratch_file.txt").write_text("written into the fixture\n", encoding="utf-8")
    teardown(fx)

    after = tree_hash(repo)
    assert before == after, "source working tree changed across materialize→use→teardown"


def test_deterministic_copy(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)

    fx1 = materialize(repo, _config_for(src_root, tmp_path / "scratch-a"))
    fx2 = materialize(repo, _config_for(src_root, tmp_path / "scratch-b"))

    # Identical working-tree file sets and per-file content hashes (excluding volatile .git).
    assert tree_hash(fx1.repo_path) == tree_hash(fx2.repo_path)
    assert fx1.manifest.tree_hash == fx2.manifest.tree_hash


def test_excluded_dirs_are_dropped(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    # A cache dir that is untracked-but-present should not be copied.
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "junk.js").write_text("// junk\n", encoding="utf-8")

    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))
    assert not (fx.repo_path / "node_modules").exists()
    assert (fx.repo_path / "src" / "main.py").exists()


def test_redundant_materialize_into_existing_dest_rejected(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    scratch = tmp_path / "scratch"

    materialize(repo, _config_for(src_root, scratch))
    with pytest.raises(FixtureError, match="already exists"):
        materialize(repo, _config_for(src_root, scratch))


# --- GitDefanging -------------------------------------------------------------------------------


def test_no_remote_remains(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=True)
    assert remotes(repo) == ["origin"], "precondition: source has a remote"

    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))
    assert remotes(fx.repo_path) == []
    assert "removed_remotes:origin" in fx.manifest.defang_actions


def test_push_is_structurally_blocked(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=True)
    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))

    # No remote → push fails on resolution.
    no_remote = subprocess.run(
        ["git", "-C", str(fx.repo_path), "push"], capture_output=True, text=True
    )
    assert no_remote.returncode != 0

    # Even if a remote is forced back in, the pre-push hook hard-fails.
    _git(fx.repo_path, "remote", "add", "origin", "https://example.com/x.git")
    with_remote = subprocess.run(
        ["git", "-C", str(fx.repo_path), "push", "origin", "HEAD"],
        capture_output=True,
        text=True,
    )
    assert with_remote.returncode != 0


def test_pre_push_hook_installed_and_executable(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))

    hook = fx.repo_path / ".git" / "hooks" / "pre-push"
    assert hook.exists()
    import os

    assert os.access(hook, os.X_OK), "pre-push hook must be executable"
    assert "installed_failing_pre_push_hook" in fx.manifest.defang_actions


def test_sentinel_identity_set(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))

    assert _git(fx.repo_path, "config", "user.email").strip() == "fixture@verisim.invalid"
    assert _git(fx.repo_path, "config", "verisim.fixture").strip() == "true"


def test_manifest_traceable_to_source_revision(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    head = _git(repo, "rev-parse", "HEAD").strip()

    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))
    assert fx.manifest.source_head_sha == head
    assert head.startswith(fx.manifest.source_head_short_sha)
    assert fx.manifest.source_path == str(repo.resolve())

    # The on-disk manifest round-trips and matches.
    loaded = load_manifest(fx.root)
    assert loaded == fx.manifest


# --- FixtureTeardown ----------------------------------------------------------------------------


def test_teardown_removes_fixture(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))
    assert fx.root.exists()

    teardown(fx)
    assert not fx.root.exists()
    # Idempotent: tearing down again is a no-op, not an error.
    teardown(fx)
    assert not fx.root.exists()


def test_teardown_accepts_path(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))

    teardown(fx.root)
    assert not fx.root.exists()


# --- validate_source surface --------------------------------------------------------------------


def test_validate_source_returns_resolved_path(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    config = _config_for(src_root, tmp_path / "scratch")

    resolved = validate_source(repo, config)
    assert resolved == repo.resolve()


def test_returned_fixture_shape(tmp_path: Path) -> None:
    src_root = tmp_path / "allowed"
    src_root.mkdir()
    repo = _make_source_repo(src_root, with_remote=False)
    fx = materialize(repo, _config_for(src_root, tmp_path / "scratch"))

    assert isinstance(fx, Fixture)
    assert fx.repo_path.parent == fx.root
    assert fx.repo_path.name == "repo"
    assert (fx.root / "FIXTURE.json").exists()
