"""De-fanged real-codebase fixture materializer (OpenSpec ``add-defanged-codebase-fixture``).

The Verisim ↔ OpenLore prototype (findings doc §9) grounds Verisim's simulated
host/filesystem/network world in a **real** codebase and lets a **human-gated CD loop**
prepare commits against it. That needs a subject under test that is, all three at once:

  - **real** — an actual repository with a non-trivial call graph OpenLore can analyze;
  - **isolated** — a throwaway copy under Verisim-owned scratch, so the loop's writes,
    speculative rollbacks, and prepared commits never touch a repo the user cares about;
  - **inert with respect to delivery** — its ``.git`` is **de-fanged** so that even a bug in
    the CD pipeline (Change 6) cannot commit to or push to the original's history or remotes.

This module provides that subject. :func:`materialize` copies a source repository's working
tree (plus a *neutralized* ``.git``) into ``<scratch>/<name>-<short-sha>/repo`` and de-fangs
it **by construction** — there are no remotes to push to, the identity is a sentinel, and a
``pre-push`` hook hard-fails — mirroring the SPEC-11 hermeticity-by-construction discipline
(``oracle/sandbox.py``): safety is a property of the structure, not of trusting downstream code.
:func:`teardown` removes a fixture in full; the round-trip leaves the source provably untouched
because every write lands under scratch and the source is only ever *read*.

**Selection criteria (the "prototype-suitable" repo).** Steer toward small, real, multi-file
repos with a single dominant language and a genuine call graph — those exercise the static↔dynamic
correction on real structure without paying for scale the prototype does not yet need. On-disk
candidates under the default root, by working-tree file count, smallest-first:

  - ``worldify`` (~36) · ``securifine`` (~42) · ``agent-replay`` (~65) · ``armorly`` (~68) ·
    ``proxilion-mcp`` (~130) — all good defaults.
  - ``invariant`` (~339k) · ``proxilion`` (~321k) — **out of scope** for the prototype; their
    size defeats the "small, real" intent. Selection is explicit (a name/path), never "pick any".

**Honest limits.** A source with submodules or Git-LFS pointers cannot be copied as a single
self-contained tree; the materializer detects both and **fails loudly** rather than producing a
half-fixture (proposal §"Risks"). ``.git`` *internals* (pack layout, timestamps) are not
byte-deterministic; the determinism contract covers the **working tree**, which is.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# The default source-roots allowlist (findings doc §9.3). A source path outside every root is
# rejected; the prototype only ever materializes from here.
DEFAULT_SOURCE_ROOT = Path("/Users/user/Documents/development/public")

# Directories never worth copying into a fixture: language/dependency caches and build output.
# They bloat the copy, are regenerable, and are not part of the call graph OpenLore analyzes.
# ``.git`` is deliberately *not* here — it is copied and then neutralized in place.
DEFAULT_EXCLUDE: frozenset[str] = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".DS_Store",
    }
)

# The sentinel fixture identity stamped into the copy's local git config. A ``.invalid`` TLD
# (RFC 2606) can never resolve, so a stray commit is attributable and a stray push has nowhere
# to go even before the hook fires.
FIXTURE_USER_NAME = "Verisim Fixture"
FIXTURE_USER_EMAIL = "fixture@verisim.invalid"

# A ``pre-push`` hook that unconditionally refuses. Belt to the no-remotes braces: even if a
# remote were somehow added, every push is rejected before a single object leaves the fixture.
_PRE_PUSH_HOOK = (
    "#!/bin/sh\n"
    "# Installed by verisim.fixture: a de-fanged fixture may never push (OpenSpec\n"
    "# add-defanged-codebase-fixture, GitDefanging). Hard-fail unconditionally.\n"
    'echo "verisim-fixture: push is structurally blocked in a de-fanged fixture" >&2\n'
    "exit 1\n"
)

MANIFEST_NAME = "FIXTURE.json"
REPO_SUBDIR = "repo"


class FixtureError(RuntimeError):
    """A fixture could not be materialized, de-fanged, or torn down safely.

    Raised loudly (proposal §"Risks") rather than producing a half-fixture: an out-of-allowlist
    source, a non-git source, a submodule/LFS source we cannot copy whole, a scratch root that
    overlaps the source or ``.openlore``, or an incomplete copy.
    """


class GitUnavailable(FixtureError):
    """No ``git`` executable is available to de-fang or inspect a fixture.

    A first-class, disclosed failure (the SPEC-11 ``SystemOracleUnavailable`` discipline): the
    materializer cannot guarantee de-fanging without git, so it refuses rather than silently
    shipping an un-neutralized copy.
    """


@dataclass(frozen=True)
class FixtureConfig:
    """Where fixtures may come from, where they go, and what is left out of the copy.

    ``source_roots`` is the allowlist: a source path must resolve inside one of them or
    materialization is rejected. ``scratch_root`` is the Verisim-owned destination — it must be
    outside every source root and outside any ``.openlore/`` (validated by construction).
    ``exclude`` names directories dropped from the copy (caches/build output).
    """

    source_roots: tuple[Path, ...] = (DEFAULT_SOURCE_ROOT,)
    scratch_root: Path = field(default_factory=lambda: _default_scratch_root())
    exclude: frozenset[str] = DEFAULT_EXCLUDE


@dataclass(frozen=True)
class FixtureManifest:
    """The traceability record written to ``FIXTURE.json`` (GitDefanging requirement).

    Pins the fixture to an exact source revision (``source_head_sha``) and records every de-fang
    action applied, so every downstream artifact (trace, synthesized edge, prepared commit) is
    attributable to a known source state. ``tree_hash`` is the content hash of the copied working
    tree — the completeness witness and the determinism key.
    """

    source_path: str
    source_head_sha: str
    source_head_short_sha: str
    copy_timestamp: str
    defang_actions: tuple[str, ...]
    tree_hash: str
    file_count: int

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, stable separators) — stable across dict ordering."""
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Fixture:
    """A materialized, de-fanged fixture: its root, the subject ``repo/`` tree, and the manifest."""

    root: Path
    repo_path: Path
    manifest: FixtureManifest


def _default_scratch_root() -> Path:
    """Verisim-owned scratch under the OS temp dir — never inside any repo, so never committed.

    ``VERISIM_FIXTURE_SCRATCH`` overrides it (e.g. CI). Repo-relative defaults are avoided so the
    module is correct regardless of the working directory; the gitignored ``.verisim-fixtures/``
    entry covers callers who *opt into* a repo-relative scratch root.
    """
    override = os.environ.get("VERISIM_FIXTURE_SCRATCH")
    if override:
        return Path(override)
    import tempfile

    return Path(tempfile.gettempdir()) / "verisim-fixtures"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ``git -C repo <args>`` capturing text output. Raises :class:`GitUnavailable` if git is
    absent, :class:`FixtureError` on a non-zero exit when ``check``."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # no git on PATH
        raise GitUnavailable("git executable not found on PATH") from exc
    if check and proc.returncode != 0:
        raise FixtureError(
            f"git {' '.join(args)} failed in {repo} (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc


def _resolve_within(path: Path, roots: tuple[Path, ...]) -> bool:
    """True iff ``path`` (resolved) is equal to or inside one of the resolved ``roots``."""
    rp = path.resolve()
    for root in roots:
        rr = root.resolve()
        if rp == rr or rr in rp.parents:
            return True
    return False


def validate_source(source: Path, config: FixtureConfig) -> Path:
    """Resolve and validate a source path: inside the allowlist, a git repo, copyable whole.

    Returns the resolved source path. Raises :class:`FixtureError` if the path is outside every
    configured root, is not a git working tree, or carries submodules / Git-LFS pointers that a
    plain tree copy would silently truncate (proposal §"Risks": fail loudly, never half-copy).
    """
    resolved = source.resolve()
    if not resolved.exists():
        raise FixtureError(f"source path does not exist: {resolved}")
    if not _resolve_within(resolved, config.source_roots):
        roots = ", ".join(str(r) for r in config.source_roots)
        raise FixtureError(
            f"source {resolved} is outside the roots allowlist ({roots}); refusing to materialize"
        )
    if not (resolved / ".git").exists():
        raise FixtureError(f"source {resolved} is not a git repository (no .git)")
    if (resolved / ".gitmodules").exists():
        raise FixtureError(
            f"source {resolved} has submodules (.gitmodules); a flat copy would be incomplete"
        )
    gitattributes = resolved / ".gitattributes"
    if gitattributes.exists() and "filter=lfs" in gitattributes.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise FixtureError(
            f"source {resolved} uses Git-LFS (filter=lfs); pointer files would copy, not content"
        )
    return resolved


def _validate_scratch(scratch: Path, config: FixtureConfig) -> None:
    """The scratch root must not sit inside any source root or any ``.openlore/`` (spec invariant).

    A fixture written inside the source tree would defeat isolation; inside ``.openlore/`` it would
    collide with OpenLore's regenerable analysis cache.
    """
    sp = scratch.resolve()
    if _resolve_within(sp, config.source_roots):
        raise FixtureError(f"scratch root {sp} is inside a source root; refusing (isolation)")
    if ".openlore" in sp.parts:
        raise FixtureError(f"scratch root {sp} is inside an .openlore/ directory; refusing")


def _tree_entries(root: Path, exclude: frozenset[str]) -> list[tuple[str, str]]:
    """Deterministic ``(relpath, content-hash)`` list of a working tree, excluding ``.git`` and
    every ``exclude`` directory.

    The hash input distinguishes files (SHA-256 of bytes), symlinks (the link target string), and
    empty directories (a sentinel) so the witness is sensitive to structure as well as content.
    Entries are sorted by relpath, so the list — and thus the tree hash — is order-independent.
    """
    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs and .git in place so os.walk does not descend into them.
        dirnames[:] = sorted(d for d in dirnames if d not in exclude and d != ".git")
        rel_dir = Path(dirpath).relative_to(root)
        # Record empty directories explicitly (they are part of the tree shape).
        if not filenames and not dirnames:
            entries.append((str(rel_dir / "") if str(rel_dir) != "." else "./", "<empty-dir>"))
        for name in sorted(filenames):
            if name in exclude:
                continue
            full = Path(dirpath) / name
            rel = str(rel_dir / name) if str(rel_dir) != "." else name
            if full.is_symlink():
                entries.append((rel, "symlink:" + os.readlink(full)))
            elif full.is_file():
                entries.append((rel, "file:" + _hash_file(full)))
    entries.sort()
    return entries


def _hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed so large files do not load wholesale."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(root: Path, exclude: frozenset[str] = DEFAULT_EXCLUDE) -> str:
    """Content hash of a working tree (``.git`` and ``exclude`` dirs omitted).

    The determinism key (two copies of one source hash equal) and the round-trip safety witness
    (the source's hash before materialize equals its hash after teardown — provably untouched).
    """
    entries = _tree_entries(root, exclude)
    blob = "\n".join(f"{rel}\0{digest}" for rel, digest in entries)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _copy_ignore(exclude: frozenset[str]) -> Callable[[str, list[str]], set[str]]:
    """A ``shutil.copytree`` ignore callable dropping ``exclude`` dirs from the copy."""

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in exclude}

    return ignore


def _defang(repo: Path) -> tuple[str, ...]:
    """Neutralize a copied ``.git`` by construction; return the ordered list of actions applied.

    By construction means: nothing downstream is trusted to *avoid* a push — the structure makes
    a push impossible. Remotes are removed (nowhere to push), the reflog is dropped (no path back
    to the original's history references), every hook is replaced by a single failing ``pre-push``,
    and the identity is a sentinel. Finally we *assert* no remote URL resolves — de-fanging that
    cannot be confirmed is a :class:`FixtureError`, not a warning.
    """
    actions: list[str] = []

    # 1. Remove every remote.
    remotes = [r for r in _git(repo, "remote").stdout.splitlines() if r.strip()]
    for remote in remotes:
        _git(repo, "remote", "remove", remote)
    actions.append(f"removed_remotes:{','.join(remotes) if remotes else 'none'}")

    # 2. Drop the reflog so the copy carries no reference path to the source's history.
    logs = repo / ".git" / "logs"
    if logs.exists():
        shutil.rmtree(logs)
    actions.append("removed_reflog")

    # 3. Replace all hooks with a single unconditional pre-push refusal.
    hooks = repo / ".git" / "hooks"
    if hooks.exists():
        shutil.rmtree(hooks)
    hooks.mkdir(parents=True)
    pre_push = hooks / "pre-push"
    pre_push.write_text(_PRE_PUSH_HOOK, encoding="utf-8")
    pre_push.chmod(pre_push.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    actions.append("installed_failing_pre_push_hook")

    # 4. Sentinel identity + an explicit fixture marker.
    _git(repo, "config", "user.name", FIXTURE_USER_NAME)
    _git(repo, "config", "user.email", FIXTURE_USER_EMAIL)
    _git(repo, "config", "verisim.fixture", "true")
    actions.append(f"set_identity:{FIXTURE_USER_NAME} <{FIXTURE_USER_EMAIL}>")

    # 5. Assert the de-fang held: zero remotes resolve.
    remaining = [r for r in _git(repo, "remote").stdout.splitlines() if r.strip()]
    if remaining:
        raise FixtureError(f"de-fang failed: remotes still present after removal: {remaining}")
    actions.append("asserted_no_remote")

    return tuple(actions)


def materialize(source: str | Path, config: FixtureConfig | None = None) -> Fixture:
    """Materialize ``source`` into an isolated, de-fanged fixture and return its handle.

    Steps: validate the source (allowlist + git + copyable-whole) and the scratch root; copy the
    working tree and ``.git`` (minus ``exclude`` dirs) into ``<scratch>/<name>-<short-sha>/repo``;
    de-fang the copy's ``.git`` by construction; verify completeness against a re-hash of the copy;
    and write the ``FIXTURE.json`` manifest. Raises :class:`FixtureError` (loudly) on any failure,
    leaving no half-fixture behind.
    """
    config = config or FixtureConfig()
    src = validate_source(Path(source), config)
    _validate_scratch(config.scratch_root, config)

    head = _git(src, "rev-parse", "HEAD").stdout.strip()
    short = _git(src, "rev-parse", "--short", "HEAD").stdout.strip()
    fixture_root = config.scratch_root / f"{src.name}-{short}"
    repo_path = fixture_root / REPO_SUBDIR

    if fixture_root.exists():
        raise FixtureError(
            f"fixture destination already exists: {fixture_root} "
            "(tear it down before re-materializing)"
        )
    if _resolve_within(repo_path, (src,)):
        raise FixtureError("refusing to copy a source into itself")

    fixture_root.mkdir(parents=True)
    try:
        shutil.copytree(
            src, repo_path, symlinks=True, ignore=_copy_ignore(config.exclude)
        )
        source_hash = tree_hash(src, config.exclude)
        defang_actions = _defang(repo_path)

        # Completeness: the de-fang only touches .git, so the working tree must still hash equal
        # to the source's. A mismatch means the copy dropped or altered tracked content.
        copy_hash = tree_hash(repo_path, config.exclude)
        if copy_hash != source_hash:
            raise FixtureError(
                "fixture working tree does not match source after copy "
                f"(source {source_hash[:12]} != copy {copy_hash[:12]}); incomplete materialization"
            )

        file_count = sum(
            1 for _, kind in _tree_entries(repo_path, config.exclude) if kind != "<empty-dir>"
        )
        manifest = FixtureManifest(
            source_path=str(src),
            source_head_sha=head,
            source_head_short_sha=short,
            copy_timestamp=datetime.now(UTC).isoformat(),
            defang_actions=defang_actions,
            tree_hash=copy_hash,
            file_count=file_count,
        )
        (fixture_root / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    except Exception:
        # Never leave a half-fixture; clean up the partial scratch dir before re-raising.
        shutil.rmtree(fixture_root, ignore_errors=True)
        raise

    return Fixture(root=fixture_root, repo_path=repo_path, manifest=manifest)


def remotes(repo: Path) -> list[str]:
    """List a fixture repo's configured remotes (empty for a de-fanged fixture)."""
    return [r for r in _git(repo, "remote").stdout.splitlines() if r.strip()]


def load_manifest(fixture_root: Path) -> FixtureManifest:
    """Read and parse a fixture's ``FIXTURE.json`` manifest."""
    data = json.loads((fixture_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    # JSON has no tuples; restore the declared shape so a loaded manifest equals the produced one.
    data["defang_actions"] = tuple(data["defang_actions"])
    return FixtureManifest(**data)


def teardown(fixture: Fixture | Path) -> None:
    """Remove a fixture in full (the ``FixtureTeardown`` requirement).

    Accepts a :class:`Fixture` or its root path. Idempotent: a non-existent fixture is a no-op.
    Only ever removes Verisim-owned scratch — never the source, which it does not even reference.
    """
    root = fixture.root if isinstance(fixture, Fixture) else fixture
    shutil.rmtree(root, ignore_errors=True)
    if root.exists():
        raise FixtureError(f"teardown failed: {root} still exists")
