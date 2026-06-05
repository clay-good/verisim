"""``ReferenceOracle``: the deterministic reference interpreter (SPEC-2 §3.1).

This is the symbolic half of the neuro-symbolic system (SPEC.md §6) and the
executable definition of the v0 semantics -- ``docs/semantics.md`` is the
normative English; this code is the truth, and any disagreement is a bug
resolved by the golden tests (SPEC-2 §16).

Each command handler returns the structural/cwd/env edits plus an ``(exit_code,
stdout)`` observation. ``step`` then appends a ``SetResult`` edit (every command
updates ``last``) and produces the next state via the shared :func:`apply`, so
``apply(s, step(s,a).delta) == step(s,a).state`` holds by construction (the M1
invariant). Handlers compute edits against ``state`` and never mutate it.
"""

from __future__ import annotations

from verisim.delta.apply import apply
from verisim.delta.edits import (
    Chmod,
    Create,
    Delete,
    Delta,
    Edit,
    Modify,
    Move,
    SetCwd,
    SetEnv,
    SetResult,
)
from verisim.env.action import Action
from verisim.env.state import (
    Dir,
    File,
    Node,
    State,
    basename,
    content_hash,
    parent,
    resolve,
)

from .base import EXIT_ERR, EXIT_OK, DeterminismReport, Oracle, StepResult


def _is_dir(fs: dict[str, Node], path: str) -> bool:
    return isinstance(fs.get(path), Dir)


def _is_file(fs: dict[str, Node], path: str) -> bool:
    return isinstance(fs.get(path), File)


def _has_children(fs: dict[str, Node], path: str) -> bool:
    prefix = path + "/"
    return any(p.startswith(prefix) for p in fs)


def _subtree(fs: dict[str, Node], root: str) -> list[str]:
    prefix = root + "/"
    return [p for p in fs if p == root or p.startswith(prefix)]


def _dir_ok_parent(fs: dict[str, Node], path: str) -> bool:
    """True if ``path``'s parent exists and is a directory (precondition for
    creating ``path``)."""
    return _is_dir(fs, parent(path))


# --- handlers: each returns (edits, exit_code, stdout) ----------------------

_Outcome = tuple[list[Edit], int, str]


def _fail() -> _Outcome:
    return ([], EXIT_ERR, "")


class ReferenceOracle(Oracle):
    """A pure, total, deterministic interpreter of the v0 grammar."""

    version = "ref-1"

    def step(self, state: State, action: Action) -> StepResult:
        edits, exit_code, stdout = self._dispatch(state, action)
        full: Delta = [*edits, SetResult(exit_code, content_hash(stdout))]
        return StepResult(
            state=apply(state, full), delta=full, exit_code=exit_code, stdout=stdout
        )

    def reset(self, state: State) -> State:
        return state.copy()

    def determinism_report(self) -> DeterminismReport:
        return DeterminismReport(
            clock_sealed=True,
            rng_sealed=True,
            concurrency_sealed=True,
            env_leakage_sealed=True,
            notes="Reference interpreter: O(s,a) is a pure function of (s,a) only.",
        )

    # -- dispatch ------------------------------------------------------------

    def _dispatch(self, state: State, action: Action) -> _Outcome:
        fs = state.fs
        name = action.name
        if name == "mkdir":
            return self._mkdir(fs, resolve(state.cwd, action.args[0]))
        if name == "rmdir":
            return self._rmdir(fs, resolve(state.cwd, action.args[0]))
        if name == "touch":
            return self._touch(fs, resolve(state.cwd, action.args[0]))
        if name == "rm":
            return self._rm(fs, resolve(state.cwd, action.args[0]), action.recursive)
        if name == "mv":
            return self._copy_or_move(state, action, move=True, recursive=True)
        if name == "cp":
            return self._copy_or_move(state, action, move=False, recursive=action.recursive)
        if name == "write":
            return self._write(fs, resolve(state.cwd, action.args[0]), action.args[1])
        if name == "append":
            return self._append(fs, resolve(state.cwd, action.args[0]), action.args[1])
        if name == "chmod":
            return self._chmod(fs, action.args[0], resolve(state.cwd, action.args[1]))
        if name == "cd":
            return self._cd(fs, resolve(state.cwd, action.args[0]))
        if name == "cat":
            return self._cat(fs, resolve(state.cwd, action.args[0]))
        if name == "ls":
            return self._ls(fs, resolve(state.cwd, action.args[0]))
        if name == "export":
            return ([SetEnv(action.args[0], action.args[1])], EXIT_OK, "")
        raise ValueError(f"oracle has no handler for {name!r}")  # pragma: no cover

    # -- structure-building --------------------------------------------------

    def _mkdir(self, fs: dict[str, Node], path: str) -> _Outcome:
        if path in fs or not _dir_ok_parent(fs, path):
            return _fail()
        return ([Create(path, Dir())], EXIT_OK, "")

    def _touch(self, fs: dict[str, Node], path: str) -> _Outcome:
        if path in fs:
            return ([], EXIT_OK, "")  # exists (file or dir): no-op success
        if not _dir_ok_parent(fs, path):
            return _fail()
        return ([Create(path, File())], EXIT_OK, "")

    def _write(self, fs: dict[str, Node], path: str, token: str) -> _Outcome:
        if _is_dir(fs, path):
            return _fail()
        if _is_file(fs, path):
            return ([Modify(path, token)], EXIT_OK, "")
        if not _dir_ok_parent(fs, path):
            return _fail()
        return ([Create(path, File(content=token))], EXIT_OK, "")

    def _append(self, fs: dict[str, Node], path: str, token: str) -> _Outcome:
        if _is_dir(fs, path):
            return _fail()
        if _is_file(fs, path):
            node = fs[path]
            assert isinstance(node, File)
            return ([Modify(path, node.content + token)], EXIT_OK, "")
        if not _dir_ok_parent(fs, path):
            return _fail()
        return ([Create(path, File(content=token))], EXIT_OK, "")

    # -- structure-mutating --------------------------------------------------

    def _rm(self, fs: dict[str, Node], path: str, recursive: bool) -> _Outcome:
        if path == "/" or path not in fs:
            return _fail()  # root is undeletable (cf. real rm --preserve-root)
        if not recursive and _is_dir(fs, path):
            return _fail()  # rm without -r cannot remove a directory
        if recursive:
            return ([Delete(p) for p in sorted(_subtree(fs, path))], EXIT_OK, "")
        return ([Delete(path)], EXIT_OK, "")

    def _rmdir(self, fs: dict[str, Node], path: str) -> _Outcome:
        if path == "/" or not _is_dir(fs, path) or _has_children(fs, path):
            return _fail()  # root is undeletable
        return ([Delete(path)], EXIT_OK, "")

    def _chmod(self, fs: dict[str, Node], mode_str: str, path: str) -> _Outcome:
        if path not in fs:
            return _fail()
        return ([Chmod(path, int(mode_str, 8))], EXIT_OK, "")

    def _copy_or_move(
        self, state: State, action: Action, *, move: bool, recursive: bool
    ) -> _Outcome:
        fs = state.fs
        src = resolve(state.cwd, action.args[0])
        dst = resolve(state.cwd, action.args[1])
        if src == "/" or src not in fs:
            return _fail()  # root cannot be moved or copied as a whole in v0
        if not recursive and _is_dir(fs, src):
            return _fail()  # cp without -r cannot copy a directory

        # Resolve the final target: copy/move *into* dst when it is a directory.
        if _is_dir(fs, dst):
            final = resolve(dst, basename(src))
        else:
            final = dst
            if not _dir_ok_parent(fs, final):
                return _fail()

        if final == src:
            return ([], EXIT_OK, "")  # no-op (e.g. mv a a)

        # A directory cannot be moved or copied into its own subtree. Real mv/cp reject
        # this ("cannot move '/e' to a subdirectory of itself, '/e/e'"); without the guard
        # ``Move``/recursive-``Create`` orphans the moved subtree -- an *invalid* state whose
        # parent path no longer exists. Surfaced by the SPEC-11 system-oracle differential
        # harness (H28: the real kernel is a debugger for the reference model).
        if _is_dir(fs, src) and final.startswith(src + "/"):
            return _fail()

        if move:
            # Overwriting an existing target is rejected in v0 (keeps mv total
            # and unambiguous; documented in docs/semantics.md).
            if final in fs:
                return _fail()
            return ([Move(src, final)], EXIT_OK, "")

        # copy
        if _is_file(fs, src):
            if _is_file(fs, final):
                node = fs[src]
                assert isinstance(node, File)
                return ([Modify(final, node.content)], EXIT_OK, "")  # overwrite file
            return ([Create(final, fs[src])], EXIT_OK, "")
        # recursive directory copy
        if final in fs:
            return _fail()
        edits: list[Edit] = []
        for p in sorted(_subtree(fs, src)):
            new_path = final + p[len(src) :]
            edits.append(Create(new_path, fs[p]))
        return (edits, EXIT_OK, "")

    # -- navigation / read-only ----------------------------------------------

    def _cd(self, fs: dict[str, Node], path: str) -> _Outcome:
        if not _is_dir(fs, path):
            return _fail()
        return ([SetCwd(path)], EXIT_OK, "")

    def _cat(self, fs: dict[str, Node], path: str) -> _Outcome:
        if not _is_file(fs, path):
            return _fail()
        node = fs[path]
        assert isinstance(node, File)
        return ([], EXIT_OK, node.content)

    def _ls(self, fs: dict[str, Node], path: str) -> _Outcome:
        if path not in fs:
            return _fail()
        if _is_file(fs, path):
            return ([], EXIT_OK, basename(path))
        prefix = path + "/" if path != "/" else "/"
        names = sorted(
            p[len(prefix) :]
            for p in fs
            if p != path and p.startswith(prefix) and "/" not in p[len(prefix) :]
        )
        return ([], EXIT_OK, "\n".join(names))
