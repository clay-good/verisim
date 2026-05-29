"""``apply(state, delta) -> state'`` (SPEC-2 §5.1).

This is the *single* state-transition primitive: both the oracle's truth and the
model's prediction become next-states by applying their respective deltas through
this same function (SPEC-2 §5.1 -- "only the deltas are compared/produced
differently"). It is pure: the input state is never mutated.
"""

from __future__ import annotations

from verisim.env.state import Dir, File, Last, Node, State

from .edits import (
    Chmod,
    Create,
    Delete,
    Delta,
    Modify,
    Move,
    SetCwd,
    SetEnv,
    SetResult,
)


def _subtree(fs: dict[str, Node], root: str) -> list[str]:
    """All paths at or under ``root`` (the node itself plus its descendants)."""
    prefix = root + "/"
    return [p for p in fs if p == root or p.startswith(prefix)]


def apply(state: State, delta: Delta) -> State:
    """Return a fresh state with ``delta`` applied. Does not mutate ``state``."""
    fs = dict(state.fs)
    cwd = state.cwd
    env = dict(state.env)
    last = state.last

    for edit in delta:
        if isinstance(edit, Create):
            fs[edit.path] = edit.node
        elif isinstance(edit, Delete):
            fs.pop(edit.path, None)
        elif isinstance(edit, Modify):
            old = fs[edit.path]
            mode = old.mode if isinstance(old, File) else 0o644
            fs[edit.path] = File(content=edit.content, mode=mode)
        elif isinstance(edit, Move):
            moved = _subtree(fs, edit.src)
            relocated: dict[str, Node] = {}
            for path in moved:
                new_path = edit.dst + path[len(edit.src) :]
                relocated[new_path] = fs[path]
            for path in moved:
                del fs[path]
            fs.update(relocated)
        elif isinstance(edit, Chmod):
            node = fs[edit.path]
            fs[edit.path] = (
                File(content=node.content, mode=edit.mode)
                if isinstance(node, File)
                else Dir(mode=edit.mode)
            )
        elif isinstance(edit, SetCwd):
            cwd = edit.path
        elif isinstance(edit, SetEnv):
            env[edit.key] = edit.token
        elif isinstance(edit, SetResult):
            last = Last(exit_code=edit.exit_code, stdout_hash=edit.stdout_hash)

    return State(fs=fs, cwd=cwd, env=env, last=last)
