"""Delta <-> canonical dict/JSON serialization (SPEC-2 §4 trajectory records).

Each edit serializes to a dict tagged by ``op``. Round-tripping is an identity,
tested in M1.
"""

from __future__ import annotations

from typing import Any

from verisim.env.serialize import node_from_dict, node_to_dict

from .edits import (
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


def edit_to_dict(edit: Edit) -> dict[str, Any]:
    if isinstance(edit, Create):
        return {"op": "create", "path": edit.path, "node": node_to_dict(edit.node)}
    if isinstance(edit, Delete):
        return {"op": "delete", "path": edit.path}
    if isinstance(edit, Modify):
        return {"op": "modify", "path": edit.path, "content": edit.content}
    if isinstance(edit, Move):
        return {"op": "move", "src": edit.src, "dst": edit.dst}
    if isinstance(edit, Chmod):
        return {"op": "chmod", "path": edit.path, "mode": edit.mode}
    if isinstance(edit, SetCwd):
        return {"op": "setcwd", "path": edit.path}
    if isinstance(edit, SetEnv):
        return {"op": "setenv", "key": edit.key, "token": edit.token}
    return {"op": "setresult", "exit_code": edit.exit_code, "stdout_hash": edit.stdout_hash}


def edit_from_dict(d: dict[str, Any]) -> Edit:
    op = d["op"]
    if op == "create":
        return Create(path=d["path"], node=node_from_dict(d["node"]))
    if op == "delete":
        return Delete(path=d["path"])
    if op == "modify":
        return Modify(path=d["path"], content=d["content"])
    if op == "move":
        return Move(src=d["src"], dst=d["dst"])
    if op == "chmod":
        return Chmod(path=d["path"], mode=d["mode"])
    if op == "setcwd":
        return SetCwd(path=d["path"])
    if op == "setenv":
        return SetEnv(key=d["key"], token=d["token"])
    if op == "setresult":
        return SetResult(exit_code=d["exit_code"], stdout_hash=d["stdout_hash"])
    raise ValueError(f"unknown edit op {op!r}")


def delta_to_list(delta: Delta) -> list[dict[str, Any]]:
    return [edit_to_dict(e) for e in delta]


def delta_from_list(items: list[dict[str, Any]]) -> Delta:
    return [edit_from_dict(d) for d in items]
