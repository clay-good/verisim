"""Canonical serialization of states (SPEC-2 §2.1).

There is exactly one serialized form for a given state: paths sorted, env keys
sorted, nodes tagged by type. Two semantically equal states serialize byte-for-
byte identically -- the precondition for the divergence metric (SPEC-2 §7.1) and
for exact-match scoring. Round-tripping ``from_canonical(to_canonical(s)) == s``
is an identity (tested in M0).
"""

from __future__ import annotations

import json
from typing import Any

from .state import Dir, File, Last, Node, State


def node_to_dict(node: Node) -> dict[str, Any]:
    if isinstance(node, File):
        return {
            "type": "file",
            "mode": node.mode,
            "content": node.content,
            "content_hash": node.content_hash,
            "size": node.size,
        }
    return {"type": "dir", "mode": node.mode}


def node_from_dict(d: dict[str, Any]) -> Node:
    if d["type"] == "file":
        return File(content=d["content"], mode=d["mode"])
    return Dir(mode=d["mode"])


def to_canonical(state: State) -> dict[str, Any]:
    """The canonical dict form: deterministic key/element ordering throughout."""
    return {
        "fs": [[path, node_to_dict(state.fs[path])] for path in sorted(state.fs)],
        "cwd": state.cwd,
        "env": dict(sorted(state.env.items())),
        "last": {"exit_code": state.last.exit_code, "stdout_hash": state.last.stdout_hash},
    }


def from_canonical(d: dict[str, Any]) -> State:
    fs = {path: node_from_dict(node) for path, node in d["fs"]}
    last = Last(exit_code=d["last"]["exit_code"], stdout_hash=d["last"]["stdout_hash"])
    return State(fs=fs, cwd=d["cwd"], env=dict(d["env"]), last=last)


def to_canonical_str(state: State) -> str:
    """Canonical JSON string; stable across processes (sorted keys, no spaces)."""
    return json.dumps(to_canonical(state), sort_keys=True, separators=(",", ":"))
