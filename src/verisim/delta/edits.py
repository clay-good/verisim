"""Delta types: the structured set of edits an action makes (SPEC-2 §5.1).

A ``Delta`` is the model's prediction target and the oracle's truth representation
(SPEC.md §6.1: delta prediction bounds the hallucination surface and localizes
verification). Each edit is a small frozen dataclass; the union is closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.env.state import Node


@dataclass(frozen=True)
class Create:
    path: str
    node: Node


@dataclass(frozen=True)
class Delete:
    path: str


@dataclass(frozen=True)
class Modify:
    path: str
    content: str


@dataclass(frozen=True)
class Move:
    """Relocate a path and, if it is a directory, its entire subtree."""

    src: str
    dst: str


@dataclass(frozen=True)
class Chmod:
    path: str
    mode: int


@dataclass(frozen=True)
class SetCwd:
    path: str


@dataclass(frozen=True)
class SetEnv:
    key: str
    token: str


@dataclass(frozen=True)
class SetResult:
    exit_code: int
    stdout_hash: str


Edit = Create | Delete | Modify | Move | Chmod | SetCwd | SetEnv | SetResult
Delta = list[Edit]
