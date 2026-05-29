"""State `S` for the v0 shell/filesystem world (SPEC-2 §2.1).

A state is a fully serializable, canonicalizable snapshot: the filesystem tree,
the current working directory, the (small, fixed-keyset) environment, and the
result of the last action. It is *immutable by convention* -- the oracle and
``apply`` never mutate a state in place; they return a fresh one (SPEC-2 §3.1
purity guarantee).

v0 deviation from SPEC-2 §2.1, recorded in docs/semantics.md: file content is
stored inline (a string over the fixed content-token vocabulary, concatenated by
``append``) rather than in a separate content-addressable blob store. The
``content_hash`` is derived from it, so file equality is still hash equality and
states remain canonical and round-trippable from a single JSONL record without a
side store. The store is a pure optimization deferred until it pays for itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace


def content_hash(content: str) -> str:
    """Content-addressable hash of a file's bytes. Deterministic, no salt."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


EMPTY_HASH = content_hash("")


@dataclass(frozen=True)
class File:
    """A regular file. ``content`` is the source of truth; hash/size derive from it."""

    content: str = ""
    mode: int = 0o644

    @property
    def content_hash(self) -> str:
        return content_hash(self.content)

    @property
    def size(self) -> int:
        return len(self.content.encode("utf-8"))


@dataclass(frozen=True)
class Dir:
    """A directory."""

    mode: int = 0o755


Node = File | Dir


@dataclass(frozen=True)
class Last:
    """Observation of the last action (SPEC-2 §2.1 ``last``)."""

    exit_code: int = 0
    stdout_hash: str = EMPTY_HASH


@dataclass
class State:
    """The complete environment configuration.

    ``fs`` maps normalized absolute paths to nodes; root ``/`` is always present.
    Mutation is by convention forbidden -- use :func:`verisim.delta.apply.apply`,
    which builds a fresh ``State``.
    """

    fs: dict[str, Node]
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    last: Last = field(default_factory=Last)

    @staticmethod
    def empty() -> State:
        """The initial state: an empty root directory, cwd ``/``, no env vars."""
        return State(fs={"/": Dir()})

    def copy(self) -> State:
        """A deep-enough copy: new fs/env dicts; nodes and ``last`` are immutable."""
        return State(fs=dict(self.fs), cwd=self.cwd, env=dict(self.env), last=self.last)

    def with_last(self, exit_code: int, stdout_hash: str) -> State:
        return replace(self, last=Last(exit_code, stdout_hash))


# --- path helpers -----------------------------------------------------------


def normalize(path: str) -> str:
    """Normalize a path to a canonical absolute form, resolving ``.`` and ``..``.

    ``..`` at the root is clamped to the root (matching how a real kernel treats
    ``/..`` as ``/``). The result always begins with ``/`` and has no trailing
    slash except for the root itself.
    """
    parts: list[str] = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/" + "/".join(parts)


def resolve(cwd: str, path: str) -> str:
    """Resolve a (possibly relative) path argument against ``cwd`` to absolute."""
    base = path if path.startswith("/") else f"{cwd}/{path}"
    return normalize(base)


def parent(path: str) -> str:
    """The parent directory of a normalized absolute path. ``parent('/') == '/'``."""
    if path == "/":
        return "/"
    idx = path.rfind("/")
    return "/" if idx == 0 else path[:idx]


def basename(path: str) -> str:
    """The final segment of a path. ``basename('/') == ''``."""
    if path == "/":
        return ""
    return path[path.rfind("/") + 1 :]
