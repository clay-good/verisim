"""Action `A`: the fixed v0 command grammar (SPEC-2 §2.2).

An :class:`Action` is the parsed form of one shell command. Parsing is total
over the fixed grammar and rejects anything outside it, so the oracle only ever
sees well-formed commands; *semantic* validity (does the target exist? is it a
dir?) is the oracle's job, not the parser's.
"""

from __future__ import annotations

from dataclasses import dataclass

# Commands taking a single path argument.
_UNARY = {"mkdir", "rmdir", "touch", "cd", "cat", "ls"}
# Commands taking <path> <token>.
_PATH_TOKEN = {"write", "append"}
# Commands taking two paths.
_BINARY = {"mv", "cp"}


@dataclass(frozen=True)
class Action:
    """A parsed command. ``raw`` is the canonical string form.

    Argument layout by ``name``:
      - mkdir/rmdir/touch/cd/cat/ls : args = (path,)
      - rm                          : args = (path,)   ; recursive set by ``-r``
      - write/append                : args = (path, token)
      - mv                          : args = (src, dst)
      - cp                          : args = (src, dst) ; recursive set by ``-r``
      - chmod                       : args = (mode_octal_str, path)
      - export                      : args = (key, token)
    """

    raw: str
    name: str
    args: tuple[str, ...]
    recursive: bool = False


class ParseError(ValueError):
    """Raised when a string is not a valid command in the v0 grammar."""


def parse_action(raw: str) -> Action:
    """Parse a command string into an :class:`Action`, or raise :class:`ParseError`."""
    text = raw.strip()
    if not text:
        raise ParseError("empty command")
    toks = text.split()
    name = toks[0]
    rest = toks[1:]

    if name == "export":
        # export <KEY>=<token>  (single token, contains exactly one '=')
        if len(rest) != 1 or "=" not in rest[0]:
            raise ParseError(f"bad export: {raw!r}")
        key, _, token = rest[0].partition("=")
        if not key:
            raise ParseError(f"bad export key: {raw!r}")
        return Action(raw=f"export {key}={token}", name="export", args=(key, token))

    if name == "chmod":
        if len(rest) != 2:
            raise ParseError(f"chmod needs <mode> <path>: {raw!r}")
        mode_str, path = rest
        try:
            int(mode_str, 8)
        except ValueError as exc:
            raise ParseError(f"bad octal mode {mode_str!r}") from exc
        return Action(raw=f"chmod {mode_str} {path}", name="chmod", args=(mode_str, path))

    if name == "rm":
        recursive = False
        if rest and rest[0] == "-r":
            recursive = True
            rest = rest[1:]
        if len(rest) != 1:
            raise ParseError(f"rm needs one path: {raw!r}")
        canon = f"rm -r {rest[0]}" if recursive else f"rm {rest[0]}"
        return Action(raw=canon, name="rm", args=(rest[0],), recursive=recursive)

    if name == "cp":
        recursive = False
        if rest and rest[0] == "-r":
            recursive = True
            rest = rest[1:]
        if len(rest) != 2:
            raise ParseError(f"cp needs <src> <dst>: {raw!r}")
        canon = f"cp -r {rest[0]} {rest[1]}" if recursive else f"cp {rest[0]} {rest[1]}"
        return Action(raw=canon, name="cp", args=(rest[0], rest[1]), recursive=recursive)

    if name in _BINARY:  # mv
        if len(rest) != 2:
            raise ParseError(f"{name} needs <src> <dst>: {raw!r}")
        return Action(raw=f"{name} {rest[0]} {rest[1]}", name=name, args=(rest[0], rest[1]))

    if name in _PATH_TOKEN:
        if len(rest) != 2:
            raise ParseError(f"{name} needs <path> <token>: {raw!r}")
        return Action(raw=f"{name} {rest[0]} {rest[1]}", name=name, args=(rest[0], rest[1]))

    if name in _UNARY:
        if len(rest) != 1:
            raise ParseError(f"{name} needs one path: {raw!r}")
        return Action(raw=f"{name} {rest[0]}", name=name, args=(rest[0],))

    raise ParseError(f"unknown command {name!r}")
