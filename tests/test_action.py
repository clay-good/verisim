"""Command grammar parsing (SPEC-2 §2.2)."""

from __future__ import annotations

import pytest

from verisim.env import parse_action
from verisim.env.action import ParseError


def test_parses_each_command_shape():
    cases = {
        "mkdir /a": ("mkdir", ("/a",), False),
        "rm /a": ("rm", ("/a",), False),
        "rm -r /a": ("rm", ("/a",), True),
        "cp /a /b": ("cp", ("/a", "/b"), False),
        "cp -r /a /b": ("cp", ("/a", "/b"), True),
        "mv /a /b": ("mv", ("/a", "/b"), False),
        "write /a alpha": ("write", ("/a", "alpha"), False),
        "append /a beta": ("append", ("/a", "beta"), False),
        "chmod 755 /a": ("chmod", ("755", "/a"), False),
        "cd /a": ("cd", ("/a",), False),
        "cat /a": ("cat", ("/a",), False),
        "ls /a": ("ls", ("/a",), False),
        "export HOME=alpha": ("export", ("HOME", "alpha"), False),
    }
    for raw, (name, args, recursive) in cases.items():
        action = parse_action(raw)
        assert (action.name, action.args, action.recursive) == (name, args, recursive)
        assert action.raw == raw


def test_extra_whitespace_canonicalized():
    assert parse_action("  mkdir   /a  ").raw == "mkdir /a"


@pytest.mark.parametrize(
    "raw",
    ["", "bogus /a", "mkdir", "mkdir /a /b", "chmod /a", "chmod 9z9 /a", "export NOEQ", "rm"],
)
def test_invalid_commands_raise(raw: str):
    with pytest.raises(ParseError):
        parse_action(raw)
