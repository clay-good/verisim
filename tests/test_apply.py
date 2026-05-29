"""`apply` is total over grammar-valid deltas (SPEC-2 §5.2).

The neural model is a stochastic proposer: it can emit a grammar-valid delta that
references a missing path or the wrong node type. `apply` must absorb that as a
best-effort no-op (the oracle/divergence catches the error) rather than raise --
otherwise the propose-verify-correct loop would crash mid-rollout.
"""

from __future__ import annotations

from verisim.delta import Chmod, Create, Delete, Modify, Move, apply
from verisim.env import Dir, File, State


def _state() -> State:
    return State(fs={"/": Dir(), "/f": File(content="alpha"), "/d": Dir()})


def test_modify_missing_path_is_noop():
    s = _state()
    out = apply(s, [Modify("/ghost", "beta")])
    assert out.fs == s.fs  # unchanged, no KeyError


def test_modify_on_directory_is_noop():
    s = _state()
    out = apply(s, [Modify("/d", "beta")])
    assert isinstance(out.fs["/d"], Dir)


def test_modify_existing_file_keeps_mode():
    s = _state()
    s.fs["/f"] = File(content="alpha", mode=0o600)
    out = apply(s, [Modify("/f", "omega")])
    assert out.fs["/f"] == File(content="omega", mode=0o600)


def test_chmod_missing_path_is_noop():
    s = _state()
    out = apply(s, [Chmod("/ghost", 0o700)])
    assert out.fs == s.fs


def test_move_missing_source_is_noop():
    s = _state()
    out = apply(s, [Move("/ghost", "/x")])
    assert out.fs == s.fs


def test_create_and_delete_are_total():
    s = _state()
    out = apply(s, [Create("/n", File(content="beta")), Delete("/ghost")])
    assert out.fs["/n"] == File(content="beta")
    assert "/ghost" not in out.fs
