"""Hand-authored semantics goldens (SPEC-2 §16).

Each case fixes a specific (state, action) and asserts the resulting state facts
by hand, from docs/semantics.md -- not regenerated from the oracle, so these pin
the semantics and CI fails on any drift. Covers each command's success path and
its documented failure modes.
"""

from __future__ import annotations

from verisim.env import Dir, File, State, parse_action
from verisim.oracle import ReferenceOracle


def run(commands: list[str], start: State | None = None) -> State:
    oracle = ReferenceOracle()
    state = start if start is not None else State.empty()
    for cmd in commands:
        state = oracle.step(state, parse_action(cmd)).state
    return state


def step(cmd: str, start: State | None = None):
    oracle = ReferenceOracle()
    return oracle.step(start if start is not None else State.empty(), parse_action(cmd))


def content_at(state: State, path: str) -> str:
    node = state.fs[path]
    assert isinstance(node, File)
    return node.content


# --- create / build ---------------------------------------------------------


def test_mkdir_creates_dir():
    s = run(["mkdir /a"])
    assert isinstance(s.fs["/a"], Dir)


def test_mkdir_fails_if_exists():
    r = step("mkdir /a", run(["mkdir /a"]))
    assert r.exit_code == 1
    assert sorted(r.state.fs) == ["/", "/a"]


def test_mkdir_fails_without_parent():
    r = step("mkdir /a/b")
    assert r.exit_code == 1
    assert sorted(r.state.fs) == ["/"]


def test_touch_creates_empty_file():
    s = run(["touch /f"])
    assert s.fs["/f"] == File(content="")


def test_touch_existing_is_noop_success():
    r = step("touch /a", run(["mkdir /a"]))
    assert r.exit_code == 0
    assert isinstance(r.state.fs["/a"], Dir)  # unchanged


def test_write_then_append_concatenates():
    s = run(["write /f alpha", "append /f beta"])
    assert isinstance(s.fs["/f"], File)
    assert content_at(s, "/f") == "alphabeta"


def test_write_on_dir_fails():
    r = step("write /a alpha", run(["mkdir /a"]))
    assert r.exit_code == 1
    assert isinstance(r.state.fs["/a"], Dir)


def test_append_creates_when_absent():
    s = run(["append /f gamma"])
    assert content_at(s, "/f") == "gamma"


# --- remove -----------------------------------------------------------------


def test_rm_file():
    s = run(["touch /f", "rm /f"])
    assert "/f" not in s.fs


def test_rm_dir_without_recursive_fails():
    r = step("rm /a", run(["mkdir /a"]))
    assert r.exit_code == 1
    assert "/a" in r.state.fs


def test_rm_recursive_cascades():
    s = run(["mkdir /a", "mkdir /a/b", "touch /a/b/f", "rm -r /a"])
    assert sorted(s.fs) == ["/"]


def test_rmdir_nonempty_fails():
    r = step("rmdir /a", run(["mkdir /a", "touch /a/f"]))
    assert r.exit_code == 1
    assert "/a/f" in r.state.fs


def test_rmdir_empty_succeeds():
    s = run(["mkdir /a", "rmdir /a"])
    assert "/a" not in s.fs


# --- move / copy ------------------------------------------------------------


def test_mv_renames_file():
    s = run(["write /f alpha", "mv /f /g"])
    assert "/f" not in s.fs
    assert content_at(s, "/g") == "alpha"


def test_mv_dir_moves_subtree():
    s = run(["mkdir /a", "touch /a/x", "mkdir /a/sub", "mv /a /b"])
    assert sorted(s.fs) == ["/", "/b", "/b/sub", "/b/x"]


def test_mv_into_existing_dir():
    s = run(["touch /f", "mkdir /d", "mv /f /d"])
    assert "/f" not in s.fs
    assert "/d/f" in s.fs


def test_mv_onto_existing_target_fails():
    r = step("mv /f /g", run(["touch /f", "touch /g"]))
    assert r.exit_code == 1
    assert "/f" in r.state.fs and "/g" in r.state.fs


def test_cp_file_keeps_source():
    s = run(["write /f alpha", "cp /f /g"])
    assert content_at(s, "/f") == "alpha"
    assert content_at(s, "/g") == "alpha"


def test_cp_dir_without_recursive_fails():
    r = step("cp /a /b", run(["mkdir /a"]))
    assert r.exit_code == 1
    assert "/b" not in r.state.fs


def test_cp_recursive_copies_subtree():
    s = run(["mkdir /a", "write /a/f alpha", "cp -r /a /b"])
    assert content_at(s, "/a/f") == "alpha"
    assert content_at(s, "/b/f") == "alpha"


def test_cp_overwrites_existing_file():
    s = run(["write /f alpha", "write /g beta", "cp /f /g"])
    assert content_at(s, "/g") == "alpha"


# --- chmod / cd / read-only -------------------------------------------------


def test_chmod_sets_mode():
    s = run(["touch /f", "chmod 600 /f"])
    assert s.fs["/f"].mode == 0o600


def test_cd_into_dir():
    s = run(["mkdir /a", "cd /a"])
    assert s.cwd == "/a"


def test_cd_into_file_fails():
    r = step("cd /f", run(["touch /f"]))
    assert r.exit_code == 1
    assert r.state.cwd == "/"


def test_relative_paths_resolve_against_cwd():
    s = run(["mkdir /a", "cd /a", "touch f"])
    assert "/a/f" in s.fs


def test_cat_emits_content():
    r = step("cat /f", run(["write /f omega"]))
    assert r.exit_code == 0
    assert r.stdout == "omega"


def test_cat_missing_fails():
    r = step("cat /nope")
    assert r.exit_code == 1
    assert r.stdout == ""


def test_ls_lists_sorted_direct_children():
    r = step("ls /", run(["mkdir /b", "mkdir /a", "touch /c"]))
    assert r.stdout == "a\nb\nc"


def test_export_sets_env():
    s = run(["export HOME=alpha"])
    assert s.env["HOME"] == "alpha"


def test_failure_leaves_state_unchanged_but_sets_exit():
    base = run(["mkdir /a"])
    r = step("rmdir /missing", base)
    assert r.exit_code == 1
    assert r.state.fs == base.fs
    assert r.state.cwd == base.cwd
    assert r.state.last.exit_code == 1
