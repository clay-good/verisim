"""SandboxOracle -- the real-/bin/sh system oracle behind the v0 Oracle protocol (SPEC-11 §2).

The system-oracle tests assert *platform-independent* properties (the SPEC-11 §2.5 macOS-first
principle): they pass on the macOS development host and, for free, on Linux CI. The headline --
bit-exact agreement on the structure-building grammar -- is the regime v0 is designed to model
and is pure POSIX, so it reproduces identically everywhere.
"""

from __future__ import annotations

import random

import pytest

from verisim.data.drivers import Driver
from verisim.env.action import parse_action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.serialize import to_canonical_str
from verisim.env.state import Dir, State
from verisim.oracle.base import Oracle
from verisim.oracle.differential import canonical_world
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    _ORACLE: SandboxOracle | None = SandboxOracle()
except SystemOracleUnavailable:  # pragma: no cover - only on a host with no shell
    _ORACLE = None

requires_shell = pytest.mark.skipif(_ORACLE is None, reason="no real shell (disclosed)")


@requires_shell
def test_sandbox_oracle_satisfies_protocol():
    assert isinstance(SandboxOracle(), Oracle)


@requires_shell
def test_structure_building_grammar_is_bit_exact():
    """The headline: on the modeled (structure-building) regime, ref == real /bin/sh, bit-exact."""
    ref = ReferenceOracle()
    sysorc = SandboxOracle()
    for driver in ("structural", "trivial"):
        for seed in range(6):
            drv = Driver(driver, DEFAULT_CONFIG, random.Random(seed))
            s = State.empty()
            for _ in range(25):
                a = drv.sample(s)
                r_ref = ref.step(s, a)
                r_sys = sysorc.step(s, a)
                assert canonical_world(r_ref.state) == canonical_world(r_sys.state), a.raw
                assert r_ref.exit_code == r_sys.exit_code, a.raw
                s = r_ref.state


@requires_shell
def test_reset_and_determinism_report():
    sysorc = SandboxOracle()
    s = State.empty()
    assert sysorc.reset(s).fs == s.fs
    rep = sysorc.determinism_report()
    assert rep.clock_sealed and rep.rng_sealed and rep.concurrency_sealed and rep.env_leakage_sealed


@requires_shell
def test_hermeticity_report_channels():
    h = SandboxOracle().hermeticity()
    assert h.fs_confined and h.network_blocked and h.no_privilege_gain
    assert h.no_persistence and h.resource_capped


@requires_shell
def test_basic_commands_match_a_real_shell():
    ref = ReferenceOracle()
    sysorc = SandboxOracle()
    s = State.empty()
    for raw in ("mkdir /a", "write /a/f alpha", "append /a/f beta", "cp /a/f /a/g",
                "chmod 600 /a/f", "mkdir /a/b", "mv /a/g /a/b/h", "rm /a/f"):
        a = parse_action(raw)
        assert canonical_world(ref.step(s, a).state) == canonical_world(sysorc.step(s, a).state), raw  # noqa: E501
        s = ref.step(s, a).state


@requires_shell
def test_cat_stdout_matches():
    ref = ReferenceOracle()
    sysorc = SandboxOracle()
    s = State.empty()
    for raw in ("write /f alpha", "append /f beta"):
        s = ref.step(s, parse_action(raw)).state
    a = parse_action("cat /f")
    assert ref.step(s, a).stdout == sysorc.step(s, a).stdout == "alphabeta"


def test_off_grammar_action_is_refused():
    """An action with no real-kernel rendering is refused before exec (the network allowlist)."""
    if _ORACLE is None:  # pragma: no cover
        pytest.skip("no shell")
    from dataclasses import replace

    bad = replace(parse_action("mkdir /a"), name="curl", args=("http://evil/x",))
    with pytest.raises(SystemOracleUnavailable):
        _ORACLE.step(State.empty(), bad)


@requires_shell
def test_rm_minus_r_root_does_not_break_the_harness():
    """``rm -r /`` (which v0 forbids) at most removes the fs-root subdir; the harness survives."""
    sysorc = SandboxOracle()
    s = State.empty()
    s = sysorc.reset(State(fs={"/": Dir(), "/a": Dir()}))
    result = sysorc.step(s, parse_action("rm -r /"))
    # the system oracle returns *some* coherent state (it does not raise); v0 reference keeps root
    assert isinstance(to_canonical_str(result.state), str)


def test_reference_subtree_move_bug_is_fixed():
    """Regression: mv/cp of a directory into its own subtree must not orphan the subtree.

    Surfaced by the SPEC-11 differential harness (H28). Pre-fix, ``mv /e /e/e`` produced an
    invalid state with ``/e/e`` whose parent ``/e`` no longer existed.
    """
    ref = ReferenceOracle()
    s = State(fs={"/": Dir(), "/e": Dir(), "/e/c": Dir()})
    for raw in ("mv /e /e/e", "cp -r /e /e/sub"):
        out = ref.step(s, parse_action(raw))
        assert out.exit_code == 1, f"{raw} should be rejected"
        # every non-root path's parent still exists and is a directory (a valid state)
        for p in out.state.fs:
            if p == "/":
                continue
            parent = p.rsplit("/", 1)[0] or "/"
            assert isinstance(out.state.fs.get(parent), Dir), f"{raw} orphaned {p}"
