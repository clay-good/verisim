"""The differential-validation harness (SPEC-11 §3): differential_step + the divergence classifier.

The classifier tests are pure (no shell): they assert that the named v0 modeling boundaries are
correctly identified from the input ``(state, action)`` alone -- so the SY1 residual (unexplained
disagreements) is provably zero by construction.
"""

from __future__ import annotations

import random

import pytest

from verisim.data.drivers import Driver
from verisim.env.action import parse_action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import Dir, File, State
from verisim.oracle.differential import (
    AGREE,
    BOUNDARY_CLASSES,
    C_OVERWRITE,
    C_PERMISSION,
    C_ROOT,
    C_SELF_SUBTREE,
    RESIDUAL,
    canonical_world,
    classify_divergence,
    differential_step,
)
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    _SYS: SandboxOracle | None = SandboxOracle()
except SystemOracleUnavailable:  # pragma: no cover
    _SYS = None

requires_shell = pytest.mark.skipif(_SYS is None, reason="no real shell")


def test_canonical_world_excludes_last():
    s1 = State(fs={"/": Dir()}).with_last(0, "x")
    s2 = State(fs={"/": Dir()}).with_last(1, "y")
    assert canonical_world(s1) == canonical_world(s2)  # different last, same world


def test_classify_root_protection():
    s = State(fs={"/": Dir(), "/a": Dir()})
    assert classify_divergence(s, parse_action("rm -r /")) == C_ROOT
    assert classify_divergence(s, parse_action("rmdir /")) == C_ROOT
    assert classify_divergence(s, parse_action("cp -r / /a")) == C_ROOT


def test_classify_overwrite_policy():
    s = State(fs={"/": Dir(), "/a": File(content="x"), "/b": File(content="y")})
    assert classify_divergence(s, parse_action("mv /a /b")) == C_OVERWRITE


def test_classify_permission_enforcement():
    # a non-executable directory blocks traversal in the kernel but not in v0
    s = State(fs={"/": Dir(), "/d": Dir(mode=0o600), "/d/f": File(content="x")})
    assert classify_divergence(s, parse_action("cat /d/f")) == C_PERMISSION
    # the root itself non-executable blocks everything beneath it
    s2 = State(fs={"/": Dir(mode=0o600), "/a": Dir()})
    assert classify_divergence(s2, parse_action("mkdir /a/b")) == C_PERMISSION


def test_classify_self_subtree():
    s = State(fs={"/": Dir(), "/e": Dir()})
    assert classify_divergence(s, parse_action("cp -r /e /e/sub")) == C_SELF_SUBTREE


@requires_shell
def test_differential_step_agrees_on_structure_grammar():
    assert _SYS is not None
    ref = ReferenceOracle()
    s = State.empty()
    for raw in ("mkdir /a", "touch /a/f", "write /a/f alpha"):
        a = parse_action(raw)
        rec = differential_step(s, a, ref, _SYS)
        assert rec.agree and rec.divergence_class == AGREE
        s = ref.step(s, a).state


@requires_shell
def test_every_divergence_is_a_named_boundary_residual_is_zero():
    """The SY1 completeness invariant: across the whole grammar, residual disagreements = 0."""
    assert _SYS is not None
    ref = ReferenceOracle()
    residual = 0
    for driver in ("weighted", "adversarial"):
        for seed in range(15):
            drv = Driver(driver, DEFAULT_CONFIG, random.Random(seed))
            s = State.empty()
            for _ in range(40):
                a = drv.sample(s)
                rec = differential_step(s, a, ref, _SYS)
                if not rec.agree:
                    assert rec.divergence_class in BOUNDARY_CLASSES, (a.raw, rec.divergence_class)
                    if rec.divergence_class == RESIDUAL:
                        residual += 1
                s = ref.step(s, a).state
    assert residual == 0
