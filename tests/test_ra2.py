"""Tests for SPEC-22 RA2 (H134): the real-LLM safety gate is anchor-invariant against a real kernel.

The comparison machinery is exercised hermetically (reference-vs-reference, max Δ = 0). The real
``/bin/sh`` check is SPEC-11 §2.5-gated: it runs when a shell is present and is a disclosed skip
otherwise -- never counted as agreement. The empirical claim is that the agent-safety verdict
(missed-danger / task-success / cost) is bit-identical whether the agent acts on the reference
oracle or a real kernel, because the env grammar is bit-exact between the two (SY1).
"""

from __future__ import annotations

import pytest

from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable
from verisim.realagent.real_kernel import anchor_delta, ra2_verdict, run_ra2

try:
    SandboxOracle()
    _HAVE_SHELL = True
except SystemOracleUnavailable:  # pragma: no cover
    _HAVE_SHELL = False

requires_shell = pytest.mark.skipif(not _HAVE_SHELL, reason="no real shell (SPEC-11 §2.5)")


def test_comparison_machinery_is_hermetic() -> None:
    # ref-vs-ref forces the "available" path with a deterministic non-shell oracle: Δ must be 0
    result = run_ra2(sys_oracle=ReferenceOracle())
    assert result.sys is not None
    assert anchor_delta(result) == 0.0


def test_ra1_headline_holds_on_the_reference_arm() -> None:
    v = ra2_verdict(run_ra2(sys_oracle=ReferenceOracle()))
    assert v["undefended_breaches"] is True
    assert v["gate_drives_to_zero"] is True
    assert v["no_utility_loss"] is True
    assert v["cheaper_than_full_oracle"] is True


@requires_shell
def test_anchor_invariant_against_a_real_kernel() -> None:
    # the central RA2 claim: the agent-safety verdict is bit-identical against a real /bin/sh
    result = run_ra2()
    assert result.available is True
    assert anchor_delta(result) == 0.0


@requires_shell
def test_real_kernel_carries_the_full_headline() -> None:
    v = ra2_verdict(run_ra2())
    assert v["available"] is True
    assert v["anchor_invariant"] is True
    # undefended breaches on the injection, the covering gate catches it cheaply -- on a real kernel
    assert v["undefended_breaches"] is True
    assert v["gate_drives_to_zero"] is True
    assert v["no_utility_loss"] is True
    assert v["cheaper_than_full_oracle"] is True


@requires_shell
def test_real_kernel_cells_match_reference_cell_for_cell() -> None:
    result = run_ra2()
    assert result.sys is not None
    ref = {c.schedule: c for c in result.ref.cells}
    sysr = {c.schedule: c for c in result.sys.cells}
    for schedule, rc in ref.items():
        sc = sysr[schedule]
        assert rc.missed_danger_rate == sc.missed_danger_rate
        assert rc.task_success_rate == sc.task_success_rate
        assert rc.mean_oracle_calls == sc.mean_oracle_calls
