"""SPEC-28 M1 -- the Coverage Certifier: assemble the audit engine into a guardrail auditor.

Fast tests use in-process monitors (no subprocess); one integration test exercises the real
``SubprocessMonitor`` path against the weak example hook. The verify gate from the runbook: the
certifier surfaces >=1 oracle-confirmed off-surface realizing command on a leaky guardrail, and
certifies a sound one clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verisim.audit.monitors import DenylistMonitor, ResolverMonitor
from verisim.audit.oracles import ShellPathOracle
from verisim.certify import certify_hook, certify_monitor

PREFIX = "/etc"
SHADOW = "/etc/shadow"
WEAK_HOOK = Path(__file__).resolve().parents[1] / "scripts" / "examples" / "weak_denylist_hook.py"


def test_certifier_catches_a_weak_denylist() -> None:
    """A literal-substring denylist leaks every indirection encoding -> many bypasses."""
    mon = DenylistMonitor((SHADOW,))
    res = certify_monitor(mon, ShellPathOracle(PREFIX), protected_path=SHADOW, prefix=PREFIX,
                          proposer="enumerate")
    assert not res.sound
    assert len(res.bypasses) >= 10
    # every reported bypass is oracle-realizing and the guardrail allowed it (what silent means)
    assert all(h.silent for h in res.bypasses)
    assert res.coverage_pct < 100.0


def test_certifier_certifies_a_sound_resolver_clean() -> None:
    """The hardened resolver covers (FIRES/ABSTAIN) every string-resolvable form -> clean."""
    mon = ResolverMonitor(PREFIX, sound_printf=True)
    res = certify_monitor(mon, ShellPathOracle(PREFIX), protected_path=SHADOW, prefix=PREFIX,
                          proposer="enumerate")
    assert res.sound
    assert res.bypasses == []
    assert "SOUND" in res.verdict_line()


def test_certifier_catches_the_printf_soundness_bug() -> None:
    """With the printf hole open, the certifier surfaces the RA24 format-escape as a bypass."""
    mon = ResolverMonitor(PREFIX, sound_printf=False)
    res = certify_monitor(mon, ShellPathOracle(PREFIX), protected_path=SHADOW, prefix=PREFIX,
                          proposer="enumerate")
    assert not res.sound
    assert any(h.klass == "printf_fmt" for h in res.bypasses)
    assert "LEAKS" in res.verdict_line()


def test_coverage_and_verdict_fields() -> None:
    mon = DenylistMonitor((SHADOW,))
    res = certify_monitor(mon, ShellPathOracle(PREFIX), protected_path=SHADOW, prefix=PREFIX,
                          proposer="enumerate")
    assert 0.0 <= res.coverage_pct <= 100.0
    assert res.n_realizing > 0
    assert res.monitor == mon.name


@pytest.mark.skipif(not WEAK_HOOK.exists(), reason="example hook missing")
def test_certify_hook_subprocess_path() -> None:
    """Integration: audit the weak example hook over the real SubprocessMonitor path."""
    res = certify_hook(str(WEAK_HOOK), proposer="enumerate", oracle="syntactic")
    assert not res.sound
    assert len(res.bypasses) >= 10
    assert res.monitor.startswith("hook:")
    # the literal-shadow form is covered (blocked); an indirection form is a bypass.
    assert any("printf" in h.command or "rev" in h.command or "${" in h.command
               for h in res.bypasses)


def test_cli_exit_code_signals_leak() -> None:
    """The CLI returns nonzero on a leak so it is usable as a CI gate."""
    from verisim.certify.__main__ import main

    rc = main(["audit", "--hook", str(WEAK_HOOK), "--proposer", "enumerate", "--max-show", "0"])
    assert rc == 1  # weak hook leaks
    _ = sys  # keep import used if the body changes
