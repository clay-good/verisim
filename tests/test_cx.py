"""Smoke + structural-invariant tests for the SPEC-17 causal experiments (CX0, CX1, CX5).

The committed CX tranche is pure-oracle (no learner, no GPU): CX0 is the SCM gate (H60), CX1 the
counterfactual effect-size law (H61), CX5 the system-oracle fork (H64 — the SCM contract on the
real `/bin/sh` and Tier-B scheduler). The learned-lift bets (CX2-CX4) are deferred to the trained.
Tests assert the structural claims on tiny configs, not magnitudes (the macOS-first principle).
"""

from __future__ import annotations

from verisim.experiments.cx0 import CX0Config, run_cx0
from verisim.experiments.cx1 import CX1Config, run_cx1
from verisim.experiments.cx5 import CX5Config, run_cx5


def test_cx0_abduction_exact_all_worlds() -> None:
    stats = run_cx0(CX0Config(n_steps=24, n_seeds=6))
    assert {s.world for s in stats} == {"network", "host", "filesystem", "distributed"}
    for s in stats:
        assert s.abduction_exact_rate == 1.0  # H60: the oracle is an exact SCM
        assert 0.0 <= s.cf_differs_rate <= 1.0


def test_cx1_hidden_state_ordering() -> None:
    stats = run_cx1(CX1Config(n_steps=30, n_seeds=8, depths=(0.25, 0.5, 0.75)))
    amp = {s.world: s.amplification for s in stats}
    # The persistent-medium world amplifies most; the on-policy-complete network washes out.
    assert amp["distributed"] > amp["network"]
    assert amp["distributed"] > 1.0  # downstream effect exceeds the immediate one
    # The distributed world has a high consequential-intervention rate; the network a low one.
    cq = {s.world: s.consequential_rate for s in stats}
    assert cq["distributed"] > cq["network"]


def test_cx_determinism() -> None:
    a = [(s.world, s.amplification) for s in run_cx1(CX1Config(n_steps=24, n_seeds=4))]
    b = [(s.world, s.amplification) for s in run_cx1(CX1Config(n_steps=24, n_seeds=4))]
    assert a == b


def test_cx5_system_oracle_scm_transfer() -> None:
    # CX5 runs the real /bin/sh + Tier-B system oracles; a genuinely unavailable one self-skips
    # (available=False), never counted as a pass. Where it runs, abduction + rung-3 must be exact.
    stats = run_cx5(CX5Config(n_steps=12, n_seeds=3))
    assert {s.world for s in stats} == {"filesystem", "distributed"}
    ran = [s for s in stats if s.available]
    if not ran:
        import pytest

        pytest.skip("no system oracle available here (no /bin/sh or no thread support)")
    for s in ran:
        assert s.ref_abduction == 1.0  # the reference anchor (CX0)
        assert s.sys_abduction == 1.0  # H64: the system oracle is an exact SCM too
        assert s.sys_cf_exact == 1.0  # rung-3 counterfactuals are bit-reproducible on the system
        assert 0.0 <= s.sys_cf_differs <= 1.0


def test_cx5_deterministic() -> None:
    cfg = CX5Config(n_steps=12, n_seeds=3)
    a = [(s.world, s.sys_abduction, s.sys_cf_exact) for s in run_cx5(cfg)]
    b = [(s.world, s.sys_abduction, s.sys_cf_exact) for s in run_cx5(cfg)]
    assert a == b
