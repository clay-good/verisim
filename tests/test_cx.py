"""Smoke + structural-invariant tests for the SPEC-17 causal experiments (CX0, CX1).

The committed CX tranche is pure-oracle (no learner, no GPU): CX0 is the SCM gate (H60), CX1 the
counterfactual effect-size law (H61). The learned-lift bets (CX2-CX4) are deferred to the trained
arm.
Tests assert the structural claims on tiny configs, not magnitudes (the macOS-first principle).
"""

from __future__ import annotations

from verisim.experiments.cx0 import CX0Config, run_cx0
from verisim.experiments.cx1 import CX1Config, run_cx1


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
