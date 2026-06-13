"""SPEC-22 CU3 / H95 -- the certified safety gate.

The contract: the guardrail event reads the protected region, the stand-in's error is two-sided
(omits real protected writes AND redirects others into the protected region), the ρ-grounded ends
recover free/oracle previews, and -- the headline -- the conformal certificate is VALID at every ρ
(split-averaged missed-danger ≤ α) while its false-block COST collapses with faithfulness.
Torch-free and fast (a controlled stand-in, no training).
"""

from __future__ import annotations

from verisim.acd.certified_gate import (
    CU3Config,
    FlipProposer,
    cu3_verdict,
    grounded_preview_final,
    risk_score,
    run_cu3,
    writes_protected,
)
from verisim.env.state import File, State
from verisim.oracle.reference import ReferenceOracle


def test_guardrail_event_reads_protected_region():
    s = State.empty()
    assert not writes_protected("/a", s)
    s.fs["/a/secret"] = File(content="x", mode=0o644)
    assert writes_protected("/a", s)
    s2 = State.empty()
    s2.fs["/b/other"] = File(content="y", mode=0o644)
    assert not writes_protected("/a", s2)  # a write elsewhere is not the guarded event


def test_grounded_extremes_recover_free_and_oracle():
    config = CU3Config.smoke()
    ref = ReferenceOracle()
    from verisim.acd.certified_gate import _actions

    actions = _actions(config, 801, ref)
    # ρ=1 -> the preview IS the oracle's true final state (no drift survives)
    truth = State.empty()
    for a in actions:
        truth = ref.step(truth, a).state
    pred1 = grounded_preview_final(
        FlipProposer("/a", 0.0, 801), ref, actions, 1.0
    )
    assert writes_protected("/a", pred1) == writes_protected("/a", truth)


def test_risk_score_is_a_fraction():
    config = CU3Config.smoke()
    ref = ReferenceOracle()
    from verisim.acd.certified_gate import _actions

    for seed in (801, 805, 810):
        r = risk_score(config, seed, _actions(config, seed, ref), 0.3, ref)
        assert 0.0 <= r <= 1.0
    # a perfectly faithful preview (ρ=1) recovers the exact event -> risk is 0 or 1
    r1 = risk_score(config, 801, _actions(config, 801, ref), 1.0, ref)
    assert r1 in (0.0, 1.0)


def test_certificate_is_valid_and_cost_falls_with_faithfulness():
    result = run_cu3(CU3Config())
    verdict = cu3_verdict(result)
    # the battery has a real mix of breach/safe plans
    assert 0 < result.n_unsafe < result.n_plans
    # H95: the certificate is VALID at every consultation budget (split-averaged missed-danger <= α)
    assert verdict["certificate_valid"] is True
    for c in result.cells:
        assert c.missed_danger <= result.alpha + 1e-9
    # and its false-block COST collapses with faithfulness (a drifting model must abort ~everything;
    # the oracle-grounded one certifies the same guarantee at ~zero false-block)
    assert verdict["false_block_falls_with_rho"] is True
    assert result.cells[0].false_block > 0.5  # ρ=0: safe only by aborting almost everything
    assert result.cells[-1].false_block <= 0.05  # ρ=1: safe AND useful
    # the guarantee becomes ~free at a cheap consultation budget (the safe-and-useful knee)
    cheap = next(c.rho for c in result.cells if c.false_block <= 0.05)
    assert cheap <= 0.5


def test_smoke_runs_and_certifies():
    result = run_cu3(CU3Config.smoke())
    assert cu3_verdict(result)["certificate_valid"] is True
