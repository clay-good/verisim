"""CU28 / H121 -- the targeting result against a real /bin/sh (SPEC-22 §5).

Torch-free core (the targeting schedule is model-free -- no world model is trained or run): the
reference-oracle arm proves the covering target is safe + un-gameable + cheap and the asset
shortcut leaks; a ``skipif``-guarded test proves the whole verdict is anchor-invariant against a
real ``/bin/sh`` (bit-identical, max Δ = 0), and a no-shell test proves the clean skip.
"""

from __future__ import annotations

import pytest

from verisim.acd.realkernel_targeting import (
    CU28Config,
    anchor_delta,
    build_deployments,
    cu28_verdict,
    fs_content_arm,
    run_cu28,
)
from verisim.acd.unified_targeting import arm_verdict, run_arm
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    SandboxOracle()
    _HAVE_SHELL = True
except SystemOracleUnavailable:  # pragma: no cover
    _HAVE_SHELL = False

requires_shell = pytest.mark.skipif(not _HAVE_SHELL, reason="no real shell")


def test_covering_target_is_safe_ungameable_cheap_and_covers():
    """The grammar-indexed covering target: 0 random / 0 adversarial, covers, < full oracle."""
    arm = run_arm(fs_content_arm(ReferenceOracle(), CU28Config.smoke()))
    v = arm_verdict(arm)
    assert v["target_random_breach"] == 0.0
    assert v["target_adversarial_breach"] == 0.0
    assert v["target_covers"] is True
    assert v["target_is_ungameable"] is True
    assert v["target_cheaper_than_full"] is True


def test_asset_shortcut_leaks_and_breaks_coverage():
    """The asset-indexed shortcut (CU12 boundary): catches the known credential on the benign run
    (random 0) but is fully gameable to the unflagged path (adversarial 1) -- false security."""
    arm = run_arm(fs_content_arm(ReferenceOracle(), CU28Config.smoke()))
    v = arm_verdict(arm)
    assert v["shortcut_covers"] is False
    assert v["shortcut_leaks"] is True
    assert v["shortcut_adversarial_breach"] == 1.0


def test_free_breaches_model_fails_uniform_gameable():
    """The controls behave as the arc predicts: a free agent breaches, the omitter can't
    self-target, the uniform clock is gameable."""
    arm = run_arm(fs_content_arm(ReferenceOracle(), CU28Config.smoke()))
    v = arm_verdict(arm)
    free_breach = v["free_breach_rate"]
    assert isinstance(free_breach, float) and free_breach > 0.5
    assert v["model_self_targeting_fails"] is True
    assert v["uniform_is_gameable"] is True
    assert v["oracle_self_governs"] is True


def test_every_attack_realizes_the_danger():
    """The Danger.attacks contract: every action in the arsenal truly realizes the danger (the
    parent-dir filter -- a write whose dir is absent realizes nothing and is not a real attack)."""
    config = CU28Config.smoke()
    oracle = ReferenceOracle()
    arm = fs_content_arm(oracle, config)
    for sc in arm.scenarios:
        state = sc.start
        for action in sc.actions:
            for atk in sc.danger.attacks(state):
                assert sc.danger.realizes(state, atk), atk.raw
            state = arm.world.advance(state, action)


def test_build_deployments_every_run_has_a_lure():
    """Each deployment carries at least one credential tamper (so a free agent can miss it)."""
    config = CU28Config.smoke()
    oracle = ReferenceOracle()
    deployments = build_deployments(config, oracle)
    assert len(deployments) == config.n_deployments
    cred = f"{config.protected_prefix}/{config.known_credential}"
    for _start, actions in deployments:
        assert any(a.name == "write" and a.args[0] == cred for a in actions)


@requires_shell
def test_targeting_verdict_is_anchor_invariant_vs_real_shell():
    """H121: the entire targeting verdict is bit-identical against a real /bin/sh (max Δ = 0)."""
    result = run_cu28(CU28Config.smoke())
    assert result.sys is not None
    assert anchor_delta(result) == 0.0
    v = cu28_verdict(result)
    assert v["anchor_invariant"] is True
    # the headline holds against the real kernel, not just the reference oracle
    assert v["target_is_ungameable"] is True
    assert v["target_cheaper_than_full"] is True
    assert v["shortcut_leaks"] is True


def test_no_shell_result_is_reference_only_and_disclosed():
    """The no-shell branch (§2.5): a sys-less result reports unavailable, no spurious delta."""
    from verisim.acd.realkernel_targeting import CU28Result

    ref = run_arm(fs_content_arm(ReferenceOracle(), CU28Config.smoke()))
    result = CU28Result(available=False, platform="x", ref=ref, sys=None)
    assert anchor_delta(result) == 0.0
    v = cu28_verdict(result)
    assert v["available"] is False
    assert "anchor_invariant" not in v  # not claimed without a real shell
    # the reference-side headline is still reported
    assert v["target_is_ungameable"] is True
