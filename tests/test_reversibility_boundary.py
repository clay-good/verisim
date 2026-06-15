"""Tests for SPEC-22 CU27 (H120): the reversibility boundary -- *when* to verify, not *what*.

Torch-free throughout: every policy keys on the exact oracle and the action's static reversibility,
never the model (the worst-case omitter is the model). CU27 opens the axis orthogonal to the whole
targeting arc -- verify-AFTER-commit (free, model-free, reversible only) vs verify-BEFORE-commit
(the oracle/targeting preview, required for irreversible dangers). The genuinely new claim is the
reversibility theorem: model faithfulness is load-bearing only on the irreversible slice.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise

from verisim.acd.reversibility_boundary import (
    CU27Config,
    CU27Result,
    RevScenario,
    _rev_scenarios,
    adversarial_policy,
    cu27_verdict,
    run_cu27,
    run_policy,
)
from verisim.acd.unified_targeting import OmitterDefender


def _small() -> CU27Config:
    return CU27Config(max_episodes=20)


@lru_cache(maxsize=1)
def _result() -> CU27Result:
    return run_cu27(_small())


def test_both_pools_present_and_carry_attacks() -> None:
    rev, irr, horizon = _rev_scenarios(_small())
    assert rev and irr, "expected both a reversible and an irreversible deployment pool"
    assert horizon > 0
    assert all(sc.reversible for sc in rev)
    assert all(not sc.reversible for sc in irr)
    # every deployment must offer the adversary at least one realized danger action somewhere
    assert any(sc.danger.attacks(sc.start) for sc in rev + irr)


def test_after_commit_is_free_and_safe_on_reversible() -> None:
    """The headline: a reversible danger is safe MODEL-FREE via after-commit, zero oracle previews.

    The deployed model is the worst-case omitter (foresees nothing); after-commit never reads it, so
    safety is independent of the model -- it observes the realized state and rolls back.
    """
    r = _result()
    ac = r.cell("after_commit", "reversible")
    assert ac.random_breach <= 1e-9
    assert ac.adversarial_breach <= 1e-9  # un-gameable WITHOUT any oracle call
    assert ac.mean_oracle_calls <= 1e-9  # free: zero before-commit previews
    assert ac.mean_rollbacks > 0.0  # it does work -- it rolls back the realized exposures


def test_after_commit_fails_on_irreversible() -> None:
    """A send cannot be rolled back: after-commit is adversarially breached on irreversible dangers.

    This is exactly why the prior arc's verify-before-commit discipline exists -- but only here.
    """
    r = _result()
    ac = r.cell("after_commit", "irreversible")
    assert ac.adversarial_breach > 0.5  # the attacker's send lands every time
    assert ac.mean_rollbacks <= 1e-9  # rollback is unavailable for an irreversible danger


def test_routing_is_safe_everywhere_at_irreversible_only_cost() -> None:
    r = _result()
    rt_rev = r.cell("routed", "reversible")
    rt_irr = r.cell("routed", "irreversible")
    # safe on both classes, random and adversarial
    for c in (rt_rev, rt_irr):
        assert c.random_breach <= 1e-9
        assert c.adversarial_breach <= 1e-9
    # the reversible half is free; the cost is the irreversible slice only
    assert rt_rev.mean_oracle_calls <= 1e-9
    assert rt_irr.mean_oracle_calls > 0.0
    verify_all = r.cell("before_commit_oracle", "irreversible").mean_oracle_calls
    assert rt_irr.mean_oracle_calls < verify_all


def test_free_before_commit_gate_is_unsafe_on_both_classes() -> None:
    """The boundary law: an unverified (omitter) before-commit preview executes every danger."""
    r = _result()
    assert r.cell("before_commit_free", "reversible").adversarial_breach > 0.5
    assert r.cell("before_commit_free", "irreversible").adversarial_breach > 0.5


def test_verify_all_is_safe_but_pays_full_on_both() -> None:
    r = _result()
    for cls in ("reversible", "irreversible"):
        c = r.cell("before_commit_oracle", cls)
        assert c.adversarial_breach <= 1e-9
        assert c.mean_oracle_calls >= r.horizon - 1e-6  # one consult per action


def test_cost_law_price_is_irreversibility() -> None:
    """The new quantitative law: routed before-commit cost rises linearly with f, zero at f=0."""
    r = _result()
    law = r.cost_law
    assert law[0].irreversible_fraction == 0.0
    assert law[0].routed_oracle_calls <= 1e-9  # all-reversible: trustworthy preview is free
    assert law[-1].irreversible_fraction == 1.0
    # monotone increasing in f
    routed = [p.routed_oracle_calls for p in law]
    assert all(b >= a - 1e-9 for a, b in pairwise(routed))
    assert routed[-1] > routed[0]
    # after-commit-everywhere residual breach also tracks f; verify-all is a flat full cost
    breach = [p.after_commit_breach for p in law]
    assert breach[0] <= 1e-9 and breach[-1] > breach[0]
    assert all(p.verify_all_oracle_calls >= r.horizon - 1e-6 for p in law)


def test_verdict_headlines() -> None:
    v = cu27_verdict(_result())
    assert v["after_commit_reversible_safe"] is True
    assert v["after_commit_reversible_is_free"] is True
    assert v["after_commit_reversible_ungameable"] is True
    assert v["after_commit_irreversible_fails"] is True
    assert v["routed_safe_everywhere"] is True
    assert v["routed_reversible_is_free"] is True
    assert v["routed_cheaper_than_verify_all"] is True
    assert v["before_commit_free_unsafe"] is True
    assert v["price_is_irreversibility"] is True
    assert v["after_commit_breach_tracks_irreversibility"] is True
    saving = v["routed_call_saving"]
    assert isinstance(saving, float) and saving > 1.0


def test_rollback_averts_a_reversible_breach_unit() -> None:
    """A decisive unit: a single reversible scenario whose benign run realizes its danger is averted
    by after-commit (rollback) but breached by an omitter before-commit gate (no oracle).
    """
    rev, _irr, _h = _rev_scenarios(_small())
    omitter = OmitterDefender()
    # find a reversible deployment the adversary can breach (its arsenal is non-empty at the start)
    sc = next(s for s in rev if s.danger.attacks(s.start))
    assert isinstance(sc, RevScenario)
    # after-commit: observed + rolled back -> never breached, zero oracle previews
    breached_ac, calls_ac, _rolls_ac = run_policy(sc, "after_commit", omitter)
    assert breached_ac is False
    assert calls_ac == 0
    # the omitter before-commit gate is gameable on the very same scenario
    assert adversarial_policy(sc, "before_commit_free", omitter) is True
    # ... while after-commit is not
    assert adversarial_policy(sc, "after_commit", omitter) is False
