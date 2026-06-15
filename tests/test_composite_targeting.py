"""Tests for SPEC-22 CU24 (H117): the composite defense -- the whole threat model at once.

Torch-free throughout (the worst-case-omitter substrate of the targeting arc). CU24 builds three
coexisting network dangers (exfil / exposure / outage) on CU22's provisioned-work battery and runs
every point / partial / union schedule against the composite threat model. The composition theorem
(the union of covering targets covers the union danger) predicts the union is safe + un-gameable at
the union of the surfaces; the boundary predicts every partial leaks exactly its omitted leg; and
``covers`` calls all of it a priori.
"""

from __future__ import annotations

from functools import lru_cache

from verisim.acd.availability_targeting import CU22Config
from verisim.acd.composite_targeting import (
    LEG_NAMES,
    CU24Config,
    CU24Result,
    _candidate,
    cu24_verdict,
    run_cu24,
    union_danger,
    union_target,
)


def _small() -> CU24Config:
    return CU24Config(
        battery=CU22Config(horizon=16, n_seeds=240, max_episodes=12, rhos=(0.0, 0.5, 1.0)),
        rhos=(0.0, 0.5, 1.0),
    )


@lru_cache(maxsize=1)
def _result() -> CU24Result:
    return run_cu24(_small())


def test_three_legs_coexist_on_the_battery() -> None:
    r = _result()
    assert r.n_episodes > 0
    assert LEG_NAMES == ("exfil", "exposure", "outage")
    # every leg's surface is non-empty across the battery (each danger is genuinely present)
    for leg in LEG_NAMES:
        assert r.single_leg_calls[leg] > 0.0, f"the {leg} surface never fired"


def test_union_target_is_safe_cheap_and_ungameable_on_every_leg() -> None:
    """THE HEADLINE: one union target defends the whole threat model, safe + un-gameable + cheap."""
    comp = _candidate(_result(), "composite")
    full = _result().full_oracle
    assert comp.covers_composite is True  # coverage holds (union of covering targets covers union)
    assert comp.composite_random_breach <= full.composite_random_breach + 1e-9  # oracle safety
    assert comp.composite_adversarial_breach <= 1e-9  # un-gameable on the composite
    assert all(b <= 1e-9 for b in comp.per_leg_adversarial.values())  # safe on EVERY leg
    assert comp.mean_calls < full.mean_calls  # cheaper than verifying everything


def test_union_surface_is_the_sum_of_the_disjoint_per_leg_surfaces() -> None:
    """Defense in depth = the union of the rare per-leg surfaces (subadditive; here disjoint)."""
    r = _result()
    comp = _candidate(r, "composite")
    assert comp.mean_calls <= sum(r.single_leg_calls.values()) + 1e-9  # subadditive


def test_every_partial_leaks_exactly_its_omitted_leg() -> None:
    """THE BOUNDARY: a partial union covers its own legs and is gameable on the omitted ones."""
    r = _result()
    partials = [c for c in r.candidates if c.name != "composite"]
    assert partials
    for c in partials:
        covered = set(c.legs)
        assert c.covers_composite is False  # a proper subset cannot cover the union danger
        for leg, breach in c.per_leg_adversarial.items():
            if leg in covered:
                assert breach <= 1e-9, f"{c.name} should be safe on its covered leg {leg}"
            else:
                assert breach > 1e-9, f"{c.name} should leak its omitted leg {leg}"


def test_most_quoted_point_defense_is_gameable_on_the_composite() -> None:
    """The CU10 exfil target is un-gameable on its OWN leg but wide open on the threat model."""
    exfil = _candidate(_result(), "exfil_only")
    assert exfil.per_leg_adversarial["exfil"] <= 1e-9  # un-gameable on confidentiality
    assert exfil.composite_adversarial_breach > 1e-9  # gameable on the composite


def test_covers_predicts_every_candidate() -> None:
    """The generative claim on composition: covers() <=> adversarially safe, for every row."""
    v = cu24_verdict(_result())
    assert v["covers_predicts_every_candidate"] is True
    assert v["composite_safe_on_every_leg"] is True
    assert v["all_partials_break_coverage"] is True
    assert v["partial_leaks_exactly_omitted_leg"] is True


def test_union_danger_and_target_are_disjunctions() -> None:
    """The composition primitives: realizes = OR of legs; target fires iff any chosen fires."""
    r = _result()
    # rebuild one deployment's legs to exercise union_danger / union_target directly
    from verisim.acd.availability_targeting import build_deployments
    from verisim.acd.composite_targeting import _legs
    from verisim.netoracle.reference import ReferenceNetworkOracle

    battery = _small().battery
    oracle = ReferenceNetworkOracle()
    deps = build_deployments(battery, oracle)
    legs = _legs(battery, oracle, deps[0])
    d_all = union_danger(legs, LEG_NAMES)
    t_all = union_target(legs, LEG_NAMES)
    s = deps[0].start
    # the union arsenal is the concatenation of the per-leg arsenals
    expected = sum(len(legs[n].danger.attacks(s)) for n in LEG_NAMES)
    assert len(d_all.attacks(s)) == expected
    # every attack in the union arsenal is on the union target's surface (coverage, by construction)
    for a in d_all.attacks(s):
        assert t_all(s, a)
    assert r.n_episodes > 0


def test_baselines_behave() -> None:
    v = cu24_verdict(_result())
    assert v["uniform_is_gameable"] is True
    assert v["model_self_targeting_fails"] is True
    assert v["oracle_self_governs"] is True
    saving = v["composite_call_saving"]
    assert isinstance(saving, float) and saving > 1.0
