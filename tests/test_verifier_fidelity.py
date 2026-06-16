"""Tests for SPEC-22 CU35 (H128): the verifier-fidelity condition -- the dual coverage law.

The arc proved the *target* must cover the danger surface; CU35 proves the matching dual on the
*verifier*: it must be faithful ON the danger surface. On-surface fidelity is load-bearing (sloped
vs nature, a cliff vs the adversary); off-surface fidelity is irrelevant to safety (a verifier
globally wrong but exact on the danger grammar is as safe as a perfect oracle, buying only false
blocks). Both coverage conditions are independently necessary. Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.acd.unified_targeting import Danger, Scenario, World, net_flow_arm
from verisim.acd.verifier_fidelity import (
    CU35Config,
    ExactVerifier,
    OffSurfaceDrifter,
    SurfaceOmitter,
    adversarial_with_verifier,
    arm_verdict,
    cu35_verdict,
    faithful_on_surface,
    make_offsurface_drifter,
    run_cu35,
    run_with_verifier,
    sweep_arm,
)


@dataclass(frozen=True)
class _A:
    name: str


def _realizes(s: object, a: object) -> bool:
    return isinstance(a, _A) and a.name.startswith("danger")


def _danger() -> Danger:
    return Danger(realizes=_realizes, attacks=lambda s: [_A("danger0"), _A("danger1")])


def _covering_target(s: object, a: object) -> bool:
    # fires on every danger action (and some benign "near" ones) -> covers the surface
    return isinstance(a, _A) and (a.name.startswith("danger") or a.name.startswith("near"))


def _shortcut(s: object, a: object) -> bool:
    # fires only on the benign "near" actions -> does NOT cover the danger
    return isinstance(a, _A) and a.name.startswith("near")


def _scenario(actions: tuple[_A, ...]) -> Scenario:
    return Scenario(start=frozenset(), actions=actions, danger=_danger(),
                    target=_covering_target, shortcut=_shortcut)


_WORLD = World(advance=lambda s, a: s)


# --- the verifier verdict mechanics (the new variable) --------------------------------------------


def test_exact_verifier_equals_oracle():
    v = ExactVerifier(_realizes)
    assert v.verdict(None, _A("danger0")) is True
    assert v.verdict(None, _A("benign")) is False


def test_surface_omitter_blind_at_phi0_exact_at_phi1():
    blind = SurfaceOmitter(_realizes, 0.0)
    exact = SurfaceOmitter(_realizes, 1.0)
    # blind never flags a danger; exact always does
    assert blind.verdict(None, _A("danger0")) is False
    assert exact.verdict(None, _A("danger0")) is True
    # both agree OFF the surface (no danger -> no flag): the gap lands only on the surface
    assert blind.verdict(None, _A("benign")) is False
    assert exact.verdict(None, _A("benign")) is False


def test_offsurface_drifter_exact_on_surface_hallucinates_off():
    drift = OffSurfaceDrifter(_realizes, 0.0)  # psi=0 -> hallucinate everywhere off-surface
    clean = OffSurfaceDrifter(_realizes, 1.0)  # psi=1 -> never hallucinate
    # both are EXACT on the danger surface
    assert drift.verdict(None, _A("danger0")) is True
    assert clean.verdict(None, _A("danger0")) is True
    # off-surface: psi=0 hallucinates danger, psi=1 does not
    assert drift.verdict(None, _A("benign")) is True
    assert clean.verdict(None, _A("benign")) is False


def test_faithful_on_surface_is_the_dual_of_covers():
    sc = _scenario((_A("benign"),))
    assert faithful_on_surface(_WORLD, sc, ExactVerifier(_realizes)) is True
    assert faithful_on_surface(_WORLD, sc, SurfaceOmitter(_realizes, 0.0)) is False


# --- the gate outcome: a covering target is only as safe as its verifier on the surface -----------


def test_covering_target_safe_with_exact_verifier_leaks_with_blind():
    sc = _scenario((_A("danger0"),))
    breached_exact, calls, _ = run_with_verifier(_WORLD, sc, ExactVerifier(_realizes),
                                                 _covering_target)
    breached_blind, _, _ = run_with_verifier(_WORLD, sc, SurfaceOmitter(_realizes, 0.0),
                                             _covering_target)
    assert calls == 1  # the danger action is on the covering surface -> consulted
    assert breached_exact is False  # exact verifier blocks the danger
    assert breached_blind is True  # blind-on-surface verifier waves it through


def test_offsurface_drift_costs_false_blocks_not_safety():
    # a benign "near" action is on the covering target but off the danger surface
    sc = _scenario((_A("near0"),))
    _, _, fb_drift = run_with_verifier(_WORLD, sc, OffSurfaceDrifter(_realizes, 0.0),
                                       _covering_target)
    _, _, fb_clean = run_with_verifier(_WORLD, sc, OffSurfaceDrifter(_realizes, 1.0),
                                       _covering_target)
    assert fb_drift == 1  # the drifter false-blocks the benign on-target action
    assert fb_clean == 0  # the exact-off-surface verifier does not


def test_offsurface_drift_never_breaches_adversarially():
    # the localization theorem on a synthetic arm: exact-on-surface verifier is un-gameable for any
    # off-surface drift (every attack is on-surface and flagged)
    sc = _scenario((_A("near0"), _A("benign")))
    assert adversarial_with_verifier(_WORLD, sc, OffSurfaceDrifter(_realizes, 0.0),
                                     _covering_target) is False


# --- the localization on a REAL arm (net exfil) ---------------------------------------------------


def test_offsurface_drifter_safe_on_real_net_arm():
    arm = net_flow_arm()
    sweep = sweep_arm(arm, fidelities=(0.0, 1.0), max_scenarios=20)
    # off-surface fidelity (psi) is irrelevant to safety: adversarial breach flat at 0 across psi
    assert all(p.adversarial_breach <= 1e-9 for p in sweep.off_surface)
    # while on-surface: blind (phi=0) leaks, exact (phi=1) is safe
    assert sweep.on_surface[0].adversarial_breach >= 0.5  # phi=0
    assert sweep.on_surface[-1].adversarial_breach <= 1e-9  # phi=1


# --- the integration verdict (smoke) --------------------------------------------------------------


def test_cu35_verdict_smoke():
    result = run_cu35(CU35Config.smoke())
    v = cu35_verdict(result)
    assert v["exact_verifier_safe_everywhere"] is True
    assert v["surface_blind_leaks_everywhere"] is True
    assert v["offsurface_fidelity_irrelevant_to_safety"] is True
    # the 2x2: only (covers AND faithful) is safe; the other three corners leak
    assert v["both_conditions_necessary"] is True
    grid = result.grid
    assert grid.covers_exact <= 1e-9
    assert grid.covers_blind >= 0.5 and grid.leak_exact >= 0.5 and grid.leak_blind >= 0.5


def test_arm_verdict_reports_horizon_and_cliff():
    result = run_cu35(CU35Config(fidelities=(0.0, 0.5, 1.0), max_scenarios=40))
    a = arm_verdict(result.arms[0])  # network
    assert a["exact_verifier_safe"] is True
    assert a["surface_blind_verifier_leaks"] is True
    assert a["adversarial_is_a_cliff"] is True


def test_make_offsurface_drifter_factory_uses_scenario_realizes():
    make = make_offsurface_drifter(0.0)
    v = make(_realizes)
    assert isinstance(v, OffSurfaceDrifter)
    assert v.verdict(None, _A("danger0")) is True
