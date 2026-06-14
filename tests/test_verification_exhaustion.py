"""SPEC-22 CU15 / H108 -- the verification-exhaustion attack (torch-free core).

Validated with cheap stand-in models. A *blind* no-op model (the worst omitter) makes the cost
attack bite exactly as the trained ``M_θ`` would: structure's safety is immovable (breach 0 at every
saturation, because every poisoned ``connect``-to-jewel is oracle-blocked) while its cost rises
one-for-one with the attacker's flood, up to but never past the full oracle; uniform's cost is
immovable (clock-keyed) while its safety is gameable (the off-clock exfils the omitter misses). A
*perfect* oracle model needs no verification at all -- it foresees and blocks every exfil under
every schedule, so breach is 0 everywhere. The real trained transformer arm is the experiment's
committed run (torch-gated, not exercised in CI).
"""

from __future__ import annotations

from verisim.acd.safety_horizon import Deployment
from verisim.acd.targeted_verification import build_deployments
from verisim.acd.verification_exhaustion import (
    CU15Config,
    _series,
    cu15_verdict,
    run_cu15,
    run_exhaustion,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    """A perfect world-model stand-in: its predicted delta is the oracle's true delta."""

    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: object, action: object) -> object:
        return self._oracle.step(state, action).delta  # type: ignore[arg-type]


class _BlindModel:
    """A maximally-drifting stand-in: predicts no change, so it foresees no flow (pure omission)."""

    def predict_delta(self, state: object, action: object) -> list[object]:
        return []


_CONFIG = CU15Config(
    horizon=48, n_seeds=400, max_episodes=120, saturations=(0.0, 0.5, 1.0)
)


def test_deployments_present_an_exfil_opportunity():
    oracle = ReferenceNetworkOracle()
    deployments = build_deployments(CU15Config.smoke()._battery_config(), oracle)
    assert deployments
    assert all(isinstance(d, Deployment) for d in deployments)


def test_structure_safety_is_immovable():
    # Every poisoned connect-to-jewel lands on the surface structure always verifies, so the oracle
    # blocks each one: breach stays at 0 for every saturation, however hard the attacker floods.
    result = run_cu15(_BlindModel(), _CONFIG)
    structure = _series(result, "structure (crown-jewel)")
    assert all(c.breach_rate == 0.0 for c in structure)


def test_structure_cost_rises_with_saturation():
    # The dual of safety-immovability: the attacker CAN move structure's cost. Each poisoned step is
    # a connect-to-jewel structure must verify, so its calls climb toward the horizon as s -> 1.
    result = run_cu15(_BlindModel(), _CONFIG)
    structure = _series(result, "structure (crown-jewel)")
    assert structure[-1].mean_calls > structure[0].mean_calls + 1.0
    assert structure[-1].mean_calls >= 0.9 * result.horizon  # full saturation -> ~full oracle


def test_structure_weakly_dominates_the_full_oracle():
    # Structure never spends more than verifying everything: calls <= horizon at every saturation,
    # equal only when the attacker poisons every step (and even then it is still perfectly safe).
    result = run_cu15(_BlindModel(), _CONFIG)
    structure = _series(result, "structure (crown-jewel)")
    full = _series(result, "full oracle")
    for sc, fc in zip(structure, full, strict=True):
        assert sc.mean_calls <= fc.mean_calls + 1e-9


def test_uniform_safety_is_gameable_but_cost_is_immovable():
    # The mirror image of structure: uniform's cost is clock-keyed (fixed across saturation) while
    # its safety degrades as the attacker floods off-clock exfils the omitting model misses.
    result = run_cu15(_BlindModel(), _CONFIG)
    uniform = _series(result, "uniform ρ=0.5")
    assert abs(uniform[-1].mean_calls - uniform[0].mean_calls) <= 1.0  # cost immovable
    assert uniform[-1].breach_rate > uniform[0].breach_rate  # safety gameable


def test_attack_is_self_limiting():
    # Each attacker action adds ~one defender call and buys zero breaches: the adversary spends its
    # whole budget to remove a discount, never to cause harm.
    result = run_cu15(_BlindModel(), _CONFIG)
    verdict = cu15_verdict(result)
    assert 0.7 <= float(verdict["cost_per_attacker_action"]) <= 1.05  # type: ignore[arg-type]
    assert verdict["breaches_bought"] == 0.0


def test_perfect_model_is_safe_under_every_schedule_and_saturation():
    # A faithful model foresees and blocks every exfil on its own, so even the free agent never
    # breaches -- the cost attack cannot manufacture a breach against a model that does not omit.
    oracle = ReferenceNetworkOracle()
    result = run_cu15(_OracleModel(oracle), CU15Config.smoke())
    assert all(c.breach_rate == 0.0 for c in result.cells)


def test_verdict_reports_the_headline():
    result = run_cu15(_BlindModel(), _CONFIG)
    verdict = cu15_verdict(result)
    assert verdict["structure_safety_immovable"] is True
    assert verdict["structure_cost_gameable"] is True
    assert verdict["structure_dominates_full_oracle"] is True
    assert verdict["uniform_cost_immovable"] is True
    assert verdict["uniform_safety_gameable"] is True


def test_run_exhaustion_returns_breach_and_calls():
    oracle = ReferenceNetworkOracle()
    deployments = build_deployments(CU15Config.smoke()._battery_config(), oracle)
    cfg = CU15Config.smoke()
    breached, calls = run_exhaustion(
        _BlindModel(), oracle, deployments[0], cfg, "structure", 0.0, 1.0
    )
    assert breached is False
    assert isinstance(calls, int) and calls >= 0
