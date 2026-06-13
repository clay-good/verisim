"""SPEC-22 CU12 / H105 -- knowledge-free targeting (torch-free core).

Validated with cheap stand-in models: a *blind* no-op model (the worst omitter) makes the
asset-indexed target degrade to the unverified breach as the inventory shrinks below the true
sensitive set, while the grammar-indexed target (verify every connect) reaches zero breach
inventory-independently and stays cheaper than the full oracle; a *perfect* oracle model needs no
verification at all. The real trained transformer arm is the committed run (torch-gated,
not exercised in CI).
"""

from __future__ import annotations

from verisim.acd.knowledge_free_targeting import (
    CU12Config,
    adversarial_breach_kf,
    cu12_verdict,
    run_cu12,
)
from verisim.acd.safety_horizon import Deployment
from verisim.acd.targeted_verification import build_deployments
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


_CONFIG = CU12Config(horizon=48, n_seeds=400, max_episodes=120, rhos=(0.5, 1.0))


def test_grammar_target_is_safe_inventory_independent():
    # Verifying every connect catches every exfil regardless of which hosts are flagged -> 0 breach
    # both on the random workload and against an adversary who picks the target host.
    result = run_cu12(_BlindModel(), _CONFIG)
    assert result.grammar.breach_random == 0.0
    assert result.grammar.breach_adversarial == 0.0
    # ... and cheaper than the full oracle (uniform ρ=1)
    full = next(c for c in result.uniform if c.label == "uniform ρ=1")
    assert full.breach_random == 0.0
    assert result.grammar.mean_calls < full.mean_calls


def test_asset_target_degrades_with_incomplete_inventory():
    # Complete inventory (K == T) is safe; dropping a true crown jewel lets exfil to it through,
    # so the breach rises toward the unverified rate -- a false sense of security.
    result = run_cu12(_BlindModel(), _CONFIG)
    complete = result.asset[-1]
    incomplete = result.asset[1]  # K = {h0}, h4 unflagged
    empty = result.asset[0]  # K = {} -> never consults -> the free agent
    assert complete.breach_random == 0.0
    assert incomplete.breach_random > complete.breach_random + 0.1
    assert empty.breach_random >= incomplete.breach_random  # the fewer flagged, the worse


def test_asset_target_is_gameable_grammar_is_not():
    # Adversarially, an incomplete asset target is fully gameable (the attacker exfils to the
    # unflagged host, never consulted), while the grammar target holds at zero.
    result = run_cu12(_BlindModel(), _CONFIG)
    incomplete = result.asset[1]  # K = {h0}
    assert incomplete.breach_adversarial >= 0.9
    assert result.grammar.breach_adversarial == 0.0


def test_perfect_model_needs_no_verification():
    oracle = ReferenceNetworkOracle()
    result = run_cu12(_OracleModel(oracle), CU12Config.smoke())
    for c in (*result.uniform, *result.asset, result.grammar):
        assert c.breach_random == 0.0  # a faithful preview foresees every exfil and the gate blocks


def test_adversarial_breach_kf_returns_bool():
    oracle = ReferenceNetworkOracle()
    T = frozenset(("h0", "h4"))
    deployments = build_deployments(CU12Config.smoke().base(), oracle)
    out = adversarial_breach_kf(oracle, deployments[0], T, "grammar", frozenset())
    assert out is False
    assert isinstance(
        adversarial_breach_kf(oracle, deployments[0], T, "asset", frozenset(("h0",))), bool
    )


def test_verdict_reports_the_headline():
    result = run_cu12(_BlindModel(), _CONFIG)
    verdict = cu12_verdict(result)
    assert verdict["asset_target_degrades"] is True
    assert verdict["asset_target_gameable"] is True
    assert verdict["grammar_is_safe"] is True
    assert verdict["grammar_cheaper_than_full"] is True


def test_deployments_keyed_on_true_sensitive_set():
    oracle = ReferenceNetworkOracle()
    deployments = build_deployments(CU12Config.smoke().base(), oracle)
    assert deployments
    assert all(isinstance(d, Deployment) for d in deployments)
