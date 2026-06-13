"""SPEC-22 CU8 / H101 -- the drift asymmetry (torch-free core).

The probe classifies per-step flow-prediction errors as omissions or hallucinations against the
oracle. Validated with stand-ins: a perfect (oracle) model makes neither error; a blind no-op model
makes pure omissions (it misses every real flow) and zero hallucinations -- the maximal omit bias
the trained arm is hypothesized to approach. The real trained M_θ is the experiment's committed run.
"""

from __future__ import annotations

from verisim.acd.drift_asymmetry import (
    CU8Config,
    cu8_verdict,
    run_drift_asymmetry,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: object, action: object) -> object:
        return self._oracle.step(state, action).delta  # type: ignore[arg-type]


class _BlindModel:
    def predict_delta(self, state: object, action: object) -> list[object]:
        return []


def test_perfect_model_has_no_omission_or_hallucination():
    result = run_drift_asymmetry(_OracleModel(ReferenceNetworkOracle()), CU8Config.smoke())
    assert sum(result.omitted.values()) == 0
    assert sum(result.hallucinated.values()) == 0
    assert result.recall("protected") == 1.0


def test_blind_model_is_pure_omission():
    # the blind model misses every real flow and invents none -- maximal omission bias
    result = run_drift_asymmetry(_BlindModel(), CU8Config())
    assert result.omitted["protected"] == result.true_opens["protected"] > 0
    assert result.hallucinated["protected"] == 0
    assert result.recall("protected") == 0.0
    verdict = cu8_verdict(result)
    assert verdict["drift_is_omission_biased"] is True
    assert verdict["danger_hidden_by_omission"] is True


def test_real_flows_exist_in_both_classes():
    # the battery exercises both danger and benign flow opens (non-degenerate denominators)
    result = run_drift_asymmetry(_BlindModel(), CU8Config())
    assert result.true_opens["protected"] > 0
    assert result.true_opens["work"] > 0
