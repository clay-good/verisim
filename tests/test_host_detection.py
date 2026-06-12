"""SPEC-20 UA12 / H92 -- the operational detection characteristic.

The contract: the confusion matrix is exact, a faithful predictor scores precision = recall = 1.0, a
drifted predictor loses *both* (the false-alarm cost UA8's recall-only metric could not see), the
ρ-grounded extremes recover faithful/free, and -- with the trained host `M_θ` -- the free detector's
F1 collapses below its recall and grounding restores it at a sub-linear knee. Torch-free core.
"""

from __future__ import annotations

import pytest

from verisim.acd.host_detection import (
    confusion,
    detection_scores,
    grounded_detection,
    mean_detection,
)
from verisim.acd.host_integrity import make_workload, oracle_step
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle


class _BlindModel:
    """Predicts no deltas -- never writes anything (the degenerate free predictor)."""

    def predict_delta(self, state: HostState, action: HostAction) -> list[object]:
        return []


def test_confusion_matrix_is_exact():
    d = confusion({"a", "b", "c"}, {"b", "c", "d"})
    assert (d.tp, d.fp, d.fn, d.n_flagged) == (2, 1, 1, 3)
    assert d.precision == pytest.approx(2 / 3)
    assert d.recall == pytest.approx(2 / 3)
    assert d.f1 == pytest.approx(2 / 3)


def test_empty_predictions_and_truth_are_well_defined():
    # nothing flagged, nothing corrupted -> a vacuously perfect detector (clean workload, no alarms)
    d = confusion(set(), set())
    assert d.precision == 1.0 and d.recall == 1.0 and d.f1 == 1.0
    # flagged files but nothing was truly corrupted -> all false alarms (precision is vacuous-1 only
    # when nothing is flagged; here tp=0,fp=2 -> precision 0)
    d2 = confusion({"x", "y"}, set())
    assert d2.precision == 0.0 and d2.fp == 2 and d2.recall == 1.0
    # missed everything
    d3 = confusion(set(), {"x"})
    assert d3.recall == 0.0 and d3.fn == 1


def test_faithful_detector_is_perfect_on_every_workload():
    oracle = ReferenceHostOracle()
    step = oracle_step(oracle)
    for seed in (700, 701, 702):
        start, actions = make_workload(seed, 16, oracle=oracle)
        d = detection_scores(step, step, start, actions)
        assert d.precision == 1.0 and d.recall == 1.0
        assert d.fp == 0 and d.fn == 0


def test_blind_detector_has_zero_recall_no_false_alarms():
    # the blind model flags nothing -> no false positives, but misses every real corruption
    oracle = ReferenceHostOracle()
    start, actions = make_workload(701, 16, oracle=oracle)
    true_step = oracle_step(oracle)
    from verisim.acd.host_integrity import model_step

    d = detection_scores(model_step(_BlindModel()), true_step, start, actions)
    assert d.n_flagged == 0
    assert d.recall == 0.0  # there are writes in this workload, all missed
    assert d.precision == 1.0  # flagged nothing -> no false alarm (vacuous)


def test_grounded_extremes_recover_faithful_and_free():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(701, 16, oracle=oracle)
    blind = _BlindModel()
    d1, calls1 = grounded_detection(blind, oracle, start, actions, 1.0)
    assert d1.precision == 1.0 and d1.recall == 1.0 and calls1 == len(actions)  # ρ=1 ≡ faithful
    _, calls0 = grounded_detection(blind, oracle, start, actions, 0.0)
    assert calls0 == 0  # ρ=0 ≡ free (no oracle calls)


def test_mean_detection_well_formed():
    m = mean_detection([confusion({"a"}, {"a"}), confusion({"b"}, {"c"})])
    assert 0.0 <= m["precision"] <= 1.0 and 0.0 <= m["recall"] <= 1.0
    assert m["f1"] == pytest.approx(0.5)  # one perfect (1.0), one total miss (0.0)


# --- torch-gated: the trained-M_θ operational story -----------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.host_flagship import (  # noqa: E402
    HostFlagshipConfig,
    train_host_flagship,
)
from verisim.experiments.ua_host_detection import (  # noqa: E402
    UA12Config,
    knee_rho,
    run_ua12,
)


def test_free_detector_loses_precision_and_recall_and_knee_restores():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    result = run_ua12(model, UA12Config.smoke())
    # the faithful detector is perfect at every horizon
    for r in result.horizon_sweep:
        assert r.faithful["precision"] == pytest.approx(1.0)
        assert r.faithful["recall"] == pytest.approx(1.0)
        assert r.faithful["f1"] == pytest.approx(1.0)
    # the free detector loses BOTH precision and recall on the longest horizon (the H92 point):
    # UA8 measured only the recall collapse; the precision drop is the false-alarm cost it could not
    # see, and F1 (deployability) is degraded below the faithful 1.0
    longest = result.horizon_sweep[-1]
    assert longest.free["precision"] < 1.0
    assert longest.free["recall"] < 1.0
    assert longest.free["f1"] < 1.0
    # grounding climbs F1 from the free floor (ρ=0) to the faithful ceiling (ρ=1)
    by_rho = {k.rho: k.grounded["f1"] for k in result.knee}
    assert by_rho[1.0] >= by_rho[0.0]
    assert by_rho[1.0] == pytest.approx(1.0)
    # the knee is well-formed (a ρ in the grid)
    assert knee_rho(result) in {k.rho for k in result.knee}
