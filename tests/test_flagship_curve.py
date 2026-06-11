"""FL1 flagship-curve tests (SPEC-19 §3, milestone FL1).

The contract under test is the four-arm `H_ε(ρ)` sweep on a (smoke) flagship checkpoint and the
composition seam:

  - ``ComposedConsult`` triggers on EITHER the conformal threshold OR the speculative window
    boundary, and is a valid ``ConsultationPolicy``;
  - ``build_calibration`` returns aligned (score, breach) lists from the real model's free run;
  - ``run_flagship_curve`` yields well-formed floor/ceiling/fixed/composed points with the ordering
    invariant floor ≤ {fixed, composed} ≤ ceiling (more budget cannot reduce faithful horizon);
  - ``headline_verdict`` reports the H69 decision fields.

The smoke model is trivial; the committed curve comes from the local run on the real `l@9.6k`
checkpoint (the SPEC-9 envelope discipline). What CI guarantees is that the apparatus is correct and
deterministic, not what the curve's verdict is.
"""

from __future__ import annotations

import pytest

from verisim.experiments.flagship_curve import ComposedConsult, FlagshipCurveConfig
from verisim.loop.policy import ConsultationPolicy, StepContext


def test_composed_consult_is_a_policy_and_or_composes():
    pol = ComposedConsult(tau=0.5, window=4)
    assert isinstance(pol, ConsultationPolicy)
    # conformal clause: signal over threshold triggers even mid-window
    assert pol.should_consult(StepContext(step=0, signal=0.9))
    # window clause: step 3 (0-indexed) is the 4th step -> window boundary, triggers at low signal
    assert pol.should_consult(StepContext(step=3, signal=0.0))
    # neither clause: low signal, not a boundary -> no consult
    assert not pol.should_consult(StepContext(step=1, signal=0.0))


def test_composed_consult_rejects_bad_window():
    with pytest.raises(ValueError):
        ComposedConsult(tau=0.5, window=0)


# --- torch-gated: the real-model sweep ------------------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.flagship import FlagshipConfig, train_flagship  # noqa: E402
from verisim.experiments.flagship_curve import (  # noqa: E402
    build_calibration,
    headline_verdict,
    run_flagship_curve,
)
from verisim.net.config import DEFAULT_NET_CONFIG  # noqa: E402
from verisim.netoracle import ReferenceNetworkOracle  # noqa: E402


def _smoke_model():
    world_model, _ = train_flagship(FlagshipConfig.smoke())
    return world_model


def test_build_calibration_aligns_scores_and_breaches():
    wm = _smoke_model()
    cfg = FlagshipCurveConfig.smoke()
    scores, breaches = build_calibration(wm, cfg, ReferenceNetworkOracle(), DEFAULT_NET_CONFIG)
    assert len(scores) == len(breaches) > 0
    assert all(b in (0, 1) for b in breaches)
    assert all(s >= 0.0 for s in scores)


def test_run_flagship_curve_well_formed_and_ordered():
    wm = _smoke_model()
    cfg = FlagshipCurveConfig.smoke()
    points = run_flagship_curve(wm, cfg, oracle=ReferenceNetworkOracle())

    arms = {p.arm for p in points}
    assert {"floor", "ceiling", "fixed", "composed"} <= arms
    floor = next(p.h_mean for p in points if p.arm == "floor")
    ceiling = next(p.h_mean for p in points if p.arm == "ceiling")
    assert floor <= ceiling
    # budget monotonicity: every interior arm sits within [floor, ceiling]
    for p in points:
        if p.arm in ("fixed", "composed"):
            assert floor - 1e-9 <= p.h_mean <= ceiling + 1e-9
        assert p.ci_lo <= p.h_mean <= p.ci_hi or p.n == 1


def test_run_flagship_curve_is_deterministic():
    wm = _smoke_model()
    cfg = FlagshipCurveConfig.smoke()
    a = run_flagship_curve(wm, cfg, oracle=ReferenceNetworkOracle())
    b = run_flagship_curve(wm, cfg, oracle=ReferenceNetworkOracle())
    assert [(p.arm, p.rho, p.h_mean) for p in a] == [(p.arm, p.rho, p.h_mean) for p in b]


def test_headline_verdict_reports_h69_fields():
    wm = _smoke_model()
    points = run_flagship_curve(wm, FlagshipCurveConfig.smoke(), oracle=ReferenceNetworkOracle())
    v = headline_verdict(points)
    assert set(v) >= {"ceiling", "floor", "target_horizon", "h69_supported"}
    assert isinstance(v["h69_supported"], bool)
