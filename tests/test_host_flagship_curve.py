"""HFL1 host flagship curve tests (SPEC-19, H84).

The contract: the composed consultation policy fires on the entropy trigger OR the speculative
window backstop, the headline verdict (composed-beats-fixed) logic is correct, and the four-arm
sweep is well-formed on the smoke checkpoint. The real cross-world win comes from the frontier run.
"""

from __future__ import annotations

import pytest

from verisim.experiments.host_flagship_curve import (
    CurvePoint,
    HostComposedConsult,
    headline_verdict,
)
from verisim.loop.policy import StepContext


def test_composed_consult_triggers_on_signal_or_window():
    pol = HostComposedConsult(tau=0.5, window=4)
    # high signal -> consult regardless of step
    assert pol.should_consult(StepContext(step=0, signal=0.9, cumulative_signal=0.9))
    # low signal, not at the window boundary -> hold
    assert not pol.should_consult(StepContext(step=0, signal=0.1, cumulative_signal=0.1))
    # low signal but the window has elapsed (step+1 % 4 == 0) -> consult (the backstop)
    assert pol.should_consult(StepContext(step=3, signal=0.1, cumulative_signal=0.4))


def test_composed_consult_rejects_bad_window():
    with pytest.raises(ValueError):
        HostComposedConsult(tau=0.5, window=0)


def test_headline_verdict_logic():
    def pt(arm, rho, h):
        return CurvePoint(arm, rho, h, h, h, 4)

    # composed beats fixed at every interior ρ -> supported, with the right relative lift
    pts = [
        pt("floor", 0.0, 9.0), pt("ceiling", 1.0, 48.0),
        pt("fixed", 0.2, 9.0), pt("composed", 0.2, 13.5),
        pt("fixed", 0.5, 13.0), pt("composed", 0.5, 20.75),
    ]
    v = headline_verdict(pts)
    assert v["composed_beats_fixed"]
    assert v["composed_minus_fixed"][0.2] == pytest.approx(4.5)
    assert v["composed_rel_lift"][0.5] == pytest.approx(0.5961538, rel=1e-3)
    assert v["best_rel_lift"] == pytest.approx(0.5961538, rel=1e-3)
    # a tie everywhere -> not the win
    flat = [pt("floor", 0.0, 9.0), pt("ceiling", 1.0, 48.0),
            pt("fixed", 0.2, 9.0), pt("composed", 0.2, 9.0)]
    assert not headline_verdict(flat)["composed_beats_fixed"]


# --- torch-gated: the real four-arm sweep --------------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.host_flagship import (  # noqa: E402
    HostFlagshipConfig,
    train_host_flagship,
)
from verisim.experiments.host_flagship_curve import (  # noqa: E402
    HostFlagshipCurveConfig,
    run_host_flagship_curve,
)


def test_host_flagship_curve_well_formed():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    points = run_host_flagship_curve(model, HostFlagshipCurveConfig.smoke())
    arms = {(p.arm, p.rho) for p in points}
    assert ("floor", 0.0) in arms and ("ceiling", 1.0) in arms
    for p in points:
        assert p.h_mean >= 0.0 and p.ci_lo <= p.h_mean <= p.ci_hi
    # the ceiling (oracle every step) is the longest horizon; the floor (ρ=0) the shortest
    ceiling = next(p.h_mean for p in points if p.arm == "ceiling")
    floor = next(p.h_mean for p in points if p.arm == "floor")
    assert ceiling >= floor
    v = headline_verdict(points)
    assert set(v) >= {"floor", "ceiling", "composed_beats_fixed", "best_rel_lift"}
