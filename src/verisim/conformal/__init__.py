"""Oracle-calibrated conformal consultation (SPEC-15).

A *calibration layer beside the loop*, not a change to it: the free, exact oracle is an unlimited,
distribution-free calibration set, so a consultation trigger can be given a finite-sample coverage
guarantee instead of a hand-set threshold. Three pieces, all torch-free (they operate on recorded
``(score, oracle-divergence)`` pairs):

  - :mod:`~verisim.conformal.calibrate` -- the split-conformal / conformal-risk-control threshold
    ``τ_α`` (and the region-conditional ``τ_α(r)``) from oracle-labeled scores;
  - :mod:`~verisim.conformal.policy` -- ``ConformalTriggered`` (calibrated ``UncertaintyTriggered``)
    and ``AdaptiveConformalTriggered`` (Gibbs--Candès ACI, fed the oracle's free per-step truth);
  - :mod:`~verisim.conformal.risk` -- conformal risk control on the graded undetected-breach loss.
"""

from .calibrate import (
    ConformalThreshold,
    calibrate_threshold,
    region_thresholds,
    undetected_rate,
)
from .policy import AdaptiveConformalTriggered, ConformalTriggered
from .risk import calibrate_risk_threshold, undetected_breach_loss

__all__ = [
    "AdaptiveConformalTriggered",
    "ConformalThreshold",
    "ConformalTriggered",
    "calibrate_risk_threshold",
    "calibrate_threshold",
    "region_thresholds",
    "undetected_breach_loss",
    "undetected_rate",
]
