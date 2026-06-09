"""Conformal consultation policies (SPEC-15 §2.1, §2.4).

Two ``ConsultationPolicy`` implementations that drop into the shipped loop seam
([`loop/policy.py`](../loop/policy.py)):

  - :class:`ConformalTriggered` -- ``UncertaintyTriggered`` with a *calibrated* ``τ`` (CF1).
    Consult when the model's score exceeds the conformal threshold; the guarantee rides on the
    threshold, the policy is one line.
  - :class:`AdaptiveConformalTriggered` -- Gibbs--Candès ACI (2021). An autoregressive rollout
    drifts by construction, so the static ``τ`` may lose coverage as the state leaves the
    calibration
    distribution (the exchangeability violation, SPEC-15 §4/H52). ACI re-tunes the level ``α_t``
    online
    from each step's *realized* hit/miss -- and the verisim twist is that the realized truth is
    **free
    and exact at every step** (the oracle reveals the breach whether or not we consulted), so the
    online feedback ACI needs is available here in a way it never is in the time-series setting it
    came
    from. CF2 measures static-vs-ACI coverage along the rollout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from verisim.loop.policy import StepContext

from .calibrate import calibrate_threshold


@dataclass(frozen=True)
class ConformalTriggered:
    """Consult when the score exceeds the calibrated conformal threshold ``τ`` (SPEC-15 CF1)."""

    tau: float

    def should_consult(self, ctx: StepContext) -> bool:
        return ctx.signal > self.tau


@dataclass
class AdaptiveConformalTriggered:
    """Online conformal (ACI): track ``τ_t`` directly from realized breaches (SPEC-15 §2.4, CF2).

    Gibbs--Candès ACI in score space: ``τ_{t+1} = τ_t + γ(α − err_t)`` where ``err_t = 1`` iff step
    ``t`` was an *undetected* breach (breached but not consulted). A miss (``err=1``) drives ``τ``
    down
    (consult more); a clean step (``err=0``) lets ``τ`` drift up (consult less). The fixed point is
    ``E[err] = α``, so the long-run undetected rate returns to target **regardless of the data-
    generating process** -- which is exactly what a drifting autoregressive rollout needs (static
    split-conformal's fixed ``τ``, derived from the in-distribution calibration set, cannot reach
    the
    out-of-distribution regime). The threshold is *seeded* from the static calibration at level
    ``α``
    and then moves freely; the caller feeds the oracle's free, exact per-step truth via
    :meth:`update`.
    """

    cal_scores: Sequence[float]
    cal_breaches: Sequence[int]
    target_alpha: float
    gamma: float = 0.02
    tau: float = field(init=False)

    def __post_init__(self) -> None:
        self.tau = calibrate_threshold(self.cal_scores, self.cal_breaches, self.target_alpha).tau

    def should_consult(self, ctx: StepContext) -> bool:
        return ctx.signal > self.tau

    def update(self, *, breach: bool, consulted: bool) -> None:
        """ACI step from the oracle's free truth: a miss lowers ``τ`` (consult more), else it drifts
        up.
        """
        err = 1.0 if (breach and not consulted) else 0.0
        self.tau += self.gamma * (self.target_alpha - err)
