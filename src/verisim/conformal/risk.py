"""Conformal risk control on the graded undetected-breach loss (SPEC-15 §2.3, CF3).

Split conformal (:mod:`~verisim.conformal.calibrate`) controls the *miscoverage event* -- a 0/1
undetected-breach indicator. But the risk verisim actually cares about is graded: a breach that
lands
*just* past ``ε`` is a near-miss; one that lands far past it is a silent catastrophe. Conformal risk
control (Angelopoulos et al., 2022) bounds the expectation of any **monotone** loss, so we set ``τ``
to guarantee ``E[graded undetected-breach loss] ≤ α`` directly -- the metric that matters, not a
proxy.

The loss is non-decreasing in ``τ`` (raise the threshold -> consult less -> more, and more severe,
undetected breaches), bounded in ``[0, 1]``, so the same ``(n·R̂ + 1)/(n+1)`` finite-sample
correction
applies. CF3 asks whether this buys anything over the coverage-only threshold (H54): if the loss and
the miscoverage event coincide on this world, risk control is a clean no-op (banked); if the breach
loss is heavy-tailed, it tightens ``τ`` where the indicator would not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .calibrate import ConformalThreshold


def undetected_breach_loss(
    divergence: float, epsilon: float, *, consulted: bool, overshoot_scale: float | None = None
) -> float:
    """Graded severity of an undetected breach, bounded ``[0, 1]`` (SPEC-15 §2.3).

    ``0`` if the step did not breach (``divergence ≤ ε``) or was consulted (the breach is caught);
    otherwise the normalized overshoot ``min(1, (divergence − ε) / scale)`` -- a near-miss scores
    low,
    a silent catastrophe scores near ``1``. ``overshoot_scale`` sets the divergence overshoot that
    counts as a full-severity (``1``) breach; default ``1 − ε`` (the worst possible overshoot), but
    callers pass a breach-magnitude scale (e.g. a high divergence quantile) so the graded loss is on
    a
    comparable footing to the 0/1 indicator rather than near-zero on small-fraction divergences.
    """
    if consulted or divergence <= epsilon:
        return 0.0
    scale = overshoot_scale if overshoot_scale is not None else (1.0 - epsilon)
    if scale <= 0.0:
        return 1.0
    return min(1.0, (divergence - epsilon) / scale)


def calibrate_risk_threshold(
    scores: Sequence[float],
    divergences: Sequence[float],
    epsilon: float,
    alpha: float,
    *,
    overshoot_scale: float | None = None,
) -> ConformalThreshold:
    """Largest ``τ`` whose finite-sample-corrected *graded* undetected-breach loss is ``≤ α`` (CF3).

    Conformal risk control for the monotone graded loss: consult when ``score > τ``; a step's loss
    is
    :func:`undetected_breach_loss` evaluated at ``consulted = score > τ``. Returns the threshold
    that
    consults least while certifying ``E[graded loss] ≤ α``. ``overshoot_scale`` calibrates the
    severity
    normalization (passed through to the loss).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        raise ValueError("need a non-empty calibration set")
    if len(divergences) != n:
        raise ValueError("scores and divergences must align")

    candidates = sorted(set(scores), reverse=True)
    below_all = math.nextafter(min(scores), -math.inf)
    candidates.append(below_all)

    for tau in candidates:
        total = sum(
            undetected_breach_loss(d, epsilon, consulted=(s > tau), overshoot_scale=overshoot_scale)
            for s, d in zip(scores, divergences, strict=True)
        )
        corrected = (total + 1.0) / (n + 1)  # bounded-[0,1] loss, conservative slack
        if corrected <= alpha:
            consult = sum(1 for s in scores if s > tau) / n
            return ConformalThreshold(tau, alpha, n, corrected, consult)
    consult = sum(1 for s in scores if s > below_all) / n
    return ConformalThreshold(below_all, alpha, n, 1.0 / (n + 1), consult)
