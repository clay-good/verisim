"""The split-conformal / conformal-risk-control threshold ``τ_α`` (SPEC-15 §2.1, §2.3).

The setting (SPEC-15 §0): each step carries a model uncertainty **score** ``s`` (the nonconformity
score -- ``belief_var`` for the graph arm, decode-entropy for the flat arm) and an oracle-revealed
**breach** indicator ``b = [divergence > ε]``. The consultation policy consults when ``s > τ`` (high
uncertainty -> verify); an *undetected breach* is a step that breached but was not consulted
(``b ∧ s ≤ τ``). The oracle is a free, exact, unlimited calibration set, so we can choose ``τ`` to
**guarantee** the undetected-breach rate ``≤ α`` distribution-free and finite-sample, instead of
hand-setting it (the EH2/ED2-smart thresholds were hand-set, SPEC-15 §0).

The threshold is the **largest** ``τ`` (fewest consultations) whose finite-sample-corrected
undetected-breach rate is ``≤ α`` -- conformal risk control for the monotone undetected-breach loss
(Angelopoulos et al., 2022): the loss is non-decreasing in ``τ`` (raise the threshold -> consult
less
-> more undetected breaches), and the ``(n·R̂ + 1)/(n+1)`` correction gives ``E[undetected] ≤ α``
for
*exchangeable* calibration/test data. The exchangeability caveat -- violated along an autoregressive
rollout -- is the load-bearing scientific question handled by the online ACI policy
(:mod:`~verisim.conformal.policy`, SPEC-15 §2.4, CF2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ConformalThreshold:
    """A calibrated threshold and the calibration diagnostics behind it (SPEC-15 §2.1)."""

    tau: float  # consult when score > tau
    alpha: float  # target undetected-breach rate
    n: int  # calibration-set size
    corrected_rate: float  # finite-sample-corrected undetected rate at ``tau`` (the certificate)
    consult_rate: float  # fraction of calibration steps consulted at ``tau`` (the cost)


def undetected_rate(
    scores: Sequence[float], breaches: Sequence[int], tau: float
) -> float:
    """Marginal undetected-breach rate at ``tau``: fraction of steps with ``breach ∧ score ≤ τ``."""
    n = len(scores)
    if n == 0:
        return 0.0
    miss = sum(1 for s, b in zip(scores, breaches, strict=True) if b and s <= tau)
    return miss / n


def calibrate_threshold(
    scores: Sequence[float], breaches: Sequence[int], alpha: float
) -> ConformalThreshold:
    """Largest ``τ`` whose finite-sample-corrected undetected-breach rate is ``≤ α`` (SPEC-15 §2.3).

    Conformal risk control for the monotone undetected-breach loss. Returns the threshold that
    consults
    *least* while certifying ``E[undetected] ≤ α``; lower ``α`` -> smaller ``τ`` -> more
    consultations
    (the ρ-vs-coverage frontier, CF1/H51). If even consulting on every step cannot certify
    (impossible
    for ``α ≥ 1/(n+1)``) the threshold falls below all scores (consult always).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        raise ValueError("need a non-empty calibration set")
    if len(breaches) != n:
        raise ValueError("scores and breaches must align")

    # Candidate thresholds: each observed score (consult when score > τ), largest first, plus a
    # sentinel below all scores (consult always -> zero undetected breaches).
    candidates = sorted(set(scores), reverse=True)
    below_all = math.nextafter(min(scores), -math.inf)
    candidates.append(below_all)

    best: ConformalThreshold | None = None
    for tau in candidates:
        miss = sum(1 for s, b in zip(scores, breaches, strict=True) if b and s <= tau)
        corrected = (miss + 1) / (n + 1)  # conservative finite-sample bound (bounded 0/1 loss)
        if corrected <= alpha:
            consult = sum(1 for s in scores if s > tau) / n
            best = ConformalThreshold(tau, alpha, n, corrected, consult)
            break  # candidates are descending, so the first feasible is the largest τ
    if best is None:  # cannot certify even by consulting always -> still return the consult-all τ
        consult = sum(1 for s in scores if s > below_all) / n
        best = ConformalThreshold(below_all, alpha, n, 1.0 / (n + 1), consult)
    return best


def region_thresholds(
    scores: Sequence[float],
    breaches: Sequence[int],
    regions: Sequence[int],
    alpha: float,
) -> dict[int, ConformalThreshold]:
    """A region-conditional ``τ_α(r)`` per region key -- the CQR-style heteroscedastic threshold.

    Faithful horizon is a *region* property (depth, EN7/H22), so a single global ``τ`` can hold
    marginal coverage while a region fails (SPEC-15 §2.1 caveat). Calibrating one threshold per
    region
    restores conditional coverage where the region has enough calibration points; regions with too
    few
    points (``< 1/α``) fall back to the global threshold so the certificate stays honest.
    """
    global_tau = calibrate_threshold(scores, breaches, alpha)
    min_points = math.ceil(1.0 / alpha)
    by_region: dict[int, tuple[list[float], list[int]]] = {}
    for s, b, r in zip(scores, breaches, regions, strict=True):
        bucket = by_region.setdefault(r, ([], []))
        bucket[0].append(s)
        bucket[1].append(b)
    out: dict[int, ConformalThreshold] = {}
    for r, (rs, rb) in by_region.items():
        out[r] = calibrate_threshold(rs, rb, alpha) if len(rs) >= min_points else global_tau
    return out
