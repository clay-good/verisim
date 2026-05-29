"""Faithful horizon ``H_ε`` (SPEC.md §5.1, SPEC-2 §7.2) -- the headline metric.

``H_ε`` is the largest number of leading steps for which an autoregressive
rollout stays within divergence ``ε`` of the ground-truth rollout. Operationally,
given the per-step divergence trajectory ``[d_0, d_1, ...]`` (``d_t`` = divergence
*after* step ``t``), ``H_ε`` is the length of the longest faithful prefix: the
index of the first step whose divergence exceeds ``ε`` (or the full length if none
does). ``H_ε = 0`` means the very first step already diverged past ``ε``.
"""

from __future__ import annotations

from collections.abc import Sequence


def faithful_horizon(divergences: Sequence[float], epsilon: float) -> int:
    """Length of the longest faithful prefix of ``divergences`` at tolerance ``ε``."""
    for t, d in enumerate(divergences):
        if d > epsilon:
            return t
    return len(divergences)
