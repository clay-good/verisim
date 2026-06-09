"""Matched-coverage subsampling over the ED6 ``_medium`` statistic (SPEC-17 CX3, H62).

ED6's ``+counterfactual`` arm is fault-heavier than its on-policy control, so its lift conflates
counterfactual *branching* with the fault *coverage* it carries (the SPEC-7 §10.1 caveat). CX3
disentangles them by matching the arms on *coverage* — the fraction of examples whose action
changes the distributed medium (a partition split / crash / heal, the
[`ed6._medium`](../experiments/ed6.py) statistic). This module is the deterministic bookkeeping that
does the matching: bin a pool of examples by whether each changes the medium, then draw a subsample
of a target size with a target medium-change fraction. There is no model and no oracle here — only
seeded sampling over flags the caller already computed, so the matching runs identically everywhere.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def coverage_rate(flags: Sequence[bool]) -> float:
    """Fraction of examples that change the medium (the CX3 coverage statistic)."""
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def feasible_match(
    pools_flags: Sequence[Sequence[bool]], *, coverage: float | None = None
) -> tuple[float, int]:
    """The largest ``(coverage, count)`` every pool can supply.

    A pool of ``c``-coverage needs ``count·c`` medium-changing and ``count·(1-c)`` non-changing
    examples, so its max feasible ``count`` at coverage ``c`` is bounded by both bins. The returned
    coverage is ``coverage`` if given (clamped to ``[0, 1]``), else the **minimum natural coverage**
    across pools (the highest coverage all pools reach without up-sampling). The count is the
    floor of the smallest per-pool feasible count, so every pool can be matched to ``(c, count)``
    exactly by :func:`match_coverage` with no replacement.
    """
    covs = [coverage_rate(f) for f in pools_flags]
    c = min(covs) if coverage is None else min(max(coverage, 0.0), 1.0)
    counts: list[int] = []
    for flags in pools_flags:
        n_change = sum(1 for f in flags if f)
        n_keep = len(flags) - n_change
        from_change = n_change / c if c > 0 else float("inf")
        from_keep = n_keep / (1.0 - c) if c < 1.0 else float("inf")
        counts.append(int(min(from_change, from_keep)))
    return c, (min(counts) if counts else 0)


def match_coverage(
    examples: Sequence[T], flags: Sequence[bool], *, target_coverage: float, target_count: int,
    rng: random.Random,
) -> list[T]:
    """Subsample ``examples`` to ``target_count`` with medium-change fraction ``target_coverage``.

    Bins by ``flags`` (``True`` = the example's action changes the medium), draws
    ``round(target_count·target_coverage)`` from the changing bin and the rest from the non-changing
    bin (each capped at what the bin holds, so sampling is without replacement), then shuffles. Both
    inputs must align index-for-index. Deterministic given ``rng``.
    """
    if len(examples) != len(flags):
        raise ValueError("examples and flags must align index-for-index")
    change = [e for e, f in zip(examples, flags, strict=True) if f]
    keep = [e for e, f in zip(examples, flags, strict=True) if not f]
    n_change = min(round(target_count * target_coverage), len(change))
    n_keep = min(target_count - n_change, len(keep))
    rng.shuffle(change)
    rng.shuffle(keep)
    out = change[:n_change] + keep[:n_keep]
    rng.shuffle(out)
    return out
