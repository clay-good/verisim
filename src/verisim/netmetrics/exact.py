"""Delta-exact rate: the per-step exact-match metric (SPEC-5 §12.1, the report's next datum).

Token-level teacher-forced accuracy (``graph_teacher_forced_accuracy``) and the free-running
faithful horizon ``H_ε`` bracket a model from two sides, but neither is the *per-step* question a
delta predictor is actually asked: **did the model assemble the exact true edit set this step?**
Token accuracy can be high while the parsed delta is wrong (one mispredicted host id flips an
edit), and ``H_ε`` collapses to ``0`` the moment any single step exceeds ε, hiding how often the
model is exactly right. ``delta_exact`` is the missing middle: it is ``1`` iff the predicted delta
equals the true delta as a *multiset* of edits -- i.e. iff :func:`bits_to_correct` is ``0`` -- so it
shares the bits-to-correct gate's order-independence and needs no new comparison logic.

Pure and dependency-free, like the rest of the metric core.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from verisim.netdelta.edits import NetEdit

from .bits import correction_symbols


def delta_exact(predicted: Sequence[NetEdit], true: Sequence[NetEdit]) -> bool:
    """``True`` iff ``predicted`` equals ``true`` as an edit multiset (``bits_to_correct == 0``)."""
    return correction_symbols(predicted, true) == 0


def delta_exact_rate(pairs: Iterable[tuple[Sequence[NetEdit], Sequence[NetEdit]]]) -> float:
    """Fraction of ``(predicted, true)`` delta pairs that match exactly. ``1.0`` over no pairs."""
    total = 0
    exact = 0
    for predicted, true in pairs:
        total += 1
        exact += delta_exact(predicted, true)
    return exact / total if total else 1.0


__all__ = ["delta_exact", "delta_exact_rate"]
