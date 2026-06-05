"""Bits-to-correct over distributed deltas (SPEC-7 §9.4; the SPEC-2.1 §3 gate, generalized; DS3).

The MDL of the oracle's correction of the model's predicted ``DistDelta``: the symmetric difference
of the predicted and true edit multisets, costed by a per-edit code length. ``0`` iff the prediction
equals truth, smooth and monotone otherwise — the same scale-free gate the K-series used
for the filesystem, NW for the network, and HC for the host, now for the distributed log/replica
delta. ``delta_exact`` (``bits_to_correct == 0``) is the per-step "did the model assemble the exact
edit set?" question. Deterministic and dependency-free.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence

from verisim.dist.delta import DistEdit, edit_to_dict

_BITS_PER_SYMBOL = math.log2(64)  # nominal; any positive constant preserves the gate


def edit_symbols(edit: DistEdit) -> int:
    """Description length of one edit in symbols: the op tag plus its serialized fields."""
    return len(edit_to_dict(edit))


def _key(edit: DistEdit) -> str:
    return json.dumps(edit_to_dict(edit), sort_keys=True, separators=(",", ":"))


def correction_symbols(predicted: Sequence[DistEdit], true: Sequence[DistEdit]) -> int:
    """Symbol-length of the residual: the symmetric difference of the two edit multisets."""
    pred_by_key: dict[str, DistEdit] = {}
    true_by_key: dict[str, DistEdit] = {}
    pred_counts: Counter[str] = Counter()
    true_counts: Counter[str] = Counter()
    for edit in predicted:
        k = _key(edit)
        pred_by_key[k] = edit
        pred_counts[k] += 1
    for edit in true:
        k = _key(edit)
        true_by_key[k] = edit
        true_counts[k] += 1
    total = 0
    for k in set(pred_counts) | set(true_counts):
        residual = abs(pred_counts[k] - true_counts[k])
        if residual:
            edit = true_by_key.get(k) or pred_by_key[k]
            total += residual * edit_symbols(edit)
    return total


def bits_to_correct(predicted: Sequence[DistEdit], true: Sequence[DistEdit]) -> float:
    """Bits to encode the oracle's correction of ``predicted``. ``0.0`` iff equal as a multiset."""
    return correction_symbols(predicted, true) * _BITS_PER_SYMBOL


def delta_exact(predicted: Sequence[DistEdit], true: Sequence[DistEdit]) -> bool:
    """``True`` iff ``predicted`` equals ``true`` as an edit multiset (``bits_to_correct == 0``)."""
    return correction_symbols(predicted, true) == 0


def delta_exact_rate(
    pairs: Iterable[tuple[Sequence[DistEdit], Sequence[DistEdit]]],
) -> float:
    """Fraction of ``(predicted, true)`` delta pairs that match exactly. ``1.0`` over no pairs."""
    total = 0
    exact = 0
    for predicted, true in pairs:
        total += 1
        exact += delta_exact(predicted, true)
    return exact / total if total else 1.0


__all__ = [
    "bits_to_correct",
    "correction_symbols",
    "delta_exact",
    "delta_exact_rate",
    "edit_symbols",
]
