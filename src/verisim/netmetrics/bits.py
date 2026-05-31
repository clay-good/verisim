"""Bits-to-correct over network graph deltas (SPEC-5 §5.4; the SPEC-2.1 §3 gate, generalized).

The MDL of the oracle's correction of the model's predicted graph delta: the symmetric
difference of the predicted and true edit multisets, costed by a fixed per-edit code length.
``0`` iff the prediction equals truth, smooth and monotone otherwise, deterministic and
dependency-free -- the same smooth optimization gate the K-series used for the filesystem,
now for the network.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence

from verisim.netdelta.edits import NetEdit
from verisim.netdelta.serialize import edit_to_dict

_BITS_PER_SYMBOL = math.log2(64)  # nominal; any positive constant preserves the gate


def edit_symbols(edit: NetEdit) -> int:
    """Description length of one edit in symbols: one for the op plus one per field."""
    return len(edit_to_dict(edit))  # {"op": ..., field: ...} -> op + fields


def _key(edit: NetEdit) -> str:
    return json.dumps(edit_to_dict(edit), sort_keys=True, separators=(",", ":"))


def correction_symbols(predicted: Sequence[NetEdit], true: Sequence[NetEdit]) -> int:
    """Symbol-length of the residual: the symmetric difference of the two edit multisets."""
    pred_by_key: dict[str, NetEdit] = {}
    true_by_key: dict[str, NetEdit] = {}
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


def bits_to_correct(predicted: Sequence[NetEdit], true: Sequence[NetEdit]) -> float:
    """Bits to encode the oracle's correction of ``predicted``. ``0.0`` iff equal."""
    return correction_symbols(predicted, true) * _BITS_PER_SYMBOL


__all__ = ["bits_to_correct", "correction_symbols", "edit_symbols"]
