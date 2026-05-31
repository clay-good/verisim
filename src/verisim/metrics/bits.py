"""Bits-to-correct: the smooth, scale-free optimization gate (SPEC-2.1 §3, SPEC-3 §7).

The faithful-horizon metric ``H_ε`` and the 0/1 exact-match accuracy are the *science*
but poor *optimization signals* — sparse and flat at v0 scale (see docs/report.md). The
autoresearch ratchet and the K-series need a dense, comparable scalar to climb. This
module provides it: the **description length of the oracle's correction of the model's
predicted delta**.

Concretely (SPEC-3 §7.2's "simple prefix code" surrogate): represent a delta as a multiset
of edits, take the symmetric difference between the predicted and the true delta, and sum a
fixed per-edit code length over the residual. The result is

  - **0 iff the prediction equals the truth** (as edit multisets),
  - **monotone** in how wrong the prediction is (more/larger residual edits cost more bits),
  - **deterministic and dependency-free** (no model, no torch — it lives in the metric core),
  - **scale-free** in the sense that a perfect model needs 0 bits regardless of model/size,

so it is exactly the kind of signal a keep-if-better ratchet can climb where ``H_ε`` is flat.

This is a generalization of the §7.1 divergence metric from a *state* set-difference to a
*delta* description length; the bits/symbol constant is nominal (documented below) — what
matters for a gate is zero-iff-equal and monotonicity, both of which hold for any positive
constant.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence

from verisim.delta.edits import (
    Chmod,
    Create,
    Delete,
    Edit,
    Modify,
    Move,
    SetCwd,
    SetEnv,
    SetResult,
)
from verisim.delta.serialize import edit_to_dict
from verisim.env.state import File

# A nominal alphabet size for the symbol code. The absolute value is immaterial to the
# gate (any positive constant preserves zero-iff-equal and monotonicity); 64 symbols ->
# 6 bits/symbol is a readable default consistent with the small v0 token vocabulary.
_BITS_PER_SYMBOL = math.log2(64)


def _path_symbols(path: str) -> int:
    """A path costs one symbol per non-empty segment plus one for the path frame."""
    return 1 + sum(1 for seg in path.split("/") if seg)


def _content_symbols(content: str) -> int:
    """Content cost, monotone in length (the content frame plus its characters)."""
    return 1 + len(content)


def edit_symbols(edit: Edit) -> int:
    """Description length of one edit, in symbols, under the fixed code (SPEC-3 §7.2)."""
    if isinstance(edit, Create):
        node = edit.node
        # File: <file> + content + mode; Dir: <dir> + mode.
        node_cost = 1 + _content_symbols(node.content) + 1 if isinstance(node, File) else 2
        return 1 + _path_symbols(edit.path) + node_cost
    if isinstance(edit, Delete):
        return 1 + _path_symbols(edit.path)
    if isinstance(edit, Modify):
        return 1 + _path_symbols(edit.path) + _content_symbols(edit.content)
    if isinstance(edit, Move):
        return 1 + _path_symbols(edit.src) + _path_symbols(edit.dst)
    if isinstance(edit, Chmod):
        return 1 + _path_symbols(edit.path) + 1
    if isinstance(edit, SetCwd):
        return 1 + _path_symbols(edit.path)
    if isinstance(edit, SetEnv):
        return 1 + 1 + 1  # op key token
    # SetResult: op exit stdout-hash (the hash is opaque -> one symbol)
    assert isinstance(edit, SetResult)
    return 1 + 1 + 1


def _key(edit: Edit) -> str:
    """A canonical, hashable identity for an edit (for multiset comparison)."""
    return json.dumps(edit_to_dict(edit), sort_keys=True, separators=(",", ":"))


def correction_symbols(predicted: Sequence[Edit], true: Sequence[Edit]) -> int:
    """Symbol-length of the residual: the symmetric difference of the two edit multisets."""
    pred_by_key: dict[str, Edit] = {}
    true_by_key: dict[str, Edit] = {}
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


def bits_to_correct(predicted: Sequence[Edit], true: Sequence[Edit]) -> float:
    """Bits to encode the oracle's correction of ``predicted``. ``0.0`` iff equal.

    The §7.2 simple-prefix-code surrogate: the symmetric difference of the predicted and
    true delta multisets, costed by :func:`edit_symbols` at a nominal bits/symbol. Smooth,
    monotone, deterministic, and 0 exactly when the prediction matches the oracle's truth.
    """
    return correction_symbols(predicted, true) * _BITS_PER_SYMBOL


__all__ = ["bits_to_correct", "correction_symbols", "edit_symbols"]
