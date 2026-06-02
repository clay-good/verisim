"""Bits-to-correct over host bundle deltas, composed **and per-subsystem** (SPEC-6 §5.4, HC3).

The scale-free gate (SPEC-2.1 §3, generalized to the network in :mod:`verisim.netmetrics.bits`) now
**decomposes by subsystem**: ``bits-to-correct(Δ̂, Δ) = Σ_subsystem MDL(correction on that
subsystem)`` (SPEC-6 §5.4). ``0`` iff the prediction equals truth on every subsystem; smooth and
monotone otherwise; deterministic and dependency-free. The per-subsystem breakdown is the diagnostic
that tells the engine (and H13) *where* faithfulness is leaking -- process table, fds, FS, or the
exit observation -- which a single scalar hides.

The composition is exact: the ``fs`` subsystem's correction is delegated to v0's own
:func:`~verisim.metrics.bits.correction_symbols` over the embedded edits (the FS sub-oracle's gate,
reused verbatim), exactly as the oracle delegates the file effect (SPEC-6 §5.1). Process/fd/exit
edits are costed by their serialized field count, matching :mod:`verisim.netmetrics.bits`.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence

from verisim.host.delta import (
    CredChange,
    FdClose,
    FdOpen,
    FsDelta,
    HostEdit,
    ProcExit,
    ProcSpawn,
    SetExit,
    edit_to_dict,
)
from verisim.metrics.bits import correction_symbols as fs_correction_symbols

_BITS_PER_SYMBOL = math.log2(64)  # nominal; any positive constant preserves the gate


def _subsystem_of(edit: HostEdit) -> str:
    """Which subsystem an edit corrects (§5.4). ``CredChange`` is process (privilege) state."""
    if isinstance(edit, (ProcSpawn, ProcExit, CredChange)):
        return "proc"
    if isinstance(edit, (FdOpen, FdClose)):
        return "fd"
    if isinstance(edit, FsDelta):
        return "fs"
    assert isinstance(edit, SetExit)
    return "global"


def edit_symbols(edit: HostEdit) -> int:
    """Symbol-length of one host edit: its serialized field count (``op`` + fields).

    An :class:`~verisim.host.delta.FsDelta` is the sum of its embedded v0 edits' symbol-lengths, so
    a bigger filesystem correction costs proportionally more (the composition is exact, §5.4).
    """
    if isinstance(edit, FsDelta):
        # The empty-vs-edit comparison gives the symbol-length of the embedded delta itself.
        return fs_correction_symbols(edit.edits, [])
    return len(edit_to_dict(edit))


def _key(edit: HostEdit) -> str:
    return json.dumps(edit_to_dict(edit), sort_keys=True, separators=(",", ":"))


def _host_correction(predicted: Sequence[HostEdit], true: Sequence[HostEdit]) -> int:
    """Symbol-length of the residual over *non-FS* host edits (multiset symmetric difference)."""
    pred_by_key: dict[str, HostEdit] = {}
    true_by_key: dict[str, HostEdit] = {}
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


def correction_symbols_by_subsystem(
    predicted: Sequence[HostEdit], true: Sequence[HostEdit]
) -> dict[str, int]:
    """Residual symbol-length per subsystem. ``fs`` delegates to v0's gate over embedded edits."""
    by_sub: dict[str, int] = {"proc": 0, "fd": 0, "fs": 0, "global": 0}
    # The FS subsystem: concatenate the embedded v0 deltas on each side and use v0's correction.
    pred_fs = [e for edit in predicted if isinstance(edit, FsDelta) for e in edit.edits]
    true_fs = [e for edit in true if isinstance(edit, FsDelta) for e in edit.edits]
    by_sub["fs"] = fs_correction_symbols(pred_fs, true_fs)
    # The non-FS subsystems: a multiset symmetric difference, partitioned by subsystem.
    for sub in ("proc", "fd", "global"):
        pred_sub = [e for e in predicted if not isinstance(e, FsDelta) and _subsystem_of(e) == sub]
        true_sub = [e for e in true if not isinstance(e, FsDelta) and _subsystem_of(e) == sub]
        by_sub[sub] = _host_correction(pred_sub, true_sub)
    return by_sub


def correction_symbols(predicted: Sequence[HostEdit], true: Sequence[HostEdit]) -> int:
    """Composed residual symbol-length: the sum over subsystems (SPEC-6 §5.4)."""
    return sum(correction_symbols_by_subsystem(predicted, true).values())


def bits_to_correct(predicted: Sequence[HostEdit], true: Sequence[HostEdit]) -> float:
    """Composed bits to encode the oracle's correction of ``predicted``. ``0.0`` iff equal."""
    return correction_symbols(predicted, true) * _BITS_PER_SYMBOL


def bits_to_correct_by_subsystem(
    predicted: Sequence[HostEdit], true: Sequence[HostEdit]
) -> dict[str, float]:
    """Per-subsystem bits-to-correct -- the §5.4 decomposition that localizes the leak."""
    return {
        sub: n * _BITS_PER_SYMBOL
        for sub, n in correction_symbols_by_subsystem(predicted, true).items()
    }


def delta_exact(predicted: Sequence[HostEdit], true: Sequence[HostEdit]) -> bool:
    """``True`` iff ``predicted`` equals ``true`` as a bundle multiset (``bits_to_correct==0``)."""
    return correction_symbols(predicted, true) == 0


__all__ = [
    "bits_to_correct",
    "bits_to_correct_by_subsystem",
    "correction_symbols",
    "correction_symbols_by_subsystem",
    "delta_exact",
    "edit_symbols",
]
