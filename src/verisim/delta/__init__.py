"""Structured state deltas: types, ``apply``, and serialization (SPEC-2 §5.1)."""

from __future__ import annotations

from .apply import apply
from .edits import (
    Chmod,
    Create,
    Delete,
    Delta,
    Edit,
    Modify,
    Move,
    SetCwd,
    SetEnv,
    SetResult,
)
from .serialize import (
    delta_from_list,
    delta_to_list,
    edit_from_dict,
    edit_to_dict,
)

__all__ = [
    "Chmod",
    "Create",
    "Delete",
    "Delta",
    "Edit",
    "Modify",
    "Move",
    "SetCwd",
    "SetEnv",
    "SetResult",
    "apply",
    "delta_from_list",
    "delta_to_list",
    "edit_from_dict",
    "edit_to_dict",
]
