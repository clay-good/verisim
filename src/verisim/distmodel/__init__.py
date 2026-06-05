"""The distributed neural world model `M_θ` (SPEC-7 §6.1, DS4): vocabulary + tokenizer.

DS4 increment 1 ships the **dependency-free foundation** -- the closed token :class:`DistVocab` and
the bidirectional :mod:`tokenizer` (``(state, action) -> Δ`` encode + the exact inverse ``parse``).
These carry no torch dependency, so importing this package stays torch-free for now; the constrained
decode grammar, the :class:`NeuralDistWorldModel`, and supervised training (which reuse v0's
:class:`~verisim.model.transformer.GPT` and the generic :mod:`verisim.train` trainers) are DS4
increment 2 and move this package into the optional ``[model]`` layer.

The high-risk, load-bearing piece is the tokenizer round-trip: ``parse_target(encode_target(Δ)) ==
Δ`` for every delta the reference oracle produces (tested exhaustively over rollouts in
``tests/test_dist_model.py``), so the learned arm is built on a verified serialization.
"""

from __future__ import annotations

from .tokenizer import (
    DistTokenizeError,
    encode_action,
    encode_prompt,
    encode_state,
    encode_target,
    parse_target,
)
from .vocab import DEFAULT_MAX_INT, DistVocab

__all__ = [
    "DEFAULT_MAX_INT",
    "DistTokenizeError",
    "DistVocab",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "parse_target",
]
