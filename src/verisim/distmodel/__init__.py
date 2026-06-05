"""The distributed neural world model `M_θ` (SPEC-7 §6.1, DS4): vocab, tokenizer, grammar, decode.

DS4 increment 1 shipped the **dependency-free foundation** -- the closed token :class:`DistVocab`
and the bidirectional :mod:`tokenizer` (``(state, action) -> Δ`` encode + the exact inverse
``parse``). DS4 increment 2 adds the **learned arm**: the LL(1) constrained-decode
:class:`DistDeltaGrammar`, the :class:`NeuralDistWorldModel` over v0's
:class:`~verisim.model.transformer.GPT`, and the supervised dataset builders that feed the generic
:mod:`verisim.train` trainers. Those pull in PyTorch, so importing *this package* now requires the
optional ``[model]`` layer -- the network / host model packages have the same property. The
:mod:`vocab`, :mod:`tokenizer`, and :mod:`grammar` modules carry no torch dependency of their own
(importing those *module files* directly stays torch-free); importing them *via this package* still
runs this ``__init__`` and therefore pulls torch.

The high-risk, load-bearing piece remains the tokenizer round-trip: ``parse_target(encode_target(Δ))
== Δ`` for every delta the reference oracle produces (tested exhaustively in
``tests/test_dist_model.py``), so the learned arm is built on a verified serialization; constrained
decode then guarantees grammar-validity by construction regardless of the model's weights.
"""

from __future__ import annotations

from .dataset import build_dist_dataset, dist_examples_from_rollout
from .decode import constrained_decode, constrained_decode_with_uncertainty
from .grammar import DistDeltaGrammar
from .tokenizer import (
    DistTokenizeError,
    encode_action,
    encode_prompt,
    encode_state,
    encode_target,
    parse_target,
)
from .vocab import DEFAULT_MAX_INT, DistVocab
from .world_model import NeuralDistWorldModel

__all__ = [
    "DEFAULT_MAX_INT",
    "DistDeltaGrammar",
    "DistTokenizeError",
    "DistVocab",
    "NeuralDistWorldModel",
    "build_dist_dataset",
    "constrained_decode",
    "constrained_decode_with_uncertainty",
    "dist_examples_from_rollout",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "parse_target",
]
