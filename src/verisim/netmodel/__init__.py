"""The network neural world model `M_θ` (NW4): vocabulary, tokenizer, grammar, decode.

Importing this package pulls in PyTorch (the transformer/decoder, reused from v0). The
pure-Python pieces -- :mod:`vocab`, :mod:`tokenizer`, :mod:`grammar` -- are importable
directly without torch if needed.
"""

from __future__ import annotations

from .dataset import build_net_dataset, net_examples_from_rollout
from .decode import constrained_decode, constrained_decode_with_uncertainty
from .grammar import NetDeltaGrammar
from .tokenizer import (
    NetTokenizeError,
    encode_action,
    encode_prompt,
    encode_state,
    encode_target,
    parse_target,
)
from .vocab import NetVocab
from .world_model import NeuralNetworkWorldModel

__all__ = [
    "NetDeltaGrammar",
    "NetTokenizeError",
    "NetVocab",
    "NeuralNetworkWorldModel",
    "build_net_dataset",
    "constrained_decode",
    "constrained_decode_with_uncertainty",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "net_examples_from_rollout",
    "parse_target",
]
