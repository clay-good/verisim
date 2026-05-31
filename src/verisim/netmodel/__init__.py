"""The network neural world model `M_θ` (NW4): vocabulary, tokenizer, grammar, decode.

Importing this package pulls in PyTorch (the transformer/decoder, reused from v0), so it is
part of the optional ``[model]`` layer, not the dependency-free deterministic core. The
:mod:`vocab`, :mod:`tokenizer`, and :mod:`grammar` modules carry no torch dependency of
their own (importing those *module files* directly stays torch-free); importing them *via
this package* still runs this ``__init__`` and therefore pulls torch.
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
