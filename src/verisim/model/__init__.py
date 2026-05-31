"""The neural world model `M_θ`: vocabulary, tokenizer, grammar, transformer.

Importing this package pulls in PyTorch (the transformer/decoder), so it is part of the
optional ``[model]`` layer, not the dependency-free deterministic core. The :mod:`vocab`,
:mod:`tokenizer`, and :mod:`grammar` modules carry no torch dependency of their own
(importing those *module files* directly stays torch-free); importing them *via this
package* still runs this ``__init__`` and therefore pulls torch.
"""

from __future__ import annotations

from .decode import (
    constrained_decode,
    constrained_decode_state,
    constrained_decode_with_uncertainty,
)
from .full_state import FullStateWorldModel, state_to_delta
from .grammar import DeltaGrammar, StateGrammar
from .tokenizer import (
    encode_prompt,
    encode_state_target,
    encode_target,
    parse_state_target,
    parse_target,
)
from .transformer import GPT, GPTConfig
from .vocab import Vocab
from .world_model import NeuralWorldModel

__all__ = [
    "GPT",
    "DeltaGrammar",
    "FullStateWorldModel",
    "GPTConfig",
    "NeuralWorldModel",
    "StateGrammar",
    "Vocab",
    "constrained_decode",
    "constrained_decode_state",
    "constrained_decode_with_uncertainty",
    "encode_prompt",
    "encode_state_target",
    "encode_target",
    "parse_state_target",
    "parse_target",
    "state_to_delta",
]
