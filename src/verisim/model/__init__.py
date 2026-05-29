"""The neural world model `M_θ`: vocabulary, tokenizer, grammar, transformer.

Importing this package pulls in PyTorch (the transformer/decoder). The pure-Python
pieces -- :mod:`vocab`, :mod:`tokenizer`, :mod:`grammar` -- are importable directly
without torch if needed.
"""

from __future__ import annotations

from .decode import constrained_decode
from .grammar import DeltaGrammar
from .tokenizer import encode_prompt, encode_target, parse_target
from .transformer import GPT, GPTConfig
from .vocab import Vocab
from .world_model import NeuralWorldModel

__all__ = [
    "GPT",
    "DeltaGrammar",
    "GPTConfig",
    "NeuralWorldModel",
    "Vocab",
    "constrained_decode",
    "encode_prompt",
    "encode_target",
    "parse_target",
]
