"""The host neural world model `M_θ` (HC4): vocabulary, tokenizer, grammar, decode.

Importing this package pulls in PyTorch (the transformer/decoder, reused from v0), so it is part of
the optional ``[model]`` layer, not the dependency-free deterministic core. The :mod:`vocab`,
:mod:`tokenizer`, and :mod:`grammar` modules carry no torch dependency of their own (importing those
*module files* directly stays torch-free); importing them *via this package* still runs this
``__init__`` and therefore pulls torch.

This is the **flat** HC4 arm (the DD-H1 baseline). The factored, interaction-graph-conditioned arm
the flat arm is measured against (SPEC-6 §6.1) is a later HC4 increment.
"""

from __future__ import annotations

from .dataset import build_host_dataset, host_examples_from_rollout
from .decode import constrained_decode, constrained_decode_with_uncertainty
from .grammar import HostDeltaGrammar
from .graph import HostGraph, build_host_graph, feature_dims
from .graph_model import GraphHostWorldModel, build_host_graph_model
from .graph_train import (
    build_host_graph_dataset,
    graph_teacher_forced_accuracy,
    train_host_graph_model,
)
from .tokenizer import (
    HostTokenizeError,
    encode_action,
    encode_prompt,
    encode_state,
    encode_target,
    parse_target,
)
from .vocab import HostVocab
from .world_model import NeuralHostWorldModel

__all__ = [
    "GraphHostWorldModel",
    "HostDeltaGrammar",
    "HostGraph",
    "HostTokenizeError",
    "HostVocab",
    "NeuralHostWorldModel",
    "build_host_dataset",
    "build_host_graph",
    "build_host_graph_dataset",
    "build_host_graph_model",
    "constrained_decode",
    "constrained_decode_with_uncertainty",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "feature_dims",
    "graph_teacher_forced_accuracy",
    "host_examples_from_rollout",
    "parse_target",
    "train_host_graph_model",
]
