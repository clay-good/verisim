"""``NeuralNetworkWorldModel``: the learned network ``M_θ`` behind the loop interface.

Wrapping the trained transformer + vocab in :meth:`predict_delta` makes the NW4 model a
drop-in for the NW5 partial-observation propose-verify-correct loop, exactly as v0's
:class:`~verisim.model.world_model.NeuralWorldModel` drops into the M5 loop: the loop never
knows whether it holds a baseline or the neural model.

This is the **flat** NW4 arm -- a from-scratch decoder-only transformer over the serialized
``(state, action)`` -> ``delta`` sequence. It is the H11 *flat-Markov baseline* against which
the message-passing/RSSM graph arm (SPEC-5 §6.1-6.2) is measured in EN4; under full
observability the RSSM belief degenerates to this Markov predictor (§6.2), so the graph arm
is the open NW4 work and lands with the partial-observation loop (NW5/NW7).
"""

from __future__ import annotations

from verisim.model.transformer import GPT
from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.edits import NetDelta

from .decode import constrained_decode, constrained_decode_with_uncertainty
from .grammar import NetDeltaGrammar
from .tokenizer import encode_prompt
from .vocab import NetVocab


class NeuralNetworkWorldModel:
    """The learned flat network ``M_θ`` (SPEC-5 §6.1)."""

    def __init__(self, model: GPT, vocab: NetVocab) -> None:
        self.model = model
        self.vocab = vocab
        self.grammar = NetDeltaGrammar(vocab)

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta:
        prompt = encode_prompt(state, action, self.vocab)
        return constrained_decode(self.model, prompt, self.vocab, self.grammar)

    def predict_delta_with_uncertainty(
        self, state: NetworkState, action: NetAction
    ) -> tuple[NetDelta, float]:
        """Predict the delta and the mean decode entropy (SPEC-5 §9.4).

        The uncertainty signal the NW5 loop's drift/uncertainty consultation policies will
        threshold (SPEC-5 §8.1), the network analogue of v0's M_θ uncertainty signal.
        """
        prompt = encode_prompt(state, action, self.vocab)
        return constrained_decode_with_uncertainty(self.model, prompt, self.vocab, self.grammar)
