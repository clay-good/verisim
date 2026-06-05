"""``NeuralDistWorldModel``: the learned distributed ``M_θ`` behind the loop interface.

Wrapping the trained transformer + vocab in :meth:`predict_delta` makes the DS4 model a drop-in for
the DS5 tiered propose-verify-correct loop, exactly as v0's
:class:`~verisim.model.world_model.NeuralWorldModel` drops into the M5 loop and the network's
:class:`~verisim.netmodel.world_model.NeuralNetworkWorldModel` into NW5: the loop never knows
whether it holds a baseline (``DistNullModel`` / ``DistOracleBackedModel`` / ``DistNoisyModel``) or
the learned model -- it satisfies the same :class:`~verisim.distloop.model.DistModel` protocol.

This is the **flat** DS4 arm -- a from-scratch decoder-only transformer over the serialized
``(state, action)`` -> ``delta`` sequence, the distributed analogue of the flat net/host arms and
v0's flat-Markov predictor. The service-graph message-passing + RSSM-belief arm (SPEC-7 §6.1-6.2) is
the open DS4/DS7 work; under full observability the RSSM belief degenerates to this Markov predictor
(§6.2), so the flat arm is the right first learned model and the baseline the graph arm is measured
against.
"""

from __future__ import annotations

from verisim.dist.action import DistAction
from verisim.dist.delta import DistDelta
from verisim.dist.state import DistributedState
from verisim.model.transformer import GPT

from .decode import constrained_decode, constrained_decode_with_uncertainty
from .grammar import DistDeltaGrammar
from .tokenizer import encode_prompt
from .vocab import DistVocab


class NeuralDistWorldModel:
    """The learned flat distributed ``M_θ`` (SPEC-7 §6.1)."""

    def __init__(self, model: GPT, vocab: DistVocab) -> None:
        self.model = model
        self.vocab = vocab
        self.grammar = DistDeltaGrammar(vocab)

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        prompt = encode_prompt(state, action, self.vocab)
        return constrained_decode(self.model, prompt, self.vocab, state, action, self.grammar)

    def predict_delta_with_uncertainty(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, float]:
        """Predict the delta and the mean decode entropy (SPEC-7 §9.4).

        The uncertainty signal the DS5 loop's drift/uncertainty consultation policies will threshold
        (SPEC-7 §8.1), the distributed analogue of v0's M_θ uncertainty signal.
        """
        prompt = encode_prompt(state, action, self.vocab)
        return constrained_decode_with_uncertainty(
            self.model, prompt, self.vocab, state, action, self.grammar
        )
