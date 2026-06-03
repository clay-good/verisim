"""``NeuralHostWorldModel``: the learned host ``M_θ`` behind the loop interface.

Wrapping the trained transformer + vocab in :meth:`predict_delta` makes the HC4 model a drop-in for
the HC5 composed propose-verify-correct loop, exactly as v0's
:class:`~verisim.model.world_model.NeuralWorldModel` drops into the M5 loop and the network
:class:`~verisim.netmodel.world_model.NeuralNetworkWorldModel` drops into NW5: the loop never knows
whether it holds a baseline or the neural model.

This is the **flat** HC4 arm -- a from-scratch decoder-only transformer over the serialized
``(bundle_state, action)`` -> ``bundle_delta`` sequence. It is the DD-H1 *flat-serializer baseline*
against which the factored, interaction-graph-conditioned arm (SPEC-6 §6.1) is measured in EH4: the
flat arm flattens the composition the factored arm is built to exploit, so it is the honest floor
the factored model must beat.
"""

from __future__ import annotations

from verisim.host.action import HostAction
from verisim.host.delta import HostDelta
from verisim.host.state import HostState
from verisim.model.transformer import GPT

from .decode import constrained_decode, constrained_decode_with_uncertainty
from .grammar import HostDeltaGrammar
from .tokenizer import encode_prompt
from .vocab import HostVocab


class NeuralHostWorldModel:
    """The learned flat host ``M_θ`` (SPEC-6 §6.1, the DD-H1 baseline arm)."""

    def __init__(self, model: GPT, vocab: HostVocab) -> None:
        self.model = model
        self.vocab = vocab
        self.grammar = HostDeltaGrammar(vocab)

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta:
        prompt = encode_prompt(state, action, self.vocab)
        return constrained_decode(self.model, prompt, self.vocab, self.grammar)

    def predict_delta_with_uncertainty(
        self, state: HostState, action: HostAction
    ) -> tuple[HostDelta, float]:
        """Predict the bundle delta and the mean decode entropy (SPEC-6 §5.4).

        The uncertainty signal the HC5 loop's drift/uncertainty consultation policies will threshold
        (SPEC-6 §8.1), the host analogue of v0's M_θ uncertainty signal.
        """
        prompt = encode_prompt(state, action, self.vocab)
        return constrained_decode_with_uncertainty(self.model, prompt, self.vocab, self.grammar)
