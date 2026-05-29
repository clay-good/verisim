"""``NeuralWorldModel``: the learned ``M_θ`` behind the loop's ``Model`` protocol.

Wrapping the trained transformer + vocab in :meth:`predict_delta` makes the M4
model a drop-in for the M5 propose-verify-correct loop (``verisim.loop.Model``):
the loop never knows whether it holds a baseline or the neural model.
"""

from __future__ import annotations

from verisim.delta.edits import Delta
from verisim.env.action import Action
from verisim.env.state import State

from .decode import constrained_decode
from .grammar import DeltaGrammar
from .tokenizer import encode_prompt
from .transformer import GPT
from .vocab import Vocab


class NeuralWorldModel:
    def __init__(self, model: GPT, vocab: Vocab) -> None:
        self.model = model
        self.vocab = vocab
        self.grammar = DeltaGrammar(vocab)

    def predict_delta(self, state: State, action: Action) -> Delta:
        prompt = encode_prompt(state, action, self.vocab)
        delta, _ = constrained_decode(self.model, prompt, self.vocab, self.grammar)
        return delta
