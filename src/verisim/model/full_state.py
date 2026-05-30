"""``FullStateWorldModel``: the full-state prediction target (SPEC-2 §9 ablation).

The §10 representation ablation contrasts two prediction targets for ``M_θ``:

  - **delta** (:class:`~verisim.model.world_model.NeuralWorldModel`, the primary):
    predict the structured edits the action makes;
  - **full state** (this module, the alternative): regenerate the *whole* next
    state ``ŝ'`` token by token.

SPEC.md §6.1 argues delta prediction is the better target (it bounds the
hallucination surface and localizes verification); this model is what lets that
claim be *measured* rather than asserted. The decode is constrained to a valid
state by :class:`~verisim.model.grammar.StateGrammar`, so -- exactly as for the
delta model -- grammar-validity is guaranteed and only *faithfulness* is at stake.

To drop into the propose-verify-correct loop unchanged (which is generic over the
``Model.predict_delta`` protocol and applies the returned delta), the model emits
the delta that transforms the *current* state into its decoded prediction
(:func:`state_to_delta`), so ``apply(state, predict_delta(state, a)) == ŝ'``.
"""

from __future__ import annotations

from verisim.delta.edits import Create, Delete, Delta, SetCwd, SetEnv, SetResult
from verisim.env.action import Action
from verisim.env.state import State

from .decode import constrained_decode_state
from .grammar import StateGrammar
from .tokenizer import encode_prompt
from .transformer import GPT
from .vocab import Vocab


def state_to_delta(old: State, new: State) -> Delta:
    """The delta that transforms ``old`` into ``new``: ``apply(old, Δ) == new``.

    Used to express a full-state prediction as a delta so it flows through the
    loop's shared ``apply`` path. The diff is exact for every fact the v0 delta
    vocabulary can express. The *one* inexpressible case -- removing an env key
    present in ``old`` but absent from ``new`` -- cannot arise from oracle truth
    (``export`` only ever adds keys, so env never shrinks), and a model that
    hallucinated a dropped key would simply retain it, which can only *help* match
    the (never-shrinking) truth. Paths and keys are emitted in sorted order so the
    delta is a deterministic function of the two states.
    """
    delta: Delta = []
    for path in sorted(old.fs):
        if path not in new.fs:
            delta.append(Delete(path))
    for path in sorted(new.fs):
        if old.fs.get(path) != new.fs[path]:
            delta.append(Create(path, new.fs[path]))
    if old.cwd != new.cwd:
        delta.append(SetCwd(new.cwd))
    for key in sorted(new.env):
        if old.env.get(key) != new.env[key]:
            delta.append(SetEnv(key, new.env[key]))
    if old.last != new.last:
        delta.append(SetResult(new.last.exit_code, new.last.stdout_hash))
    return delta


class FullStateWorldModel:
    """``M_θ`` predicting the full next state (the full-state representation arm)."""

    def __init__(self, model: GPT, vocab: Vocab) -> None:
        self.model = model
        self.vocab = vocab
        self.grammar = StateGrammar(vocab)

    def predict_state(self, state: State, action: Action) -> State:
        prompt = encode_prompt(state, action, self.vocab)
        return constrained_decode_state(self.model, prompt, self.vocab, self.grammar)

    def predict_delta(self, state: State, action: Action) -> Delta:
        return state_to_delta(state, self.predict_state(state, action))
