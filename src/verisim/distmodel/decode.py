"""Constrained greedy decoding to a grammar-valid distributed delta (SPEC-7 §6.1, DS4 incr 2).

The distributed analogue of v0's :mod:`verisim.model.decode` and the network
:mod:`verisim.netmodel.decode`. At each step the model's logits are masked to the grammar's
allowed next tokens (:class:`DistDeltaGrammar`), so the decoded sequence is always parseable into a
valid :class:`~verisim.dist.delta.DistDelta` regardless of the model's weights. A ``max_edits`` cap
forces termination by restricting the top-level choice to ``<eos>`` once enough edits are emitted
(``<eos>`` is always grammar-valid at ``DELTA``).

The one distributed-specific wrinkle: the parse step needs the step's ``(state, action)`` to rebuild
the off-band ``EventAppend`` (its content is a pure function of the context, never decoded -- see
:mod:`tokenizer`), so the decode entrypoints thread them through to :func:`parse_target`. The flat
DS4 arm reuses v0's :class:`~verisim.model.transformer.GPT` directly.
"""

from __future__ import annotations

import torch

from verisim.dist.action import DistAction
from verisim.dist.delta import DistDelta
from verisim.dist.state import DistributedState
from verisim.model.transformer import GPT

from .grammar import DistDeltaGrammar
from .tokenizer import parse_target
from .vocab import DistVocab

#: Actions whose delta carries a causal-log ``EventAppend`` (SPEC-7 §5.1): only client ops, whose
#: first argument is the coordinator node the event is logged at. Fault/time ops never append an
#: event, so the decoder forbids ``<event_append>`` for them -- it is the one op whose
#: reconstruction (:func:`~verisim.distmodel.tokenizer.parse_target`) reads ``action.args[0]`` as a
#: node, so a free-running model must not emit a delta shape the oracle's language never produces.
_CLIENT_OPS = frozenset({"get", "put", "cas"})


@torch.no_grad()
def _decode_core(
    model: GPT,
    prompt_ids: list[int],
    vocab: DistVocab,
    state: DistributedState,
    action: DistAction,
    grammar: DistDeltaGrammar | None,
    *,
    max_edits: int,
    max_new_tokens: int,
) -> tuple[DistDelta, float]:
    """Greedy constrained decode; return ``(delta, mean_entropy)``.

    ``mean_entropy`` is the mean over decoded steps of the Shannon entropy (nats) of the masked
    next-token distribution -- the model's per-prediction uncertainty (SPEC-7 §9.4, the calibration
    diagnostic feeding the ``π_c`` consultation policies). It is ``0`` when every step's choice is
    forced and grows as the model spreads probability across the grammar-valid alternatives.
    """
    grammar = grammar or DistDeltaGrammar(vocab)
    model.eval()
    device = next(model.parameters()).device
    block_size = model.config.block_size

    seq = list(prompt_ids)
    generated: list[int] = []
    stack = grammar.start()
    edits = 0
    entropy_sum = 0.0
    event_append_id = vocab.id("<event_append>")
    is_client = action.name in _CLIENT_OPS

    while not grammar.is_accept(stack):
        if len(generated) >= max_new_tokens:
            raise RuntimeError("dist constrained_decode exceeded max_new_tokens")
        top = stack[0]

        window = seq[-block_size:]
        logits = model(torch.tensor([window], dtype=torch.long, device=device))[0, -1]

        allowed = grammar.allowed(stack)
        if top == "DELTA" and edits >= max_edits:
            allowed = frozenset({vocab.eos})
        elif top == "DELTA" and not is_client:
            allowed = allowed - {event_append_id}  # only client ops log a causal event (§5.1)

        mask = torch.full((len(vocab),), float("-inf"), device=device)
        mask[list(allowed)] = 0.0
        masked = logits + mask
        token = int(torch.argmax(masked).item())

        probs = torch.softmax(masked, dim=-1)
        entropy_sum += float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())

        if top == "DELTA" and token in vocab.op_ids:
            edits += 1
        stack = grammar.advance(stack, token)
        seq.append(token)
        generated.append(token)

    delta = parse_target(generated, vocab, state, action)
    return delta, entropy_sum / max(1, len(generated))


def constrained_decode(
    model: GPT,
    prompt_ids: list[int],
    vocab: DistVocab,
    state: DistributedState,
    action: DistAction,
    grammar: DistDeltaGrammar | None = None,
    *,
    max_edits: int = 64,
    max_new_tokens: int = 4096,
) -> DistDelta:
    """Greedily decode the log/replica delta for ``prompt_ids`` at ``(state, action)``.

    Termination is guaranteed: the top-level edit count is capped at ``max_edits`` (forcing
    ``<eos>``, which is always grammar-valid there). ``max_new_tokens`` is a hard backstop that
    should never trigger.
    """
    delta, _ = _decode_core(
        model, prompt_ids, vocab, state, action, grammar,
        max_edits=max_edits, max_new_tokens=max_new_tokens,
    )
    return delta


def constrained_decode_with_uncertainty(
    model: GPT,
    prompt_ids: list[int],
    vocab: DistVocab,
    state: DistributedState,
    action: DistAction,
    grammar: DistDeltaGrammar | None = None,
    *,
    max_edits: int = 64,
    max_new_tokens: int = 4096,
) -> tuple[DistDelta, float]:
    """Like :func:`constrained_decode` but also returns the mean decode entropy.

    The entropy is the per-step uncertainty signal the DS5 loop's uncertainty/drift consultation
    policies will threshold (SPEC-7 §8.1, §9.4), the distributed analogue of v0's M_θ uncertainty.
    """
    return _decode_core(
        model, prompt_ids, vocab, state, action, grammar,
        max_edits=max_edits, max_new_tokens=max_new_tokens,
    )
