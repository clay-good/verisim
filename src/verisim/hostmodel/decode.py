"""Constrained greedy decoding to a grammar-valid bundle delta (SPEC-6 §6.1, HC4).

The host analogue of v0's :mod:`verisim.model.decode` and the network
:mod:`verisim.netmodel.decode`. At each step the model's logits are masked to the grammar's allowed
next tokens (:class:`HostDeltaGrammar`), so the decoded sequence is always parseable into a valid
:class:`~verisim.host.delta.HostDelta` regardless of the model's weights. A ``max_edits`` cap forces
termination by restricting the top-level choice to ``<eos>`` once enough edits have been emitted
(``<eos>`` is always grammar-valid there). Host bundle deltas have no repeating leaf runs, so -- as
in the net arm and unlike v0's filesystem deltas -- no per-run cap is needed. The flat HC4 arm
reuses v0's :class:`~verisim.model.transformer.GPT` directly.
"""

from __future__ import annotations

import torch

from verisim.host.delta import HostDelta
from verisim.model.transformer import GPT

from .grammar import HostDeltaGrammar
from .tokenizer import parse_target
from .vocab import HostVocab


@torch.no_grad()
def _decode_core(
    model: GPT,
    prompt_ids: list[int],
    vocab: HostVocab,
    grammar: HostDeltaGrammar | None,
    *,
    max_edits: int,
    max_new_tokens: int,
) -> tuple[HostDelta, float]:
    """Greedy constrained decode; return ``(delta, mean_entropy)``.

    ``mean_entropy`` is the mean over decoded steps of the Shannon entropy (nats) of the masked
    next-token distribution -- the model's per-prediction uncertainty (SPEC-6 §5.4 / §8.1, the
    richer per-subsystem signal the HC5 loop will threshold). It is ``0`` when every step's choice
    is forced and grows as the model spreads probability across the grammar-valid alternatives.
    """
    grammar = grammar or HostDeltaGrammar(vocab)
    model.eval()
    device = next(model.parameters()).device
    block_size = model.config.block_size

    seq = list(prompt_ids)
    generated: list[int] = []
    stack = grammar.start()
    edits = 0
    entropy_sum = 0.0

    while not grammar.is_accept(stack):
        if len(generated) >= max_new_tokens:
            raise RuntimeError("host constrained_decode exceeded max_new_tokens")
        top = stack[0]

        window = seq[-block_size:]
        logits = model(torch.tensor([window], dtype=torch.long, device=device))[0, -1]

        allowed = grammar.allowed(stack)
        if top == "DELTA" and edits >= max_edits:
            allowed = frozenset({vocab.eos})

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

    delta = parse_target(generated, vocab)
    return delta, entropy_sum / max(1, len(generated))


def constrained_decode(
    model: GPT,
    prompt_ids: list[int],
    vocab: HostVocab,
    grammar: HostDeltaGrammar | None = None,
    *,
    max_edits: int = 64,
    max_new_tokens: int = 4096,
) -> HostDelta:
    """Greedily decode the bundle delta for ``prompt_ids``.

    Termination is guaranteed: the top-level edit count is capped at ``max_edits`` (forcing
    ``<eos>``, which is always grammar-valid there). ``max_new_tokens`` is a hard backstop that
    should never trigger.
    """
    delta, _ = _decode_core(
        model, prompt_ids, vocab, grammar, max_edits=max_edits, max_new_tokens=max_new_tokens
    )
    return delta


def constrained_decode_with_uncertainty(
    model: GPT,
    prompt_ids: list[int],
    vocab: HostVocab,
    grammar: HostDeltaGrammar | None = None,
    *,
    max_edits: int = 64,
    max_new_tokens: int = 4096,
) -> tuple[HostDelta, float]:
    """Like :func:`constrained_decode` but also returns the mean decode entropy.

    The entropy is the per-step uncertainty signal the HC5 loop's drift/uncertainty consultation
    policies will threshold (SPEC-6 §8.1), the host analogue of v0's M_θ uncertainty signal.
    """
    return _decode_core(
        model, prompt_ids, vocab, grammar, max_edits=max_edits, max_new_tokens=max_new_tokens
    )
