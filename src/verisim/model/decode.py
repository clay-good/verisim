"""Constrained greedy decoding to a grammar-valid delta (SPEC-2 §5.2).

At each step the model's logits are masked to the grammar's allowed next tokens
(:class:`DeltaGrammar`), so the decoded sequence is always parseable into a valid
:class:`~verisim.delta.edits.Delta` regardless of the model's weights. A
``max_edits`` cap forces termination by restricting the top-level choice to
``<eos>`` once enough edits have been emitted (``<eos>`` is always grammar-valid
there), which keeps both the validity and the termination guarantees.
"""

from __future__ import annotations

import torch

from verisim.delta.edits import Delta

from .grammar import DeltaGrammar
from .tokenizer import parse_target
from .transformer import GPT
from .vocab import Vocab

# Repeating grammar symbols and the token that closes each. Capping these runs
# guarantees termination: a model (esp. an untrained one) could otherwise keep
# emitting path/content/stdout tokens forever without ever choosing the closer.
_CLOSE_TOKEN = {"PATH_SEG": "</p>", "CONTENT_TOK": "</c>", "STDOUT_TOK": "</o>"}


@torch.no_grad()
def _decode_core(
    model: GPT,
    prompt_ids: list[int],
    vocab: Vocab,
    grammar: DeltaGrammar | None,
    *,
    max_edits: int,
    max_run: int,
    max_new_tokens: int,
) -> tuple[Delta, str, float]:
    """Greedy constrained decode; return ``(delta, stdout, mean_entropy)``.

    ``mean_entropy`` is the mean over decoded steps of the Shannon entropy (nats) of
    the masked next-token distribution -- the model's per-prediction uncertainty
    (SPEC-2 §7.2). It is ``0`` when every step's choice is forced (a single allowed
    token, e.g. a fully overfit model on a grammar with one continuation) and grows
    as the model spreads probability across the grammar-valid alternatives.
    """
    grammar = grammar or DeltaGrammar(vocab)
    model.eval()
    device = next(model.parameters()).device
    block_size = model.config.block_size

    seq = list(prompt_ids)
    generated: list[int] = []
    stack = grammar.start()
    edits = 0
    prev_top: str | None = None
    run = 0
    entropy_sum = 0.0

    while not grammar.is_accept(stack):
        if len(generated) >= max_new_tokens:
            raise RuntimeError("constrained_decode exceeded max_new_tokens")
        top = stack[0]
        run = run + 1 if top == prev_top else 0

        window = seq[-block_size:]
        logits = model(torch.tensor([window], dtype=torch.long, device=device))[0, -1]

        allowed = grammar.allowed(stack)
        if top == "DELTA" and edits >= max_edits:
            allowed = frozenset({vocab.eos})
        elif top in _CLOSE_TOKEN and run >= max_run:
            allowed = frozenset({vocab.id(_CLOSE_TOKEN[top])})

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
        prev_top = top

    delta, stdout = parse_target(generated, vocab)
    return delta, stdout, entropy_sum / max(1, len(generated))


def constrained_decode(
    model: GPT,
    prompt_ids: list[int],
    vocab: Vocab,
    grammar: DeltaGrammar | None = None,
    *,
    max_edits: int = 64,
    max_run: int = 256,
    max_new_tokens: int = 4096,
) -> tuple[Delta, str]:
    """Greedily decode the delta for ``prompt_ids``; return ``(delta, stdout)``.

    Termination is guaranteed: the top-level edit count is capped at ``max_edits``
    (forcing ``<eos>``) and each repeating leaf run is capped at ``max_run``
    (forcing its closing token). Both forced tokens are always grammar-valid in
    their state, so the result is still a valid delta. ``max_new_tokens`` is a hard
    backstop that should never trigger.
    """
    delta, stdout, _ = _decode_core(
        model,
        prompt_ids,
        vocab,
        grammar,
        max_edits=max_edits,
        max_run=max_run,
        max_new_tokens=max_new_tokens,
    )
    return delta, stdout


def constrained_decode_with_uncertainty(
    model: GPT,
    prompt_ids: list[int],
    vocab: Vocab,
    grammar: DeltaGrammar | None = None,
    *,
    max_edits: int = 64,
    max_run: int = 256,
    max_new_tokens: int = 4096,
) -> tuple[Delta, str, float]:
    """Like :func:`constrained_decode` but also returns the mean decode entropy.

    The entropy is the per-step uncertainty signal that the ``uncertainty``/``drift``
    consultation policies threshold (SPEC-2 §6.1, §7.2).
    """
    return _decode_core(
        model,
        prompt_ids,
        vocab,
        grammar,
        max_edits=max_edits,
        max_run=max_run,
        max_new_tokens=max_new_tokens,
    )
