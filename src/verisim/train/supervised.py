"""Stage-1 supervised next-delta training (SPEC-2 §5.3).

Teacher-forced cross-entropy over the serialized delta tokens, on oracle-generated
examples. v0 uses full-batch gradient descent over a small example set -- enough to
fit a tiny env to near-zero loss (the M4 verify) and deterministic given a seed.
Stage 2 (RLVR against the oracle reward) lives in :mod:`verisim.train.rlvr`.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from ..model.transformer import GPT
from .dataset import IGNORE_INDEX, Example, collate


def train_supervised(
    model: GPT,
    examples: list[Example],
    pad_id: int,
    *,
    steps: int = 300,
    lr: float = 3e-3,
    seed: int = 0,
) -> list[float]:
    """Train ``model`` on ``examples`` (full-batch); return the per-step losses."""
    torch.manual_seed(seed)
    inputs, labels = collate(examples, pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def _batch_loss(model: GPT, batch: list[Example], pad_id: int) -> Tensor:
    inputs, labels = collate(batch, pad_id)
    logits = model(inputs)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def _mean_loss(model: GPT, examples: list[Example], pad_id: int, batch_size: int) -> float:
    """Mean cross-entropy over ``examples`` (for validation / early stopping)."""
    model.eval()
    total = 0.0
    n_batches = 0
    for start in range(0, len(examples), batch_size):
        total += float(_batch_loss(model, examples[start : start + batch_size], pad_id).item())
        n_batches += 1
    return total / n_batches if n_batches else 0.0


def train_batched(
    model: GPT,
    examples: list[Example],
    pad_id: int,
    *,
    steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int = 64,
    seed: int = 0,
    warmup_frac: float = 0.1,
    val_examples: list[Example] | None = None,
    eval_interval: int = 0,
) -> list[float]:
    """Minibatch SGD with warmup+cosine LR decay and optional val early-stopping (SPEC-2.1 §6).

    This is the K2 "train properly" loop: minibatching lets the K1 coverage dataset (thousands
    of transitions) be trained without the full-batch memory/throughput wall that
    :func:`train_supervised` hits, the LR schedule converges where a flat LR stalls, and
    ``val_examples`` + ``eval_interval`` keep the *best-generalizing* checkpoint (not the most
    overfit one). Deterministic given ``seed``. ``train_supervised`` is left untouched as the
    full-batch path the M4 overfit test pins.
    """
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(examples)
    warmup = max(1, int(warmup_frac * steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    losses: list[float] = []
    best_val = math.inf
    best_state: dict[str, Tensor] | None = None
    perm: list[int] = []
    cursor = 0
    for step in range(steps):
        if cursor + batch_size > n or not perm:
            perm = torch.randperm(n, generator=gen).tolist()
            cursor = 0
        batch = [examples[i] for i in perm[cursor : cursor + batch_size]]
        cursor += batch_size
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        model.train()
        optimizer.zero_grad()
        loss = _batch_loss(model, batch, pad_id)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if val_examples and eval_interval and (step + 1) % eval_interval == 0:
            val = _mean_loss(model, val_examples, pad_id, batch_size)
            if val < best_val:
                best_val = val
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses


@torch.no_grad()
def teacher_forced_accuracy(model: GPT, examples: list[Example], pad_id: int) -> float:
    """Fraction of target tokens predicted correctly under teacher forcing."""
    inputs, labels = collate(examples, pad_id)
    model.eval()
    preds: Tensor = model(inputs).argmax(dim=-1)
    mask = labels != IGNORE_INDEX
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item()
    return float(correct) / float(total) if total else 1.0
