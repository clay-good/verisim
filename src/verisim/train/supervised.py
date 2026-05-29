"""Stage-1 supervised next-delta training (SPEC-2 §5.3).

Teacher-forced cross-entropy over the serialized delta tokens, on oracle-generated
examples. v0 uses full-batch gradient descent over a small example set -- enough to
fit a tiny env to near-zero loss (the M4 verify) and deterministic given a seed.
Stage 2 (RLVR against the oracle reward) is deferred to a later milestone.
"""

from __future__ import annotations

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
