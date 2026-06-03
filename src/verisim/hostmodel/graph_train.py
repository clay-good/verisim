"""Supervised training for the factored interaction-graph arm (SPEC-6 §6.1, HC4 increment 2).

The factored arm's forward (graph -> RSSM -> conditioned decoder) differs from the flat arm's single
concatenated GPT sequence, so it gets a focused trainer here rather than reusing
:mod:`verisim.train.supervised` (which is GPT-shaped). The objective is the same: teacher-forced
cross-entropy over the bundle-delta tokens.

The §6.3 drift levers (oracle-relabeled noise injection, self-forcing) carry over from the network
arm but are deferred to a later HC increment -- this is the clean supervised baseline the EH4
flat-vs-factored comparison needs first (the DD-4 "measure before scaling" discipline).
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from verisim.host.config import HostConfig
from verisim.host.state import HostState
from verisim.hostdata.drivers import HostDriver
from verisim.hostoracle.base import HostOracle
from verisim.train.dataset import IGNORE_INDEX

from .graph import HostGraph, build_host_graph
from .graph_model import GraphHostWorldModel, graphs_to_tensors
from .tokenizer import encode_target
from .vocab import HostVocab

GraphExample = tuple[HostGraph, list[int]]  # (featurized state/action, target delta tokens)


def build_host_graph_dataset(
    oracle: HostOracle,
    vocab: HostVocab,
    config: HostConfig,
    *,
    driver: str = "forky",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 40,
) -> list[GraphExample]:
    """``(HostGraph, target_ids)`` per step of seeded rollouts (the factored supervised dataset)."""
    examples: list[GraphExample] = []
    for seed in seeds:
        driver_obj = HostDriver(name=driver, config=config, rng=random.Random(seed))
        state = HostState.initial()
        for _ in range(n_steps):
            action = driver_obj.sample(state)
            result = oracle.step(state, action)
            examples.append(
                (
                    build_host_graph(state, action, config, vocab.max_pid),
                    encode_target(result.delta, vocab),
                )
            )
            state = result.state
    return examples


def _collate(
    batch: list[GraphExample], vocab: HostVocab, device: torch.device
) -> tuple[list[HostGraph], Tensor, Tensor]:
    """Pad a batch to ``(graphs, input_ids[B,T], labels[B,T])`` for teacher forcing."""
    graphs = [g for g, _ in batch]
    targets = [t for _, t in batch]
    width = max(len(t) for t in targets)  # input = [gen]+target[:-1], same length as target
    inp = torch.full((len(batch), width), vocab.pad, dtype=torch.long, device=device)
    lab = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long, device=device)
    for k, target in enumerate(targets):
        seq = [vocab.gen, *target[:-1]]
        inp[k, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        lab[k, : len(target)] = torch.tensor(target, dtype=torch.long, device=device)
    return graphs, inp, lab


def _batch_loss(model: GraphHostWorldModel, batch: list[GraphExample], *, sample: bool) -> Tensor:
    device = model.net.device
    graphs, inp, lab = _collate(batch, model.vocab, device)
    node, gfeat, mask, a_lin, a_share, acting = graphs_to_tensors(graphs, device)
    cond, _belief = model.net.encode(node, gfeat, mask, a_lin, a_share, acting, sample=sample)
    logits = model.net.decode_logits(cond, inp)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), lab.reshape(-1), ignore_index=IGNORE_INDEX
    )


def train_host_graph_model(
    model: GraphHostWorldModel,
    examples: list[GraphExample],
    *,
    steps: int = 800,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Minibatch teacher-forced training of the factored arm; return per-step losses."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr)
    n = len(examples)
    losses: list[float] = []
    perm: list[int] = []
    cursor = 0
    for _ in range(steps):
        if cursor + batch_size > n or not perm:
            perm = torch.randperm(n, generator=gen).tolist()
            cursor = 0
        batch = [examples[i] for i in perm[cursor : cursor + batch_size]]
        cursor += batch_size
        model.net.train()
        optimizer.zero_grad()
        loss = _batch_loss(model, batch, sample=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def graph_teacher_forced_accuracy(
    model: GraphHostWorldModel, examples: list[GraphExample]
) -> float:
    """Fraction of target tokens correct under teacher forcing (the EH4 one-step metric)."""
    model.net.eval()
    device = model.net.device
    graphs, inp, lab = _collate(examples, model.vocab, device)
    node, gfeat, mask, a_lin, a_share, acting = graphs_to_tensors(graphs, device)
    cond, _belief = model.net.encode(node, gfeat, mask, a_lin, a_share, acting, sample=False)
    preds = model.net.decode_logits(cond, inp).argmax(dim=-1)
    valid = lab != IGNORE_INDEX
    correct = ((preds == lab) & valid).sum().item()
    total = valid.sum().item()
    return float(correct) / float(total) if total else 1.0
