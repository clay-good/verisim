"""Supervised training for the GNN+RSSM graph arm, with the §6.3 noise-injection drift lever.

The graph arm's forward (graph → RSSM → conditioned decoder) differs from the flat arm's single
concatenated GPT sequence, so it gets a focused trainer here rather than reusing
:mod:`verisim.train.supervised` (which is GPT-shaped). The objective is the same: teacher-forced
cross-entropy over the delta tokens.

The one addition is the **noise-injection lever** (SPEC-5 §6.3, the GNS lesson — the single
highest-leverage, cheapest drift mitigation). Classic GNS corrupts the input so the *training*
input distribution matches the noisy distribution a model eats during oracle-free rollout. This
domain can do something stronger than classic GNS: because the oracle is *total and free*, a
corrupted (off-trajectory) state can be **relabeled exactly** — we compute ``O(s̃, a)`` and train on
``(s̃, a, O(s̃, a))``. So the lever is *oracle-relabeled state-noise augmentation*: broaden the input
distribution toward where rollout drift lands, with perfectly correct targets (the SPEC-2.1 §5 "free
infinite teacher" realized as a drift mitigation). ``noise_prob`` is the key hyperparameter (§6.3):
too much hurts one-step accuracy, too little fails to cover drift — an EN4 lever, not a default.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from verisim.net.action import NetAction, parse_net_action
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState, link_key
from verisim.netdata.drivers import NetDriver
from verisim.netoracle.base import NetOracle
from verisim.train.dataset import IGNORE_INDEX

from .graph import NetGraph, build_graph
from .graph_model import GraphRSSMWorldModel, graphs_to_tensors
from .tokenizer import encode_target
from .vocab import NetVocab

GraphExample = tuple[NetGraph, list[int]]  # (featurized state/action, target delta tokens)


# --- the noise-injection lever (§6.3) ---------------------------------------


def corrupt_state(state: NetworkState, config: NetConfig, rng: random.Random) -> NetworkState:
    """Apply one random off-trajectory mutation (the GNS perturbation, then oracle-relabeled)."""
    s = state.copy()
    hosts = config.hosts
    kind = rng.choice(["host", "svc", "fw", "link", "flow"])
    h = rng.choice(hosts)
    if kind == "host":
        s.hosts[h] = s.hosts[h].with_up(not s.hosts[h].up)
    elif kind == "svc":
        port = rng.choice(config.ports)
        s.hosts[h] = s.hosts[h].with_service(port, port not in s.hosts[h].services)
    elif kind == "fw":
        src = rng.choice(hosts)
        s.hosts[h] = s.hosts[h].with_fw(src, src not in s.hosts[h].fw_deny)
    elif kind == "link":
        other = rng.choice([x for x in hosts if x != h])
        key = link_key(h, other)
        s.links.discard(key) if key in s.links else s.links.add(key)
    else:  # flow
        dst = rng.choice([x for x in hosts if x != h])
        port = rng.choice(config.ports)
        flow = (h, dst, port)
        s.flows.discard(flow) if flow in s.flows else s.flows.add(flow)
    return s


# --- dataset ----------------------------------------------------------------


def build_graph_dataset(
    oracle: NetOracle,
    vocab: NetVocab,
    config: NetConfig,
    *,
    driver: str = "weighted",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 40,
    noise_prob: float = 0.0,
    noise_seed: int = 0,
) -> list[GraphExample]:
    """``(NetGraph, target_ids)`` per step of seeded rollouts, with optional noise injection.

    With probability ``noise_prob`` a step's state is replaced by a one-mutation corruption and the
    target is **relabeled by the oracle** for the corrupted state (§6.3). The clean trajectory still
    advances on the *true* next state, so noise augments coverage without derailing the rollout.
    """
    noise_rng = random.Random(noise_seed)
    examples: list[GraphExample] = []
    for seed in seeds:
        driver_obj = NetDriver(name=driver, config=config, rng=random.Random(seed))
        state = NetworkState.initial(config.hosts)
        for _ in range(n_steps):
            action = driver_obj.sample(state)
            true_result = oracle.step(state, action)
            if noise_prob > 0.0 and noise_rng.random() < noise_prob:
                noisy = corrupt_state(state, config, noise_rng)
                delta = oracle.step(noisy, action).delta
                examples.append((build_graph(noisy, action, config), encode_target(delta, vocab)))
            else:
                examples.append(
                    (build_graph(state, action, config), encode_target(true_result.delta, vocab))
                )
            state = true_result.state  # advance on the TRUE next state
    return examples


# --- training ---------------------------------------------------------------


def _collate(
    batch: list[GraphExample], vocab: NetVocab, device: torch.device
) -> tuple[list[NetGraph], Tensor, Tensor]:
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


def _batch_loss(
    model: GraphRSSMWorldModel, batch: list[GraphExample], *, sample: bool
) -> Tensor:
    device = model.net.device
    graphs, inp, lab = _collate(batch, model.vocab, device)
    node, gfeat, a_link, a_flow = graphs_to_tensors(graphs, device)
    cond, _belief_var = model.net.encode(node, gfeat, a_link, a_flow, sample=sample)
    logits = model.net.decode_logits(cond, inp)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), lab.reshape(-1), ignore_index=IGNORE_INDEX
    )


def train_graph_model(
    model: GraphRSSMWorldModel,
    examples: list[GraphExample],
    *,
    steps: int = 800,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Minibatch teacher-forced training of the graph arm; return per-step losses."""
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
    model: GraphRSSMWorldModel, examples: list[GraphExample]
) -> float:
    """Fraction of target tokens correct under teacher forcing (the K0 learner check)."""
    model.net.eval()
    device = model.net.device
    graphs, inp, lab = _collate(examples, model.vocab, device)
    node, gfeat, a_link, a_flow = graphs_to_tensors(graphs, device)
    cond, _belief_var = model.net.encode(node, gfeat, a_link, a_flow, sample=False)
    preds = model.net.decode_logits(cond, inp).argmax(dim=-1)
    mask = lab != IGNORE_INDEX
    correct = int(((preds == lab) & mask).sum().item())
    total = int(mask.sum().item())
    return correct / total if total else 1.0


def parse_actions(cmds: list[str]) -> list[NetAction]:
    """Convenience for tests/experiments: parse a list of command strings."""
    return [parse_net_action(c) for c in cmds]
