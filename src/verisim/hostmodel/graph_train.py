"""Supervised training for the factored interaction-graph arm (SPEC-6 §6.1, HC4 increment 2).

The factored arm's forward (graph -> RSSM -> conditioned decoder) differs from the flat arm's single
concatenated GPT sequence, so it gets a focused trainer here rather than reusing
:mod:`verisim.train.supervised` (which is GPT-shaped). The objective is the same: teacher-forced
cross-entropy over the bundle-delta tokens.

The §6.3 drift levers carry over from the network arm and ship here (the host's "required ablation
levers", §6.3). Both attack the train/deploy exposure-bias gap that pure teacher forcing hides, and
both exploit the oracle being **total and free**: a corrupted (off-trajectory) state is **relabeled
exactly** -- we compute ``O(s̃, a)`` and train on ``(s̃, a, O(s̃, a))``.

  - **noise injection** (the GNS lesson, the cheapest highest-leverage one): with probability
    ``noise_prob`` a training state is replaced by a one-mutation corruption and oracle-relabeled,
    broadening the input distribution toward where free-running drift lands -- random off-trajectory
    states, but with perfectly correct targets.
  - **self-forcing / scheduled sampling**: roll the *current model* forward on its **own**
    predictions and oracle-relabel each visited state, broadening toward the model's *actual* deploy
    (drift) distribution rather than a random proxy of it.

Each is an on/off lever measured by ``eh4_drift`` (the §6.3 / EH4 drift-ablation axis); whether
either buys free-running horizon at this scale is the honest measurement (the network found a banked
negative -- a small one-step dip, no horizon yet).
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from verisim.host.config import HostConfig
from verisim.host.delta import apply as apply_host_delta
from verisim.host.state import RUNNING, ZOMBIE, FdEntry, HostState, Process
from verisim.hostdata.drivers import HostDriver
from verisim.hostoracle.base import HostOracle
from verisim.train.dataset import IGNORE_INDEX

from .graph import HostGraph, build_host_graph
from .graph_model import GraphHostWorldModel, graphs_to_tensors
from .tokenizer import encode_target
from .vocab import HostVocab

GraphExample = tuple[HostGraph, list[int]]  # (featurized state/action, target delta tokens)


# --- the noise-injection lever (§6.3) ---------------------------------------


def corrupt_host_state(state: HostState, config: HostConfig, rng: random.Random) -> HostState:
    """Apply one random off-trajectory mutation to the process/fd subsystems (the GNS perturbation).

    The result is a *valid* :class:`HostState` one edit from ``state``; the dataset builder then
    relabels it exactly with the oracle. Mutations target ``proc``/``fd`` -- the compounding
    subsystems EH4 showed the model struggles with -- and never touch pid 1 (keep a live init).
    """
    procs = dict(state.procs)
    fds = dict(state.fds)
    running = [p for p in procs if procs[p].state == RUNNING and p != 1]
    kind = rng.choice(["proc_flip", "uid", "fd_add", "fd_remove"])
    if kind == "proc_flip" and running:
        pid = rng.choice(running)
        p = procs[pid]
        procs[pid] = Process(p.pid, p.ppid, ZOMBIE, p.uid, rng.choice((0, 1)))
        fds = {k: v for k, v in fds.items() if k[0] != pid}  # exit releases the fds
    elif kind == "uid":
        pid = rng.choice(sorted(procs))
        p = procs[pid]
        procs[pid] = Process(p.pid, p.ppid, p.state, rng.choice(config.uids), p.exit_code)
    elif kind == "fd_add":
        pid = rng.choice(sorted(p for p in procs if procs[p].state == RUNNING))
        used = {fd for (q, fd) in fds if q == pid}
        fd = next(i for i in range(len(used) + 1) if i not in used)
        fds[(pid, fd)] = FdEntry(path=rng.choice(config.paths))
    elif fds:  # fd_remove
        del fds[rng.choice(sorted(fds))]
    return HostState(procs=procs, fds=fds, fs=state.fs.copy(), next_pid=state.next_pid,
                     last_exit=state.last_exit)


def build_host_graph_dataset(
    oracle: HostOracle,
    vocab: HostVocab,
    config: HostConfig,
    *,
    driver: str = "forky",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 40,
    noise_prob: float = 0.0,
    noise_seed: int = 0,
) -> list[GraphExample]:
    """``(HostGraph, target_ids)`` per step of seeded rollouts (the factored supervised dataset).

    With probability ``noise_prob`` a step's state is replaced by a one-mutation corruption and the
    target is **relabeled by the oracle** for the corrupted state (§6.3 noise lever). The clean
    trajectory still advances on the *true* next state, so noise augments coverage without derailing
    the rollout.
    """
    noise_rng = random.Random(noise_seed)
    examples: list[GraphExample] = []
    for seed in seeds:
        driver_obj = HostDriver(name=driver, config=config, rng=random.Random(seed))
        state = HostState.initial()
        for _ in range(n_steps):
            action = driver_obj.sample(state)
            true_result = oracle.step(state, action)
            if noise_prob > 0.0 and noise_rng.random() < noise_prob:
                noisy = corrupt_host_state(state, config, noise_rng)
                delta = oracle.step(noisy, action).delta
                examples.append(
                    (build_host_graph(noisy, action, config, vocab.max_pid),
                     encode_target(delta, vocab))
                )
            else:
                examples.append(
                    (build_host_graph(state, action, config, vocab.max_pid),
                     encode_target(true_result.delta, vocab))
                )
            state = true_result.state  # advance on the TRUE next state
    return examples


def build_self_forced_host_examples(
    model: GraphHostWorldModel,
    oracle: HostOracle,
    vocab: HostVocab,
    config: HostConfig,
    *,
    driver: str,
    seeds: tuple[int, ...],
    n_steps: int,
    sample_prob: float,
    rng: random.Random,
) -> list[GraphExample]:
    """Roll the *current model* forward on its own predictions, oracle-relabeling each step (§6.3).

    Scheduled sampling: at each step, with probability ``sample_prob`` the trajectory advances on
    the **model's own predicted** next state (the off-distribution state a free-running rollout
    lands in), else on the true next state. Either way the training target is the oracle's **exact**
    bundle delta for the *visited* state -- the free-infinite-teacher again, here closing the
    train/deploy exposure-bias gap. Where the noise lever corrupts the input randomly, self-forcing
    corrupts it with the model's *own* errors, which is precisely the deploy distribution.
    """
    examples: list[GraphExample] = []
    for seed in seeds:
        driver_obj = HostDriver(name=driver, config=config, rng=random.Random(seed))
        state = HostState.initial()
        for _ in range(n_steps):
            action = driver_obj.sample(state)
            true_result = oracle.step(state, action)
            examples.append(
                (build_host_graph(state, action, config, vocab.max_pid),
                 encode_target(true_result.delta, vocab))
            )
            if sample_prob > 0.0 and rng.random() < sample_prob:
                state = apply_host_delta(state, model.predict_delta(state, action))
            else:
                state = true_result.state
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


def train_host_graph_model_self_forced(
    model: GraphHostWorldModel,
    oracle: HostOracle,
    vocab: HostVocab,
    config: HostConfig,
    *,
    driver: str = "forky",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 40,
    rounds: int = 4,
    steps_per_round: int = 200,
    sample_prob: float = 0.5,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Self-forcing trainer (§6.3): rebuild the rollout from the *current* model each round.

    A warmup round on clean teacher-forced data gives the model a usable proposer; each later round
    re-rolls on the model's own (drifting) predictions and oracle-relabels, so the input
    distribution tracks the model as it improves -- scheduled sampling with a free, exact teacher.
    Returns the
    concatenated per-step losses across rounds.
    """
    forcing_rng = random.Random(seed)
    losses: list[float] = []
    for r in range(rounds):
        if r == 0:
            examples = build_host_graph_dataset(
                oracle, vocab, config, driver=driver, seeds=seeds, n_steps=n_steps
            )
        else:
            examples = build_self_forced_host_examples(
                model, oracle, vocab, config, driver=driver, seeds=seeds, n_steps=n_steps,
                sample_prob=sample_prob, rng=forcing_rng,
            )
        losses += train_host_graph_model(
            model, examples, steps=steps_per_round, lr=lr, batch_size=batch_size, seed=seed + r
        )
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
