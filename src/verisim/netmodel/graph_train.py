"""Supervised training for the GNN+RSSM graph arm + the §6.3 noise-injection / self-forcing levers.

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

The second lever is **self-forcing / scheduled sampling** (:func:`train_graph_model_self_forced`):
where noise injection corrupts the input *randomly*, self-forcing rolls the model out on its *own*
predictions during training and relabels each visited state with the oracle — broadening the input
distribution toward the model's *actual* deploy (drift) distribution, not a random proxy of it. Both
attack the same exposure-bias gap from different angles, and both are on/off EN4 levers (§6.3).
"""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor, nn

from verisim.net.action import NetAction, parse_net_action
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState, link_key
from verisim.netdata.drivers import NetDriver
from verisim.netdelta.apply import apply
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
    warmup_frac: float = 0.0,
) -> list[float]:
    """Minibatch teacher-forced training of the graph arm; return per-step losses.

    ``warmup_frac`` is **opt-in and defaults to 0.0** (a flat LR — the original behaviour, so every
    committed caller is byte-identical). When ``> 0`` it enables the same **linear warmup + cosine
    decay** schedule :func:`verisim.train.supervised.train_batched` uses for the flat arm (SPEC-2.1
    §6) — the lever HS3-T (SPEC-10 §4.11) uses to ask whether the graph arm's `p` plateau is a
    flat-LR trainer artifact rather than an architectural ceiling.
    """
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr)
    n = len(examples)
    use_schedule = warmup_frac > 0.0
    warmup = max(1, int(warmup_frac * steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    losses: list[float] = []
    perm: list[int] = []
    cursor = 0
    for step in range(steps):
        if cursor + batch_size > n or not perm:
            perm = torch.randperm(n, generator=gen).tolist()
            cursor = 0
        batch = [examples[i] for i in perm[cursor : cursor + batch_size]]
        cursor += batch_size
        if use_schedule:  # off by default -> flat LR -> existing results unchanged
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
        model.net.train()
        optimizer.zero_grad()
        loss = _batch_loss(model, batch, sample=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def online_update(
    model: GraphRSSMWorldModel,
    optimizer: torch.optim.Optimizer,
    examples: list[GraphExample],
    *,
    steps: int = 1,
) -> float:
    """Take ``steps`` teacher-forced gradient steps on ``examples`` — the self-healing update (H7).

    The test-time-training primitive EN5 consumes: when the loop consults the oracle mid-rollout,
    the revealed ``(state, action)`` -> true-delta is a free, exactly-labeled example, so a small
    in-rollout step lets the model *adapt to the current trajectory* rather than only having its
    state corrected. The TTT discipline (SPEC-3 §6 / HW-2) is **few ``steps`` + small lr** on a
    persistent optimizer, so it nudges without catastrophic forgetting. Returns the final loss.
    """
    model.net.train()
    last = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _batch_loss(model, examples, sample=True)
        loss.backward()
        optimizer.step()
        last = float(loss.item())
    return last


# --- the self-forcing / scheduled-sampling lever (§6.3) ---------------------


def build_self_forced_examples(
    model: GraphRSSMWorldModel,
    oracle: NetOracle,
    vocab: NetVocab,
    config: NetConfig,
    *,
    driver: str,
    seeds: tuple[int, ...],
    n_steps: int,
    sample_prob: float,
    rng: random.Random,
) -> list[GraphExample]:
    """Roll the *current model* forward on its own predictions, oracle-relabeling each step.

    Scheduled sampling (§6.3): at each step, with probability ``sample_prob`` the trajectory
    advances on the **model's own predicted** next state (the off-distribution state a free-running
    rollout actually lands in), otherwise on the true next state. Either way the training target is
    the oracle's **exact** delta for the *visited* state — the SPEC-2.1 §5 "free infinite teacher"
    again, here closing the train/deploy exposure-bias gap that pure teacher forcing hides. Where
    the noise lever corrupts the input *randomly*, self-forcing corrupts it with the model's *own*
    errors, which is precisely the deploy distribution.
    """
    examples: list[GraphExample] = []
    for seed in seeds:
        driver_obj = NetDriver(name=driver, config=config, rng=random.Random(seed))
        state = NetworkState.initial(config.hosts)
        for _ in range(n_steps):
            action = driver_obj.sample(state)
            true_result = oracle.step(state, action)
            examples.append(
                (build_graph(state, action, config), encode_target(true_result.delta, vocab))
            )
            if rng.random() < sample_prob:
                state = apply(state, model.predict_delta(state, action))  # the model's own drift
            else:
                state = true_result.state
    return examples


def train_graph_model_self_forced(
    model: GraphRSSMWorldModel,
    oracle: NetOracle,
    vocab: NetVocab,
    config: NetConfig,
    *,
    driver: str = "weighted",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 40,
    steps: int = 800,
    refresh_every: int = 200,
    max_sample_prob: float = 0.5,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Scheduled-sampling trainer (§6.3): one optimizer, dataset refreshed from the current model.

    Every ``refresh_every`` steps the self-forced dataset is regenerated with a **ramping**
    ``sample_prob`` (``0`` → ``max_sample_prob`` linearly over training), so early training is
    teacher-forced and late training matches the model's own drift distribution. A single optimizer
    spans refreshes (unlike calling :func:`train_graph_model` repeatedly, which would reset Adam).
    """
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr)
    roll_rng = random.Random(seed)
    losses: list[float] = []
    examples: list[GraphExample] = []
    perm: list[int] = []
    cursor = 0
    for step in range(steps):
        if step % refresh_every == 0:
            sample_prob = max_sample_prob * (step / steps) if steps else 0.0
            examples = build_self_forced_examples(
                model, oracle, vocab, config, driver=driver, seeds=seeds,
                n_steps=n_steps, sample_prob=sample_prob, rng=roll_rng,
            )
            perm = []
            cursor = 0
        if cursor + batch_size > len(examples) or not perm:
            perm = torch.randperm(len(examples), generator=gen).tolist()
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


# --- the multi-step unrolled-loss lever (RS4, SPEC-16 §5; the pushforward made exact) ----


def build_unrolled_examples(
    model: GraphRSSMWorldModel,
    oracle: NetOracle,
    vocab: NetVocab,
    config: NetConfig,
    *,
    driver: str,
    seeds: tuple[int, ...],
    n_steps: int,
    unroll_k: int,
) -> list[GraphExample]:
    """The pushforward trick (Brandstetter et al., ICLR 2022), made exact by the free total oracle.

    Brandstetter's pushforward poses rollout stability as domain adaptation: unroll the model a few
    steps on its *own* predictions so it is supervised on the drifted states it actually visits at
    deploy, not only the oracle's true states. Its load-bearing approximation is that the *target*
    at a drifted state is unknown, so the unrolled steps run stop-gradient and only the final step
    is supervised (against the true sequence). Here the oracle removes that approximation entirely:
    the exact delta ``O(s̃, a)`` is known at *every* drifted state, so we supervise **every**
    unrolled step against the oracle's exact label there.

    ``unroll_k`` is the pushforward depth. The drifted state is re-anchored to the true trajectory
    at the start of each ``unroll_k``-length window, then advanced on the model's own predictions
    within the window (the off-distribution states a free-running rollout lands in). ``unroll_k=1``
    re-anchors every step, so the drifted state never leaves the true trajectory and the trainer
    reduces to teacher forcing byte-for-byte; larger ``unroll_k`` supervises the model deeper into
    its own compounding drift. The action sequence is sampled on the **true** trajectory (matching
    the teacher-forced and self-forced arms), so the only thing ``unroll_k`` varies is how far the
    supervised state has drifted. The advance is a discrete :func:`apply` of the model's argmax
    delta — no gradient flows through it (the discrete state transition is non-differentiable),
    which is why the *exact oracle label* at the drifted state makes the unrolled loss trainable.
    """
    examples: list[GraphExample] = []
    for seed in seeds:
        driver_obj = NetDriver(name=driver, config=config, rng=random.Random(seed))
        true_state = NetworkState.initial(config.hosts)
        drifted = true_state
        for t in range(n_steps):
            if t % unroll_k == 0:  # re-anchor the pushforward window to the true trajectory
                drifted = true_state
            action = driver_obj.sample(true_state)
            delta = oracle.step(drifted, action).delta  # the EXACT label at the drifted state
            examples.append((build_graph(drifted, action, config), encode_target(delta, vocab)))
            drifted = apply(drifted, model.predict_delta(drifted, action))  # advance on own drift
            true_state = oracle.step(true_state, action).state  # advance the anchor on truth
    return examples


def train_unrolled(
    model: GraphRSSMWorldModel,
    oracle: NetOracle,
    vocab: NetVocab,
    config: NetConfig,
    *,
    driver: str = "weighted",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 40,
    steps: int = 800,
    unroll_k: int = 2,
    refresh_every: int = 150,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Multi-step unrolled-loss trainer (RS4, §5): one optimizer, dataset refreshed from the model.

    Structurally identical to :func:`train_graph_model_self_forced` (a single optimizer spanning
    periodic dataset refreshes, so Adam's state is not reset), but the dataset is the
    :func:`build_unrolled_examples` pushforward of depth ``unroll_k`` rather than the
    scheduled-sampling rollout. Every ``refresh_every`` steps the drifted-state examples are
    regenerated from the *current* model, so late training is supervised on the model's *current*
    drift. Returns per-step losses.
    """
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr)
    losses: list[float] = []
    examples: list[GraphExample] = []
    perm: list[int] = []
    cursor = 0
    for step in range(steps):
        if step % refresh_every == 0:
            examples = build_unrolled_examples(
                model, oracle, vocab, config, driver=driver, seeds=seeds,
                n_steps=n_steps, unroll_k=unroll_k,
            )
            perm = []
            cursor = 0
        if cursor + batch_size > len(examples) or not perm:
            perm = torch.randperm(len(examples), generator=gen).tolist()
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
