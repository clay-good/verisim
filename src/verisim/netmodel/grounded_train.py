"""Oracle-grounded SSL training for the NW8 graph arm — the EN8 apparatus (SPEC-8 §4, §7; OG3).

OG1/OG2 (:mod:`verisim.netdata.grounding`, :mod:`verisim.netdata.negatives`) shipped the
*deterministic* target machinery — the decidable/residual partition and the exact
hard-negative factory — ahead of any GPU. This module is the **trainer** that consumes it on the
graph+RSSM latent arm, i.e. the thing EN8 ablates. It instantiates the two SPEC-8 axes:

  - **H24 — the objective axis** (:func:`train_grounded_decoder`): cross-entropy over *all* delta
    tokens (raw-likelihood, today's supervised baseline) versus cross-entropy over the **residual**
    tokens only — the bits the oracle does *not* fix given the observation (SPEC-8 §4.2). Masking
    the decidable edits with :data:`IGNORE_INDEX` spends gradient only on ``R`` (*even nature
    offloads* the decidable part). Non-degenerate only under partial observation, so the dataset
    carries a per-token residual mask tied to an observed-host set (the NW5 probe model).

  - **H23 — the collapse axis** (:func:`train_jepa`): a JEPA-style latent predictor over the graph
    summary (:meth:`GraphRSSMNet.embed`), with the target either a **learned** encoder (BYOL/JEPA's
    EMA) or **oracle-anchored** — a fixed projection of the *true next state's* features (§4.1), an
    external referent with full variance by construction, so there is nothing for the embedding to
    collapse toward. The collapse-prevention machinery (EMA target + VICReg variance/covariance) is
    an on/off lever. The diagnostic is :func:`representation_health` (embedding variance + effective
    rank, the standard collapse readout): the H23 question is whether the oracle-anchored target
    keeps the representation healthy *with the machinery ablated*.

Per the repo's discipline the deterministic data factory (OG1/OG2) and the per-step delta-exact
metric were committed and property-tested first; this is the GPU consumer that turns them into the
EN8 figure. Everything is seeded and process-reproducible (``torch.set_num_threads(1)``), like EN1.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from verisim.net.config import NetConfig
from verisim.net.state import NetworkState
from verisim.netdelta.edits import (
    FlowClose,
    FlowOpen,
    FwAllow,
    FwDeny,
    HostDown,
    HostUp,
    LinkAdd,
    LinkDel,
    NetDelta,
    NetEdit,
    SvcDown,
    SvcUp,
)
from verisim.netoracle.base import NetOracle
from verisim.train.dataset import IGNORE_INDEX

from .graph import NetGraph, build_graph
from .graph_model import GraphRSSMNet, GraphRSSMWorldModel, graphs_to_tensors
from .tokenizer import encode_target
from .vocab import NetVocab

# --- the decidable/residual token partition (SPEC-8 §4.2, the H24 objective) -------------


def edit_hosts(edit: NetEdit) -> frozenset[str]:
    """The host id(s) an edit references — mirrors OG1's :func:`netdata.grounding.fact_hosts`.

    The global clock-advance / set-result edits reference no host (returned empty), so they are
    decidable by construction, exactly as the global facts are in :func:`net_facts`.
    """
    if isinstance(edit, (HostUp, HostDown, SvcUp, SvcDown)):
        return frozenset({edit.host})
    if isinstance(edit, (FwDeny, FwAllow)):
        return frozenset({edit.host, edit.src})
    if isinstance(edit, (LinkAdd, LinkDel)):
        return frozenset({edit.a, edit.b})
    if isinstance(edit, (FlowOpen, FlowClose)):
        return frozenset({edit.src, edit.dst})
    return frozenset()  # ClockAdvance / SetResult — global, always decidable


def edit_is_decidable(edit: NetEdit, observed_hosts: frozenset[str] | None) -> bool:
    """``True`` iff the oracle fixes ``edit`` given the observation (OG1's ``D`` regime, per-edit).

    Full observation (``observed_hosts is None``) makes every edit decidable; otherwise an edit is
    decidable iff every host it references is observed (the global clock/result edits always are).
    """
    if observed_hosts is None:
        return True
    return edit_hosts(edit) <= observed_hosts


def residual_token_mask(
    delta: NetDelta, vocab: NetVocab, observed_hosts: frozenset[str] | None
) -> list[bool]:
    """Per target-token flag: ``True`` iff the token belongs to a **residual** edit (train on it).

    Aligned to :func:`~verisim.netmodel.tokenizer.encode_target` (each edit's token span, then the
    trailing ``<eos>``). Decidable-edit tokens and the structural ``<eos>`` are ``False`` — the
    residual objective masks them to :data:`IGNORE_INDEX` so gradient lands only on ``R`` (§4.2).
    """
    mask: list[bool] = []
    for edit in delta:
        span = len(encode_target([edit], vocab)) - 1  # drop the singleton's <eos>
        mask.extend([not edit_is_decidable(edit, observed_hosts)] * span)
    mask.append(False)  # the trailing <eos>: structural, not a residual fact
    return mask


# --- dataset ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundedExample:
    """One step's oracle-grounded training datum (SPEC-8 §4.1-4.2).

    Carries everything both EN8 axes need: the featurized ``(s, a)`` graph and target delta tokens
    (the decoder), the per-token residual mask (the H24 objective), the featurized **true next**
    state graph (the learned/EMA JEPA target), and that state's flat raw features (the
    oracle-anchored JEPA target — a fixed projection of real next-state data, §4.1).
    """

    graph: NetGraph
    target_ids: tuple[int, ...]
    residual_mask: tuple[bool, ...]
    next_graph: NetGraph
    next_rawfeat: tuple[float, ...]


def _raw_features(graph: NetGraph) -> tuple[float, ...]:
    """Flatten a graph's node + graph features into one fixed-width vector (the §4.1 target)."""
    flat: list[float] = []
    for row in graph.node_features:
        flat.extend(row)
    flat.extend(graph.graph_features)
    return tuple(flat)


def build_grounded_dataset(
    oracle: NetOracle,
    vocab: NetVocab,
    config: NetConfig,
    *,
    driver: str = "weighted",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 40,
    observed_fraction: float = 0.5,
) -> list[GroundedExample]:
    """Seeded rollouts → :class:`GroundedExample` per step (SPEC-8 OG1 targets, EN8 input).

    ``observed_fraction`` sets the partial-observation split that makes the residual non-degenerate
    (SPEC-8 §3): the first ``ceil(fraction·H)`` hosts in canonical order are observed, so an edit
    touching only those is decidable (``D``) and the rest is residual (``R``). The trajectory always
    advances on the **true** next state — the oracle is the label source, never the rollout.
    """
    from verisim.netdata.drivers import NetDriver

    n_obs = max(1, math.ceil(observed_fraction * len(config.hosts)))
    observed = frozenset(config.hosts[:n_obs])
    examples: list[GroundedExample] = []
    for seed in seeds:
        driver_obj = NetDriver(name=driver, config=config, rng=random.Random(seed))
        state = NetworkState.initial(config.hosts)
        for _ in range(n_steps):
            action = driver_obj.sample(state)
            result = oracle.step(state, action)
            next_graph = build_graph(result.state, None, config)
            examples.append(
                GroundedExample(
                    graph=build_graph(state, action, config),
                    target_ids=tuple(encode_target(result.delta, vocab)),
                    residual_mask=tuple(residual_token_mask(result.delta, vocab, observed)),
                    next_graph=next_graph,
                    next_rawfeat=_raw_features(next_graph),
                )
            )
            state = result.state
    return examples


# --- H24: the objective axis (raw-likelihood vs residual / bits-to-correct) --------------


def _decoder_batch_loss(
    model: GraphRSSMWorldModel, batch: list[GroundedExample], *, objective: str
) -> Tensor:
    """Teacher-forced cross-entropy; ``objective='residual'`` masks the decidable (``D``) tokens."""
    device = model.net.device
    vocab = model.vocab
    width = max(len(ex.target_ids) for ex in batch)
    inp = torch.full((len(batch), width), vocab.pad, dtype=torch.long, device=device)
    lab = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long, device=device)
    for k, ex in enumerate(batch):
        seq = [vocab.gen, *ex.target_ids[:-1]]
        inp[k, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        labels = list(ex.target_ids)
        if objective == "residual":
            labels = [
                t if keep else IGNORE_INDEX
                for t, keep in zip(labels, ex.residual_mask, strict=True)
            ]
        lab[k, : len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    node, gfeat, a_link, a_flow = graphs_to_tensors([ex.graph for ex in batch], device)
    cond, _ = model.net.encode(node, gfeat, a_link, a_flow, sample=True)
    logits = model.net.decode_logits(cond, inp)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), lab.reshape(-1), ignore_index=IGNORE_INDEX
    )


def train_grounded_decoder(
    model: GraphRSSMWorldModel,
    examples: list[GroundedExample],
    *,
    objective: str = "likelihood",
    steps: int = 600,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Train the graph decoder with the raw-likelihood or **residual** objective (H24).

    ``objective='likelihood'`` is cross-entropy over every delta token (supervised baseline);
    ``objective='residual'`` masks the oracle-decidable tokens so capacity concentrates on ``R``
    (SPEC-8 §4.2). Same examples, same compute — the only difference is which tokens carry gradient.
    """
    if objective not in ("likelihood", "residual"):
        raise ValueError(f"objective must be 'likelihood' or 'residual', got {objective!r}")
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
        loss = _decoder_batch_loss(model, batch, objective=objective)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def residual_token_accuracy(
    model: GraphRSSMWorldModel, examples: list[GroundedExample]
) -> tuple[float, float]:
    """Teacher-forced token accuracy ``(overall, residual-only)`` over ``examples``.

    The residual-only figure is the one H24 is really about — accuracy on the genuinely-uncertain
    bits ``R`` the oracle does not hand the model for free.
    """
    model.net.eval()
    device = model.net.device
    vocab = model.vocab
    overall_correct = overall_total = res_correct = res_total = 0
    for ex in examples:
        seq = [vocab.gen, *ex.target_ids[:-1]]
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        node, gfeat, a_link, a_flow = graphs_to_tensors([ex.graph], device)
        cond, _ = model.net.encode(node, gfeat, a_link, a_flow, sample=False)
        preds = model.net.decode_logits(cond, inp)[0].argmax(dim=-1).tolist()
        for pred, true, keep in zip(preds, ex.target_ids, ex.residual_mask, strict=True):
            overall_total += 1
            overall_correct += int(pred == true)
            if keep:
                res_total += 1
                res_correct += int(pred == true)
    overall = overall_correct / overall_total if overall_total else 1.0
    residual = res_correct / res_total if res_total else 1.0
    return overall, residual


# --- H23: the collapse axis (oracle-anchored vs learned target; VICReg on/off) -----------


class JEPAPredictor(nn.Module):
    """A small predictor head ``g`` mapping the online graph summary to the target space (§4.1)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


def _vicreg(z: Tensor) -> Tensor:
    """VICReg variance + covariance regularizer on a batch of embeddings (the collapse machinery).

    Variance: hinge each dimension's std up to 1 (forbids the constant-collapse solution).
    Covariance: penalize off-diagonal feature correlation (forbids the dimension-collapse solution).
    """
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss + cov_loss


@dataclass(frozen=True)
class JEPAResult:
    """The EN8 collapse-cell readout: training loss + the representation-health diagnostics."""

    final_loss: float
    emb_std: float
    eff_rank: float


def _oracle_projection(d_raw: int, d_out: int, seed: int) -> Tensor:
    """A fixed (frozen) random projection of true-next-state features → target space (§4.1).

    Deterministic in ``seed``; never trained. Because the target is this fixed function of *real*
    next-state data, it has full variance by construction — the external referent H23 is about.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(d_raw, d_out, generator=g) / math.sqrt(d_raw)


def train_jepa(
    model: GraphRSSMWorldModel,
    examples: list[GroundedExample],
    *,
    target: str = "oracle",
    collapse_machinery: bool = True,
    steps: int = 400,
    lr: float = 3e-3,
    batch_size: int = 32,
    ema_decay: float = 0.99,
    vicreg_coef: float = 1.0,
    seed: int = 0,
) -> JEPAResult:
    """JEPA latent-predictive pretraining of the graph encoder (the H23 collapse cell, §4.1).

    ``target='oracle'`` regresses the predicted embedding onto a fixed projection of the **true next
    state** (the oracle-anchored external referent); ``target='learned'`` regresses onto an encoder
    of the next state — an EMA copy when ``collapse_machinery`` is on (the BYOL/JEPA baseline), else
    the online encoder with stop-gradient (the naked predictive loss that collapses). VICReg
    + covariance terms are added iff ``collapse_machinery`` — so the four cells are
    ``target x machinery``. Returns the final loss and representation health (the collapse readout).
    """
    if target not in ("oracle", "learned"):
        raise ValueError(f"target must be 'oracle' or 'learned', got {target!r}")
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    device = model.net.device
    d = model.net.config.d_model
    d_raw = len(examples[0].next_rawfeat)
    predictor = JEPAPredictor(d, d).to(device)
    proj = _oracle_projection(d_raw, d, seed).to(device)
    ema: GraphRSSMNet | None = None
    if target == "learned" and collapse_machinery:
        ema = copy.deepcopy(model.net)
        for p in ema.parameters():
            p.requires_grad_(False)

    params = list(model.net.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    n = len(examples)
    perm: list[int] = []
    cursor = 0
    final_loss = 0.0
    for _ in range(steps):
        if cursor + batch_size > n or not perm:
            perm = torch.randperm(n, generator=gen).tolist()
            cursor = 0
        batch = [examples[i] for i in perm[cursor : cursor + batch_size]]
        cursor += batch_size
        model.net.train()

        node, gfeat, a_link, a_flow = graphs_to_tensors([ex.graph for ex in batch], device)
        z = model.net.embed(node, gfeat, a_link, a_flow)  # online representation [B,d]
        pred = predictor(z)

        nnode, ngfeat, na_link, na_flow = graphs_to_tensors(
            [ex.next_graph for ex in batch], device
        )
        if target == "oracle":
            raw = torch.tensor([ex.next_rawfeat for ex in batch], device=device)
            tgt = (raw @ proj).detach()
        elif ema is not None:
            ema.eval()
            tgt = ema.embed(nnode, ngfeat, na_link, na_flow).detach()
        else:  # learned target, machinery off: online encoder, stop-gradient (collapses)
            tgt = model.net.embed(nnode, ngfeat, na_link, na_flow).detach()

        loss = nn.functional.mse_loss(pred, tgt)
        if collapse_machinery:
            loss = loss + vicreg_coef * _vicreg(z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for ep, op in zip(ema.parameters(), model.net.parameters(), strict=True):
                    ep.mul_(ema_decay).add_(op, alpha=1.0 - ema_decay)
        final_loss = float(loss.item())

    std, rank = representation_health(model, examples)
    return JEPAResult(final_loss=final_loss, emb_std=std, eff_rank=rank)


@torch.no_grad()
def representation_health(
    model: GraphRSSMWorldModel, examples: list[GroundedExample]
) -> tuple[float, float]:
    """``(emb_std, eff_rank)`` of the online graph summary over ``examples`` — the collapse readout.

    ``emb_std`` is the mean per-dimension standard deviation (→ 0 under collapse). ``eff_rank`` is
    effective rank ``exp(H(p))`` of the centered embedding's normalized singular-value spectrum
    (→ 1 under collapse, → d for full rank) — the standard JEPA collapse diagnostic.
    """
    model.net.eval()
    device = model.net.device
    node, gfeat, a_link, a_flow = graphs_to_tensors([ex.graph for ex in examples], device)
    z = model.net.embed(node, gfeat, a_link, a_flow)
    emb_std = float(z.std(dim=0).mean().item())
    zc = z - z.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum().clamp_min(1e-12)
    entropy = -(p * (p.clamp_min(1e-12)).log()).sum()
    eff_rank = float(torch.exp(entropy).item())
    return emb_std, eff_rank


__all__ = [
    "GroundedExample",
    "JEPAResult",
    "build_grounded_dataset",
    "edit_hosts",
    "edit_is_decidable",
    "representation_health",
    "residual_token_accuracy",
    "residual_token_mask",
    "train_grounded_decoder",
    "train_jepa",
]
