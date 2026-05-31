"""The message-passing + RSSM graph arm of ``M_θ`` (SPEC-5 §6.1-6.2, the NW8 arm).

This is the structured alternative to the flat NW4 transformer (the H11 comparison, EN4). Where
the flat arm serializes the whole ``(state, action)`` to a token stream and decodes the delta from
it, the graph arm:

  1. **featurizes** the state as a graph (torch-free, :mod:`verisim.netmodel.graph`);
  2. **message-passes** over the host graph (link edges for topology coupling, flow edges for
     established connections), action/clock conditioned, ``mp_rounds`` deep (≈ the network
     diameter, §6.1) so a firewall change several hops away is representable;
  3. carries an **RSSM latent belief** (§6.2): a GRU recurrence with a stochastic state whose
     variance is the *calibrated-by-construction* uncertainty signal EN2 wants — the upgrade over
     the flat arm's decode-entropy. Under full observability the belief is computed per-step from
     the current graph and degenerates to a Markov predictor exactly as §6.2 says it must;
  4. decodes the structured delta with a small transformer **conditioned on the graph/RSSM
     summary**, under the *same* :class:`~verisim.netmodel.grammar.NetDeltaGrammar` as the flat arm,
     so every prediction is a valid :class:`~verisim.netdelta.edits.NetDelta` regardless of weights,
     and parsed by the *same* :func:`~verisim.netmodel.tokenizer.parse_target`.

It implements the loop's :class:`~verisim.netloop.model.NetModel` and
:class:`~verisim.netloop.model.NetUncertaintyModel` protocols, so it drops into the shipped NW5
partial-observation loop unchanged — the loop never knows whether it holds the flat arm or this one.

v1 scope (honest): flow edges are featurized by ``(src, dst)`` adjacency without their port (the
action's port *is* seen, via the graph features); if EN4 shows the missing flow-port hurts
``advance`` prediction, that is a measurable finding and a cheap feature to add (SPEC-5 §6.1). The
belief is computed per-step (full-observability degenerate form); carrying it across a partially
observed rollout is the documented next step for the §8 probe loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from verisim.model.transformer import Block, GPTConfig
from verisim.net.action import NetAction
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState
from verisim.netdelta.edits import NetDelta

from .grammar import NetDeltaGrammar
from .graph import NetGraph, build_graph, feature_dims
from .tokenizer import parse_target
from .vocab import NetVocab


@dataclass
class GraphRSSMConfig:
    """Sizing for the graph arm. Defaults are deliberately small (the SLM bet, §6.4)."""

    node_dim: int
    graph_dim: int
    vocab_size: int
    d_model: int = 64
    mp_rounds: int = 3  # message-passing depth ≈ network diameter (§6.1)
    n_layer: int = 2  # decoder transformer blocks
    n_head: int = 2
    block_size: int = 64  # delta token sequences are short; no long context needed
    stoch_dim: int = 16  # RSSM stochastic state width (§6.2)
    dropout: float = 0.0


def graphs_to_tensors(
    graphs: list[NetGraph], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Batch a list of (same-``N``) :class:`NetGraph` into dense tensors.

    Returns ``(node[B,N,Dn], gfeat[B,Dg], a_link[B,N,N], a_flow[B,N,N])``. ``a_link`` is the
    symmetric link adjacency with self-loops; ``a_flow`` is the directed src→dst flow adjacency.
    All graphs share the closed world's host count, so ``N`` is constant and dense batching is
    trivial and fast at this scale.
    """
    b = len(graphs)
    n = len(graphs[0].host_ids)
    dn = graphs[0].dims.node
    dg = graphs[0].dims.graph
    node = torch.zeros(b, n, dn, device=device)
    gfeat = torch.zeros(b, dg, device=device)
    a_link = torch.zeros(b, n, n, device=device)
    a_flow = torch.zeros(b, n, n, device=device)
    for k, g in enumerate(graphs):
        node[k] = torch.tensor(g.node_features, device=device)
        gfeat[k] = torch.tensor(g.graph_features, device=device)
        for i in range(n):
            a_link[k, i, i] = 1.0  # self-loop
        for i, j in g.link_edges:
            a_link[k, i, j] = 1.0
            a_link[k, j, i] = 1.0  # symmetric
        for s, d, _port in g.flow_edges:
            a_flow[k, s, d] = 1.0
    return node, gfeat, a_link, a_flow


def _row_normalize(a: Tensor) -> Tensor:
    """Mean-aggregation normalization: divide each row by its degree (≥1)."""
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1.0)


class GraphRSSMNet(nn.Module):
    """Message-passing encoder + RSSM belief + graph-conditioned delta decoder (§6.1-6.2)."""

    def __init__(self, config: GraphRSSMConfig, net_config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.net_config = net_config
        d = config.d_model

        # --- encoder: node/graph projections + message passing ------------------
        self.node_in = nn.Linear(config.node_dim, d)
        self.graph_in = nn.Linear(config.graph_dim, d)
        self.mp_link = nn.ModuleList([nn.Linear(d, d) for _ in range(config.mp_rounds)])
        self.mp_flow_fwd = nn.ModuleList([nn.Linear(d, d) for _ in range(config.mp_rounds)])
        self.mp_flow_rev = nn.ModuleList([nn.Linear(d, d) for _ in range(config.mp_rounds)])
        self.mp_update = nn.ModuleList(
            [nn.Linear(4 * d, d) for _ in range(config.mp_rounds)]
        )
        self.mp_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(config.mp_rounds)])

        # --- RSSM belief --------------------------------------------------------
        self.rssm = nn.GRUCell(d, d)
        self.to_mu = nn.Linear(d, config.stoch_dim)
        self.to_logsig = nn.Linear(d, config.stoch_dim)
        self.cond = nn.Linear(d + config.stoch_dim, d)  # decoder conditioning vector

        # --- decoder: graph-conditioned causal transformer ----------------------
        block_cfg = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=d,
            dropout=config.dropout,
        )
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.block_size, d)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(block_cfg) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- encoder ----------------------------------------------------------------

    def encode(
        self, node: Tensor, gfeat: Tensor, a_link: Tensor, a_flow: Tensor, *, sample: bool
    ) -> tuple[Tensor, Tensor]:
        """Message-pass + RSSM. Returns ``(cond[B,d], belief_var[B])``.

        ``belief_var`` is the mean RSSM posterior variance — the §6.2 calibrated uncertainty
        signal. ``sample`` draws the stochastic state (training); else it uses the mean (decode).
        """
        h = self.node_in(node)  # [B,N,d]
        g = self.graph_in(gfeat)  # [B,d]
        a_link_n = _row_normalize(a_link)
        a_flow_fwd = _row_normalize(a_flow)
        a_flow_rev = _row_normalize(a_flow.transpose(1, 2))
        for r in range(self.config.mp_rounds):
            m_link = a_link_n @ self.mp_link[r](h)
            m_fwd = a_flow_fwd @ self.mp_flow_fwd[r](h)
            m_rev = a_flow_rev @ self.mp_flow_rev[r](h)
            upd = self.mp_update[r](torch.cat([h, m_link, m_fwd, m_rev], dim=-1))
            upd = upd + g[:, None, :]  # broadcast action/clock conditioning to every node
            h = self.mp_norm[r](h + torch.nn.functional.gelu(upd))
        pooled = h.mean(dim=1)  # [B,d] permutation-invariant graph summary

        # RSSM: deterministic recurrence (zero prior under full obs) + stochastic state.
        h0 = torch.zeros_like(pooled)
        det = self.rssm(pooled + g, h0)  # [B,d]
        mu = self.to_mu(det)
        sigma = torch.nn.functional.softplus(self.to_logsig(det)) + 1e-4
        z = mu + sigma * torch.randn_like(sigma) if sample else mu
        cond = self.cond(torch.cat([det, z], dim=-1))
        belief_var = sigma.pow(2).mean(dim=-1)  # [B]
        return cond, belief_var

    # -- decoder ----------------------------------------------------------------

    def decode_logits(self, cond: Tensor, idx: Tensor) -> Tensor:
        """Next-token logits for token ids ``idx[B,T]`` conditioned on ``cond[B,d]``."""
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"decode length {t} exceeds block_size {self.config.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :] + cond[:, None, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits: Tensor = self.head(self.ln_f(x))
        return logits


class GraphRSSMWorldModel:
    """The learned graph ``M_θ`` behind the loop interface (SPEC-5 §6.1-6.2).

    Implements ``predict_delta`` (``NetModel``) and ``predict_delta_with_uncertainty``
    (``NetUncertaintyModel``, returning the RSSM **belief variance** — §6.2), so it is a drop-in
    for the NW5 partial-observation loop exactly as the flat arm is.
    """

    def __init__(self, net: GraphRSSMNet, vocab: NetVocab, config: NetConfig) -> None:
        self.net = net
        self.vocab = vocab
        self.config = config
        self.grammar = NetDeltaGrammar(vocab)

    @torch.no_grad()
    def _decode(
        self, state: NetworkState, action: NetAction, *, max_edits: int, max_new_tokens: int
    ) -> tuple[NetDelta, float]:
        self.net.eval()
        device = self.net.device
        g = build_graph(state, action, self.config)
        node, gfeat, a_link, a_flow = graphs_to_tensors([g], device)
        cond, belief_var = self.net.encode(node, gfeat, a_link, a_flow, sample=False)

        seq = [self.vocab.gen]  # start-of-decode marker (state lives in `cond`)
        generated: list[int] = []
        stack = self.grammar.start()
        edits = 0
        while not self.grammar.is_accept(stack):
            if len(generated) >= max_new_tokens:
                raise RuntimeError("graph constrained_decode exceeded max_new_tokens")
            top = stack[0]
            window = seq[-self.net.config.block_size :]
            logits = self.net.decode_logits(
                cond, torch.tensor([window], dtype=torch.long, device=device)
            )[0, -1]

            allowed = self.grammar.allowed(stack)
            if top == "DELTA" and edits >= max_edits:
                allowed = frozenset({self.vocab.eos})
            mask = torch.full((len(self.vocab),), float("-inf"), device=device)
            mask[list(allowed)] = 0.0
            token = int(torch.argmax(logits + mask).item())

            if top == "DELTA" and token in self.vocab.op_ids:
                edits += 1
            stack = self.grammar.advance(stack, token)
            seq.append(token)
            generated.append(token)

        return parse_target(generated, self.vocab), float(belief_var[0].item())

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta:
        delta, _ = self._decode(state, action, max_edits=64, max_new_tokens=4096)
        return delta

    def predict_delta_with_uncertainty(
        self, state: NetworkState, action: NetAction
    ) -> tuple[NetDelta, float]:
        """Return ``(delta, belief_variance)`` — the §6.2 calibrated uncertainty signal."""
        return self._decode(state, action, max_edits=64, max_new_tokens=4096)


def build_graph_model(
    vocab: NetVocab, config: NetConfig, *, d_model: int = 64, mp_rounds: int = 3, seed: int = 0
) -> GraphRSSMWorldModel:
    """Construct an (untrained) graph arm sized to ``config`` and ``vocab``."""
    torch.manual_seed(seed)
    dims = feature_dims(config)
    cfg = GraphRSSMConfig(
        node_dim=dims.node,
        graph_dim=dims.graph,
        vocab_size=len(vocab),
        d_model=d_model,
        mp_rounds=mp_rounds,
    )
    return GraphRSSMWorldModel(GraphRSSMNet(cfg, config), vocab, config)
