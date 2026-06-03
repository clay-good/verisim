"""The factored interaction-graph arm of the host ``M_θ`` (SPEC-6 §6.1, DD-H1; HC4 increment 2).

The structured alternative to the flat HC4 transformer (the EH4 comparison). Where the flat arm
serializes the whole ``(state, action)`` to a token stream, the factored arm:

  1. **featurizes** the bundle as a process-interaction graph (torch-free, :mod:`.graph`);
  2. **message-passes** over the two interaction edge sets -- lineage (the fork tree) and
     shared-file (processes coupled through a common open path) -- action-conditioned, deep, so a
     syscall's effect that depends on a reference several hops away is representable, and the
     cross-subsystem coupling H13 found load-bearing is *in the architecture*, not flattened away;
  3. carries an **RSSM latent belief** (§6.2): a GRU recurrence with a stochastic state whose
     variance is a calibrated-by-construction uncertainty signal (the upgrade over the flat arm's
     entropy). Under full observability the belief is computed per-step and degenerates to a Markov
     predictor, exactly as §6.2 says it must;
  4. decodes the structured bundle delta with a small transformer **conditioned on the graph/RSSM
     summary plus the acting process's node embedding**, under the *same*
     :class:`~verisim.hostmodel.grammar.HostDeltaGrammar` as the flat arm, so every prediction is a
     valid :class:`~verisim.host.delta.HostDelta` regardless of weights and is parsed by the *same*
     :func:`~verisim.hostmodel.tokenizer.parse_target`.

It implements the loop's :class:`~verisim.hostloop.model.HostModel` /
:class:`~verisim.hostloop.model.HostUncertaintyModel` protocols, so it drops into the shipped HC5
loop unchanged -- the loop never knows whether it holds the flat arm or this one.

The process table has a *variable* size, unlike the network world's fixed host count, so graphs are
**process-indexed with a validity mask** (node ``i`` == pid ``i``): dense batching stays trivial and
fast, and invalid pids are masked out of pooling and carry no edges. The §6.3 drift levers
(noise/self-forcing) and per-subsystem decode heads are deferred to a later HC increment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from verisim.host.action import HostAction
from verisim.host.config import HostConfig
from verisim.host.delta import HostDelta
from verisim.host.state import HostState
from verisim.hostmetrics.divergence import SUBSYSTEMS
from verisim.model.transformer import Block, GPTConfig

from .grammar import OP_SUBSYSTEM, HostDeltaGrammar
from .graph import HostGraph, build_host_graph, feature_dims
from .tokenizer import parse_target
from .vocab import HostVocab


@dataclass
class GraphHostConfig:
    """Sizing for the factored arm. Defaults are deliberately small (the SLM bet, §6.4)."""

    node_dim: int
    graph_dim: int
    vocab_size: int
    d_model: int = 64
    mp_rounds: int = 3  # message-passing depth (≈ fork-tree depth + coupling hops, §6.1)
    n_layer: int = 2  # decoder transformer blocks
    n_head: int = 2
    block_size: int = 64  # bundle-delta token sequences are short
    stoch_dim: int = 16  # RSSM stochastic state width (§6.2)
    dropout: float = 0.0


def graphs_to_tensors(
    graphs: list[HostGraph], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batch same-``N`` :class:`HostGraph` into dense tensors.

    Returns ``(node[B,N,Dn], gfeat[B,Dg], mask[B,N], a_lin[B,N,N], a_share[B,N,N], acting[B])``.
    ``a_lin`` is the directed ``parent -> child`` lineage adjacency; ``a_share`` the symmetric
    shared-file adjacency; ``mask`` marks live pids; ``acting`` indexes the syscall's process.
    """
    b = len(graphs)
    n = graphs[0].n_pids
    dn = graphs[0].dims.node
    dg = graphs[0].dims.graph
    node = torch.zeros(b, n, dn, device=device)
    gfeat = torch.zeros(b, dg, device=device)
    mask = torch.zeros(b, n, device=device)
    a_lin = torch.zeros(b, n, n, device=device)
    a_share = torch.zeros(b, n, n, device=device)
    acting = torch.zeros(b, dtype=torch.long, device=device)
    for k, gph in enumerate(graphs):
        node[k] = torch.tensor(gph.node_features, device=device)
        gfeat[k] = torch.tensor(gph.graph_features, device=device)
        mask[k] = torch.tensor(gph.node_mask, device=device)
        acting[k] = gph.acting_pid
        for ppid, pid in gph.lineage_edges:
            a_lin[k, ppid, pid] = 1.0
        for i, j in gph.share_edges:
            a_share[k, i, j] = 1.0
            a_share[k, j, i] = 1.0
    return node, gfeat, mask, a_lin, a_share, acting


def _row_normalize(a: Tensor) -> Tensor:
    """Mean-aggregation normalization: divide each row by its degree (clamped to >=1)."""
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1.0)


class GraphHostNet(nn.Module):
    """Masked message-passing encoder + RSSM belief + graph-conditioned delta decoder (§6.1-6.2)."""

    def __init__(self, config: GraphHostConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # --- encoder: node/graph projections + message passing ------------------
        self.node_in = nn.Linear(config.node_dim, d)
        self.graph_in = nn.Linear(config.graph_dim, d)
        self.mp_lin_fwd = nn.ModuleList([nn.Linear(d, d) for _ in range(config.mp_rounds)])
        self.mp_lin_rev = nn.ModuleList([nn.Linear(d, d) for _ in range(config.mp_rounds)])
        self.mp_share = nn.ModuleList([nn.Linear(d, d) for _ in range(config.mp_rounds)])
        self.mp_update = nn.ModuleList([nn.Linear(4 * d, d) for _ in range(config.mp_rounds)])
        self.mp_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(config.mp_rounds)])

        # --- RSSM belief --------------------------------------------------------
        self.rssm = nn.GRUCell(d, d)
        self.to_mu = nn.Linear(d, config.stoch_dim)
        self.to_logsig = nn.Linear(d, config.stoch_dim)
        # conditioning vector = f(det belief, stochastic z, acting-process embedding)
        self.cond = nn.Linear(d + config.stoch_dim + d, d)

        # --- decoder: graph-conditioned causal transformer ----------------------
        block_cfg = GPTConfig(
            vocab_size=config.vocab_size, block_size=config.block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=d, dropout=config.dropout,
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

    def _message_pass(
        self, node: Tensor, gfeat: Tensor, mask: Tensor, a_lin: Tensor, a_share: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the masked message-passing encoder. Returns ``(h[B,N,d], g[B,d])``."""
        m = mask[:, :, None]
        h = self.node_in(node) * m  # zero out invalid pids' embeddings
        g = self.graph_in(gfeat)  # [B,d]
        a_fwd = _row_normalize(a_lin)
        a_rev = _row_normalize(a_lin.transpose(1, 2))
        a_shr = _row_normalize(a_share)
        for r in range(self.config.mp_rounds):
            m_fwd = a_fwd @ self.mp_lin_fwd[r](h)
            m_rev = a_rev @ self.mp_lin_rev[r](h)
            m_shr = a_shr @ self.mp_share[r](h)
            upd = self.mp_update[r](torch.cat([h, m_fwd, m_rev, m_shr], dim=-1))
            upd = upd + g[:, None, :]  # broadcast action conditioning to every node
            h = self.mp_norm[r](h + torch.nn.functional.gelu(upd)) * m
        return h, g

    def encode(
        self, node: Tensor, gfeat: Tensor, mask: Tensor, a_lin: Tensor, a_share: Tensor,
        acting: Tensor, *, sample: bool,
    ) -> tuple[Tensor, Tensor]:
        """Message-pass + RSSM. Returns ``(cond[B,d], belief_var[B])``.

        ``belief_var`` is the mean RSSM posterior variance -- the §6.2 calibrated uncertainty.
        ``sample`` draws the stochastic state (training); else it uses the mean (decode).
        """
        h, g = self._message_pass(node, gfeat, mask, a_lin, a_share)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask[:, :, None]).sum(dim=1) / denom  # [B,d] masked-mean graph summary
        acting_emb = h[torch.arange(h.size(0), device=h.device), acting]  # [B,d]

        h0 = torch.zeros_like(pooled)
        det = self.rssm(pooled + g, h0)  # [B,d]
        mu = self.to_mu(det)
        sigma = torch.nn.functional.softplus(self.to_logsig(det)) + 1e-4
        z = mu + sigma * torch.randn_like(sigma) if sample else mu
        cond = self.cond(torch.cat([det, z, acting_emb], dim=-1))
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


class GraphHostWorldModel:
    """The learned factored host ``M_θ`` behind the loop interface (SPEC-6 §6.1, DD-H1).

    Implements ``predict_delta`` (``HostModel``) and ``predict_delta_with_uncertainty``
    (``HostUncertaintyModel``, returning the RSSM **belief variance** -- §6.2), a drop-in for the
    HC5 loop exactly as the flat arm is.
    """

    def __init__(
        self, net: GraphHostNet, vocab: HostVocab, config: HostConfig, max_pid: int
    ) -> None:
        self.net = net
        self.vocab = vocab
        self.config = config
        self.max_pid = max_pid
        self.grammar = HostDeltaGrammar(vocab)

    @torch.no_grad()
    def _decode(
        self, state: HostState, action: HostAction, *, max_edits: int, max_new_tokens: int
    ) -> tuple[HostDelta, float, dict[str, float]]:
        self.net.eval()
        device = self.net.device
        gph = build_host_graph(state, action, self.config, self.max_pid)
        node, gfeat, mask, a_lin, a_share, acting = graphs_to_tensors([gph], device)
        cond, belief_var = self.net.encode(
            node, gfeat, mask, a_lin, a_share, acting, sample=False
        )

        seq = [self.vocab.gen]  # start-of-decode marker (state lives in `cond`)
        generated: list[int] = []
        stack = self.grammar.start()
        edits = 0
        # Per-subsystem decode entropy (§5.4, §8.2): each token's masked-distribution entropy is
        # bucketed into the subsystem of the op being decoded, giving the smart-π_w signal.
        per_subsystem = {sub: 0.0 for sub in SUBSYSTEMS}
        current_sub = "global"
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
            mask_v = torch.full((len(self.vocab),), float("-inf"), device=device)
            mask_v[list(allowed)] = 0.0
            masked = logits + mask_v
            token = int(torch.argmax(masked).item())

            if top == "DELTA" and token in self.vocab.op_ids:
                edits += 1
                current_sub = OP_SUBSYSTEM[self.vocab.token(token)]
            probs = torch.softmax(masked, dim=-1)
            per_subsystem[current_sub] += float(
                -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
            )
            stack = self.grammar.advance(stack, token)
            seq.append(token)
            generated.append(token)

        return parse_target(generated, self.vocab), float(belief_var[0].item()), per_subsystem

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta:
        delta, _, _ = self._decode(state, action, max_edits=64, max_new_tokens=4096)
        return delta

    def predict_delta_with_uncertainty(
        self, state: HostState, action: HostAction
    ) -> tuple[HostDelta, float]:
        """Return ``(delta, belief_variance)`` -- the §6.2 calibrated uncertainty signal."""
        delta, belief_var, _ = self._decode(state, action, max_edits=64, max_new_tokens=4096)
        return delta, belief_var

    def predict_delta_with_subsystem_uncertainty(
        self, state: HostState, action: HostAction
    ) -> tuple[HostDelta, dict[str, float]]:
        """Return ``(delta, per_subsystem_decode_entropy)`` -- the smart-``π_w`` signal (§8.2).

        The per-subsystem entropy localizes *where* the predicted delta is least certain, so a
        which-subsystem policy can spend the consult on the subsystem the model is least sure about
        (the §8.2 information-gain choice), instead of a fixed or round-robin target.
        """
        delta, _, per_subsystem = self._decode(state, action, max_edits=64, max_new_tokens=4096)
        return delta, per_subsystem


def build_host_graph_model(
    vocab: HostVocab,
    config: HostConfig,
    *,
    max_pid: int = 64,
    d_model: int = 64,
    mp_rounds: int = 3,
    n_layer: int = 2,
    n_head: int = 2,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> GraphHostWorldModel:
    """Construct an (untrained) factored arm sized to ``config`` and ``vocab`` (default CPU)."""
    torch.manual_seed(seed)
    dims = feature_dims(config, max_pid)
    cfg = GraphHostConfig(
        node_dim=dims.node, graph_dim=dims.graph, vocab_size=len(vocab),
        d_model=d_model, mp_rounds=mp_rounds, n_layer=n_layer, n_head=n_head,
    )
    net = GraphHostNet(cfg)
    if device is not None:
        net = net.to(device)
    return GraphHostWorldModel(net, vocab, config, max_pid)
