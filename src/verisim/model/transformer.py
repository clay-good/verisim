"""A small from-scratch decoder-only transformer (SPEC-2 §5.2).

nanoGPT-class: token + learned positional embeddings, a stack of pre-norm
causal-self-attention blocks, and a tied-free linear head. Built from scratch (no
``transformers``) because it is small enough to train on one local GPU/CPU and
doubles as the author's transformer-internals goal (SPEC-2 §11). Sizes start tiny
(SPEC-2 §5.2); the config defaults are deliberately small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        result: Tensor = self.dropout(self.proj(out))
        return result


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Maps a token-id sequence to next-token logits over the vocabulary."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits: Tensor = self.head(self.ln_f(x))
        return logits
