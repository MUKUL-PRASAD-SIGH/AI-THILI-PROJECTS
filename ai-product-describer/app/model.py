"""The GPT, built from scratch in PyTorch.

    token ids
      -> Embeddings            (token vector + position vector)
      -> Block x n_layer       (causal self-attention + MLP, each with a
                                pre-LayerNorm and a residual connection)
      -> LayerNorm -> Linear   (scores over the vocabulary for the next token)

Nothing here uses ``nn.Transformer`` or ``scaled_dot_product_attention``:
attention is written out by hand so every step is visible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128      # longest context the model can see
    n_embd: int = 128          # width of every token vector
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class Embeddings(nn.Module):
    """token embedding + learned positional embedding."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.block_size = cfg.block_size
        self.token = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch, time) integer token ids -> (batch, time, n_embd)
        T = idx.size(1)
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        pos = torch.arange(T, device=idx.device)
        return self.drop(self.token(idx) + self.position(pos))


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position may only look at itself
    and earlier positions (so the model cannot peek at the answer)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # queries, keys, values at once
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # lower-triangular mask: 1 where attending is allowed
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, h, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        weights = att
        att = self.attn_drop(att)
        y = att @ v                                                  # (B, h, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)             # re-join the heads
        y = self.resid_drop(self.proj(y))
        return (y, weights) if return_weights else y


class MLP(nn.Module):
    """Position-wise feed-forward network (expand x4, GELU, project back)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (MLP)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.token.weight     # weight tying (as in GPT-2)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                 # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        return logits, loss
