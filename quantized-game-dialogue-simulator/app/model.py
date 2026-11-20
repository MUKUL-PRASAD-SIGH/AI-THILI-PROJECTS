"""A small GPT written from scratch: character embeddings, causal self-attention, MLP blocks.

Every learned projection is an ``nn.Linear`` inside a ``LinearSlot``:

    x -> QuantStub -> Linear -> DeQuantStub -> y

In the float model the stubs are identities, so the slot is just a Linear. They exist so
eager-mode *static* quantization can be applied to exactly these layers: the stubs
become Quantize / DeQuantize ops with calibrated activation scales, and the Linear
becomes an int8 ``quantized.Linear``. Softmax, LayerNorm, GELU and the attention
matmuls stay in float (eager static quantization has no int8 kernels for them).
Dynamic quantization finds the same ``nn.Linear`` modules without needing the stubs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 64
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class LinearSlot(nn.Module):
    """A Linear layer with quant/dequant stubs around it (identities until quantized)."""

    def __init__(self, n_in: int, n_out: int, bias: bool = True):
        super().__init__()
        self.quant = QuantStub()
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.linear(self.quant(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = LinearSlot(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = LinearSlot(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))  # no peeking at future tokens
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = LinearSlot(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = LinearSlot(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = LinearSlot(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 style: scale residual projections by depth
            if name.endswith("proj.linear.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence of {T} tokens is longer than block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
             top_k: int | None = 20, stop_token: int | None = None, generator: torch.Generator | None = None,
             block_size: int | None = None) -> torch.Tensor:
    """Autoregressive sampling. temperature 0 = greedy. Stops early when every row emits ``stop_token``."""
    block_size = block_size or model.cfg.block_size
    done = torch.zeros(idx.size(0), dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
        if stop_token is not None:
            done |= nxt.squeeze(1) == stop_token
            if bool(done.all()):
                break
    return idx
