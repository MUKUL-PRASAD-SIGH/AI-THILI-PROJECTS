"""The GPT, built from scratch in PyTorch.

Milestone 2 provides the embedding layer: every token id becomes a learned
vector, and a second learned vector for its position is added, so the model
knows both *what* a token is and *where* it sits in the sequence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


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
