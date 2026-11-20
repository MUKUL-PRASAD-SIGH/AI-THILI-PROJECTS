"""Milestone 4: train the GPT from scratch on the product-description corpus.

Steps
1. Train the BPE tokenizer on the corpus and encode every record.
2. Hold out 10% of records (whole products) for validation.
3. Sample training windows of ``block_size + 1`` tokens. Most windows start at
   a ``<|product|>`` token so the model sees complete header -> description
   pairs; the rest start anywhere for variety.
4. AdamW with linear warm-up and cosine decay, gradient clipping, periodic
   validation; the checkpoint with the best validation loss is kept.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from app import storage
from app.corpus import load_corpus, split_records
from app.model import GPT, GPTConfig
from app.tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    max_steps: int = 1500
    lr: float = 2e-3
    min_lr: float = 2e-4
    warmup: int = 100
    weight_decay: float = 0.1
    eval_every: int = 150
    eval_batches: int = 20
    aligned_frac: float = 0.75
    seed: int = 1337


def pick_device(requested: str | None = None) -> str:
    import os
    requested = requested or os.environ.get("DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class Batches:
    """Random training windows from one long token stream."""

    def __init__(self, records: list[list[int]], block_size: int, aligned_frac: float,
                 rng: random.Random):
        self.stream = torch.tensor([t for r in records for t in r], dtype=torch.long)
        starts, pos = [], 0
        for r in records:
            starts.append(pos)
            pos += len(r)
        self.block = block_size
        limit = len(self.stream) - block_size - 1
        if limit <= 0:
            raise ValueError("corpus is shorter than one training window")
        self.starts = [s for s in starts if s <= limit] or [0]
        self.limit = limit
        self.aligned = aligned_frac
        self.rng = rng

    def get(self, batch_size: int, device: str):
        ix = [self.rng.choice(self.starts) if self.rng.random() < self.aligned
              else self.rng.randint(0, self.limit) for _ in range(batch_size)]
        x = torch.stack([self.stream[i:i + self.block] for i in ix])
        y = torch.stack([self.stream[i + 1:i + 1 + self.block] for i in ix])
        return x.to(device), y.to(device)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def estimate_loss(model: GPT, batches: Batches, n: int, batch_size: int, device: str) -> float:
    model.eval()
    losses = [model(*batches.get(batch_size, device))[1].item() for _ in range(n)]
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainConfig | None = None, corpus: str | None = None,
          out_dir: Path | None = None, device: str | None = None,
          log=print) -> dict:
    cfg = cfg or TrainConfig()
    device = pick_device(device)
    out_dir = out_dir or storage.model_dir()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    text = corpus if corpus is not None else load_corpus()
    t0 = time.time()
    tok = BPETokenizer.train(text, vocab_size=cfg.vocab_size)
    storage.save_tokenizer(out_dir, tok)
    log(f"tokenizer: {tok.vocab_size} tokens ({len(tok.merges)} merges) in {time.time() - t0:.1f}s")

    records = split_records(text)
    rng.shuffle(records)
    n_val = max(1, len(records) // 10)
    encoded = [tok.encode(r + "\n") for r in records]
    val_records, train_records = encoded[:n_val], encoded[n_val:]
    n_tokens = sum(map(len, train_records))
    log(f"corpus: {len(train_records)} train / {len(val_records)} val records, "
        f"{n_tokens:,} train tokens")

    block = cfg.block_size
    train_b = Batches(train_records, block, cfg.aligned_frac, rng)
    val_b = Batches(val_records, block, cfg.aligned_frac, random.Random(cfg.seed + 1))

    mcfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block, n_embd=cfg.n_embd,
                     n_head=cfg.n_head, n_layer=cfg.n_layer, dropout=cfg.dropout)
    model = GPT(mcfg).to(device)
    log(f"model: {model.num_params():,} parameters on {device}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))

    history, best = [], float("inf")
    t0 = time.time()
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            tr = estimate_loss(model, train_b, cfg.eval_batches, cfg.batch_size, device)
            va = estimate_loss(model, val_b, cfg.eval_batches, cfg.batch_size, device)
            history.append({"step": step, "train_loss": round(tr, 4), "val_loss": round(va, 4)})
            flag = ""
            if va < best:
                best = va
                storage.save_checkpoint(out_dir, model, {"step": step, "val_loss": va})
                flag = "  (saved)"
            log(f"step {step:5d} | train {tr:.3f} | val {va:.3f} | "
                f"{time.time() - t0:6.1f}s{flag}")
        if step == cfg.max_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        x, y = train_b.get(cfg.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    result = {"train_config": cfg.__dict__, "model_config": mcfg.to_dict(),
              "params": model.num_params(), "best_val_loss": round(best, 4),
              "seconds": round(time.time() - t0, 1), "history": history}
    storage.save_log(out_dir, result)
    return result


if __name__ == "__main__":
    train()
