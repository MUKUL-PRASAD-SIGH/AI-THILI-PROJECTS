"""Train the dialogue GPT on the corpus and save a checkpoint.

Batches are random ``block_size`` windows of the training text. Validation loss is
measured on *every* non-overlapping window of the held-out text, so it is
deterministic and can be compared exactly between the float model and its
quantized versions (see ``app/quantize.py``).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from app import core, corpus, storage
from app.model import GPT, GPTConfig

CHECKPOINT = "gpt_fp32.pt"


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 100
    weight_decay: float = 0.1
    eval_every: int = 250
    seed: int = 0


def load_data(path: str | Path = corpus.CORPUS_FILE, val_frac: float = 0.1, seed: int = 0):
    """Tokenizer over the whole corpus, and (train_ids, val_ids) split by whole exchanges."""
    text = Path(path).read_text(encoding="utf-8")
    tok = core.CharTokenizer.from_text(text)
    train_text, val_text = core.split_exchanges(core.parse_corpus(text), val_frac, seed)
    as_ids = lambda t: torch.tensor(tok.encode(t), dtype=torch.long)  # noqa: E731
    return tok, as_ids(train_text), as_ids(val_text)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, data: torch.Tensor, block_size: int, batch_size: int = 64) -> float:
    """Mean next-character cross-entropy over all non-overlapping windows of ``data``."""
    was_training = model.training
    model.eval()
    n = (len(data) - 1) // block_size
    xs = data[: n * block_size].view(n, block_size)
    ys = data[1: n * block_size + 1].view(n, block_size)
    total = 0.0
    for i in range(0, n, batch_size):
        _, loss = model(xs[i: i + batch_size], ys[i: i + batch_size])
        total += float(loss) * len(xs[i: i + batch_size])
    model.train(was_training)
    return total / n


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def train(model_cfg: GPTConfig | None = None, cfg: TrainConfig | None = None, data=None,
          out_dir: str | Path | None = None, log: Callable[[str], None] = print) -> tuple[GPT, dict]:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    tok, train_ids, val_ids = data or load_data()
    model_cfg = model_cfg or GPTConfig()
    model_cfg.vocab_size = tok.vocab_size
    model = GPT(model_cfg)
    log(f"GPT: {model.num_params():,} parameters, {model_cfg.n_layer} layers, {model_cfg.n_head} heads, "
        f"d={model_cfg.n_embd}, context {model_cfg.block_size}; {len(train_ids):,} training characters")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(cfg.seed)
    history = {"evals": [], "train_config": asdict(cfg), "model_config": model_cfg.to_dict(),
               "params": model.num_params()}
    best_val, best_state = math.inf, None
    start = time.time()
    model.train()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            val = eval_loss(model, val_ids, model_cfg.block_size)
            history["evals"].append({"step": step, "val_loss": val, "seconds": round(time.time() - start, 1)})
            mark = ""
            if val < best_val:
                best_val, best_state, mark = val, {k: v.clone() for k, v in model.state_dict().items()}, "  *best"
            log(f"step {step:5d}  val loss {val:.3f}  ({time.time() - start:.0f}s){mark}")
            if step == cfg.steps:
                break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_batch(train_ids, model_cfg.block_size, cfg.batch_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.load_state_dict(best_state)
    model.eval()
    history.update(best_val_loss=best_val, train_seconds=round(time.time() - start, 1))
    if out_dir is not None:
        save_checkpoint(model, tok, history, Path(out_dir) / CHECKPOINT)
        log(f"saved {storage.display(Path(out_dir) / CHECKPOINT)} (best val loss {best_val:.3f})")
    return model, history


def save_checkpoint(model: GPT, tok: core.CharTokenizer, history: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model.cfg.to_dict(), "state_dict": model.state_dict(),
                "chars": tok.chars, "history": history}, path)
