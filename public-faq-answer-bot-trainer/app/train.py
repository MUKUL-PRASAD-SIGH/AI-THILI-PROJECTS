"""Fine-tune GPT-2 on the FAQ dataset by training only LoRA adapters.

Each example is ``prompt + " " + answer + <eos>``. The loss is computed only on
the answer tokens and the EOS token (prompt positions get label -100), so the model
learns to *answer* questions and to stop, not to re-generate questions.

The adapter with the lowest loss on the held-out paraphrases is kept.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from app import core, lora

IGNORE = -100


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 0.0
    warmup_frac: float = 0.06
    max_len: int = 128
    seed: int = 0
    lora: lora.LoRAConfig = field(default_factory=lora.LoRAConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def encode_example(tokenizer, question: str, answer: str, max_len: int = 128) -> tuple[list[int], list[int]]:
    """Token ids and labels for one example; labels mask the prompt with -100."""
    prompt_ids = tokenizer.encode(core.format_prompt(question))
    target_ids = tokenizer.encode(core.format_target(answer)) + [tokenizer.eos_token_id]
    ids = (prompt_ids + target_ids)[:max_len]
    labels = ([IGNORE] * len(prompt_ids) + target_ids)[:max_len]
    return ids, labels


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch. Padding is masked out of both attention and loss."""
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attention[i, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def batches(encoded: list, batch_size: int, seed: int | None = None):
    items = core.shuffled(encoded, seed) if seed is not None else list(encoded)
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def lm_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean next-token cross-entropy over answer tokens only."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = batch["labels"][:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE)


@torch.no_grad()
def evaluate_loss(model, encoded: list, batch_size: int, pad_id: int, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for chunk in batches(encoded, batch_size):
        batch = {k: v.to(device) for k, v in collate(chunk, pad_id).items()}
        n = int((batch["labels"][:, 1:] != IGNORE).sum())
        total += float(lm_loss(model, batch)) * n
        count += n
    return total / max(count, 1)


def lr_lambda(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


def train_lora(model, tokenizer, train_examples: list[core.Example], val_examples: list[core.Example],
               config: TrainConfig, device: str = "cpu", out_dir: str | Path | None = None,
               log: Callable[[str], None] = print, extra_meta: dict | None = None) -> dict:
    """Inject LoRA into ``model``, train it, keep the best adapter, and return the history.

    If ``out_dir`` is given the best adapter is saved there (only the LoRA weights).
    On return ``model`` holds the best adapter weights.
    """
    set_seed(config.seed)
    names = lora.inject_lora(model, config.lora)
    model.to(device)
    trainable, total = lora.count_parameters(model)
    log(f"LoRA r={config.lora.r} alpha={config.lora.alpha} on {len(names)} layers: "
        f"{trainable:,} trainable of {total:,} parameters ({100 * trainable / total:.2f}%)")

    pad_id = tokenizer.eos_token_id
    train_enc = [encode_example(tokenizer, e.question, e.answer, config.max_len) for e in train_examples]
    val_enc = [encode_example(tokenizer, e.question, e.answer, config.max_len) for e in val_examples]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_enc) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(total_steps, int(config.warmup_frac * total_steps)))

    history = {"epochs": [], "trainable_params": trainable, "total_params": total,
               "lora_layers": len(names), "config": config.to_dict()}
    val0 = evaluate_loss(model, val_enc, config.batch_size, pad_id, device)
    log(f"epoch 0  held-out loss {val0:.3f} (pre-trained model, adapter is zero)")
    history["initial_val_loss"] = val0
    best = (math.inf, None)
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for chunk in batches(train_enc, config.batch_size, seed=config.seed + epoch):
            batch = {k: v.to(device) for k, v in collate(chunk, pad_id).items()}
            loss = lm_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n += 1
        train_loss = running / max(n, 1)
        val_loss = evaluate_loss(model, val_enc, config.batch_size, pad_id, device)
        history["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                  "seconds": round(time.time() - start, 1)})
        marker = ""
        if val_loss < best[0]:
            best = (val_loss, {k: v.clone() for k, v in lora.lora_state_dict(model).items()})
            marker = "  *best"
        log(f"epoch {epoch:2d}  train loss {train_loss:.3f}  held-out loss {val_loss:.3f}"
            f"  ({time.time() - start:.0f}s){marker}")

    lora.load_lora_state_dict(model, best[1])
    history["best_val_loss"] = best[0]
    history["best_epoch"] = min(history["epochs"], key=lambda e: e["val_loss"])["epoch"]
    history["train_seconds"] = round(time.time() - start, 1)
    model.eval()
    if out_dir is not None:
        lora.save_adapter(model, config.lora, out_dir, extra={
            "train": {k: v for k, v in config.to_dict().items() if k != "lora"},
            "best_val_loss": best[0], "best_epoch": history["best_epoch"], **(extra_meta or {})})
        log(f"saved adapter to {out_dir}")
    return history
