"""Milestone 3: fine-tune GPT-2 on the labelled requests with the HF Trainer.

Each example is ``Request: <text>\\nCategory: <|label|><|endoftext|>``. The
loss is masked (label -100) everywhere except the category token and the
end-of-text token, so the model is trained to *choose a category*, not to
memorise how clients write.

Accuracy is measured inside the Trainer on the held-out split: for each
example we read the logits just before the category token, restricted to
the 8 category tokens, and compare the arg-max with the true label.
"""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import PrinterCallback, Trainer, TrainingArguments, set_seed

from app.core import LABELS, Example, label_token, load_examples, prompt, stratified_split
from app.llm import build_model_and_tokenizer, classify_batch, label_token_ids, model_dir, pick_device


class RequestDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_length: int = 96):
        self.items = []
        for ex in examples:
            p = tokenizer.encode(prompt(ex.text), add_special_tokens=False)[: max_length - 2]
            target = [tokenizer.convert_tokens_to_ids(label_token(ex.label)), tokenizer.eos_token_id]
            ids = p + target
            labels = [-100] * len(p) + target
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * (width - n))
            labels.append(b["labels"] + [-100] * (width - n))
            mask.append([1] * n + [0] * (width - n))
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def make_metric_fns(cand_ids: list[int]):
    cand = torch.tensor(cand_ids)

    def preprocess(logits, labels):
        # position t-1 predicts the token at t; find the category position
        is_cat = torch.isin(labels, cand.to(labels.device))
        pos = is_cat.int().argmax(dim=1)
        rows = torch.arange(labels.size(0), device=labels.device)
        cat_logits = logits[rows, pos - 1][:, cand.to(logits.device)]
        return cat_logits.argmax(dim=-1)

    def compute(eval_pred):
        preds, label_ids = eval_pred
        lookup = {tid: i for i, tid in enumerate(cand_ids)}
        gold = np.array([next(lookup[t] for t in row if t in lookup) for row in label_ids])
        return {"accuracy": float((np.asarray(preds) == gold).mean())}

    return preprocess, compute


def evaluate(model, tokenizer, examples: list[Example], device: str) -> dict:
    preds = classify_batch(model, tokenizer, [e.text for e in examples], device)
    correct = [p.label == e.label for p, e in zip(preds, examples)]
    confusion = {g: Counter() for g in LABELS}
    for p, e in zip(preds, examples):
        confusion[e.label][p.label] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in LABELS)
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[label] = {"precision": round(prec, 3), "recall": round(rec, 3),
                            "f1": round(f1, 3), "support": support}
    errors = [{"text": e.text, "true": e.label, "predicted": p.label,
               "confidence": round(p.confidence, 3)}
              for p, e, ok in zip(preds, examples, correct) if not ok]
    return {"accuracy": round(sum(correct) / len(examples), 4),
            "macro_f1": round(sum(c["f1"] for c in per_class.values()) / len(LABELS), 4),
            "mean_confidence": round(float(np.mean([p.confidence for p in preds])), 4),
            "per_class": per_class,
            "confusion": {g: dict(c) for g, c in confusion.items()},
            "errors": errors}


def train(base: str | None = None, out_dir: str | Path | None = None, epochs: float = 10,
          lr: float = 5e-5, batch_size: int = 8, seed: int = 42, device: str | None = None,
          extra_examples: list[Example] | None = None, examples: list[Example] | None = None,
          log=print) -> dict:
    set_seed(seed)
    out = model_dir(out_dir)
    device = pick_device(device)
    train_ex, test_ex = stratified_split(examples or load_examples())
    if extra_examples:
        test_texts = {e.text.lower() for e in test_ex}
        train_ex = train_ex + [e for e in extra_examples if e.text.lower() not in test_texts]
    log(f"data: {len(train_ex)} train / {len(test_ex)} test requests, {len(LABELS)} categories")

    model, tok = build_model_and_tokenizer(base)
    model.to(device)
    before = evaluate(model, tok, test_ex, device)
    log(f"before fine-tuning: accuracy {before['accuracy']:.3f} (chance is {1 / len(LABELS):.3f})")

    preprocess, compute = make_metric_fns(label_token_ids(tok))
    args = TrainingArguments(
        output_dir=str(out / "_trainer"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=seed,
        use_cpu=device == "cpu",
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=RequestDataset(train_ex, tok),
                      eval_dataset=RequestDataset(test_ex, tok),
                      data_collator=Collator(tok.pad_token_id),
                      compute_metrics=compute,
                      preprocess_logits_for_metrics=preprocess)
    trainer.remove_callback(PrinterCallback)       # we print our own summary per epoch
    t0 = time.time()
    trainer.train()
    seconds = round(time.time() - t0, 1)

    history = [{"epoch": round(h["epoch"], 2), **{k: round(v, 4) for k, v in h.items()
                                                    if k in ("loss", "eval_loss", "eval_accuracy")}}
               for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    for h in history:
        if "eval_accuracy" in h:
            log(f"epoch {h['epoch']:>5}: eval loss {h['eval_loss']:.3f}, "
                f"accuracy {h['eval_accuracy']:.3f}")

    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    shutil.rmtree(out / "_trainer", ignore_errors=True)
    tok.save_pretrained(str(out))
    after = evaluate(model, tok, test_ex, device)
    metrics = {"base_model": base or model.config.name_or_path, "epochs": epochs, "lr": lr,
               "train_size": len(train_ex), "test_size": len(test_ex), "seconds": seconds,
               "before": {k: before[k] for k in ("accuracy", "macro_f1", "mean_confidence")},
               "after": after, "history": history}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"after fine-tuning: accuracy {after['accuracy']:.3f}, macro-F1 {after['macro_f1']:.3f} "
        f"({seconds}s) -> saved to {out}")
    return metrics
