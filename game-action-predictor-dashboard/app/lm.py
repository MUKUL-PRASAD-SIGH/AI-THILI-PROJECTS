"""Continued pretraining of a causal language model on possession sequences, and next-action scoring.

This follows `thilimarketbasket.pretrain` (plain next-token loss, AdamW, linear warm-up then cosine decay to a
tenth of the peak, gradient clipping at 1.0, validation loss on held-out data), but in PyTorch instead of MLX so
it runs on any laptop CPU. One deliberate difference: thilimarketbasket shuffles a basket every time it is shown,
because a shopping basket is a set. A possession is a *sequence*, and its order is exactly what we want the
model to learn, so it is written in play order:

    Possession (Comets): inbound; bring up; pick and roll; drive; layup made<|endoftext|>

Scoring follows `thilimarketbasket.evaluate.score_candidates`: for a partial possession, every action in the
vocabulary is a candidate; its score is log P(" action" | text so far), summed over its tokens, plus the
log-probability that the action is complete there (followed by ";" or the end of the possession).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.simulate import ACTIONS


def label(action: str) -> str:
    return action.replace("_", " ")


def prefix_text(team: str, history: Sequence[str]) -> str:
    """The possession so far, ending where the next action goes."""
    text = f"Possession ({team}):"
    return text + (" " + "; ".join(label(a) for a in history) + ";" if history else "")


def possession_text(team: str, actions: Sequence[str]) -> str:
    return f"Possession ({team}): " + "; ".join(label(a) for a in actions)


def _torch():
    import torch

    return torch


class ActionLM:
    """A causal LM (any Hugging Face model) that ranks the next action of a possession."""

    def __init__(self, model, tokenizer, actions: Sequence[str] = ACTIONS):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.actions = list(actions)
        self.candidate_ids = [tokenizer.encode(" " + label(a)) for a in self.actions]
        self.end_ids = [tokenizer.encode(";")[0], tokenizer.eos_token_id]

    # -- loading ---------------------------------------------------------------
    @classmethod
    def load(cls, name_or_dir: str) -> "ActionLM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
        model = AutoModelForCausalLM.from_pretrained(name_or_dir)
        return cls(model, tokenizer)

    @classmethod
    def tiny(cls, seed: int = 0, n_layer: int = 2, n_embd: int = 128, tokenizer_name: str = "EleutherAI/pythia-14m") -> "ActionLM":
        """A small GPT-2-shaped model with random weights (trained from scratch, for comparison and tests)."""
        torch = _torch()
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        config = GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=n_embd, n_layer=n_layer,
                            n_head=max(1, n_embd // 32), bos_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id)
        return cls(GPT2LMHeadModel(config), tokenizer)

    def save(self, out_dir: str | Path, history: list | None = None, meta: dict | None = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        (out / "training.json").write_text(json.dumps({"history": history or [], **(meta or {})}, indent=2))
        return out

    # -- scoring ---------------------------------------------------------------
    def score_batch(self, contexts: Sequence[tuple[str, Sequence[str]]], batch_rows: int = 96) -> np.ndarray:
        """log P(action | context) for every action, for each (team, history). Shape (len(contexts), n_actions).

        Every (context, candidate) pair is one row; rows are padded on the right and run in large batches.
        """
        torch = _torch()
        rows, spans = [], []
        for team, history in contexts:
            prefix = self.tokenizer.encode(prefix_text(team, history))
            for cand in self.candidate_ids:
                rows.append(prefix + cand)
                spans.append((len(prefix), len(cand)))
        scores = np.zeros(len(rows))
        pad = self.tokenizer.eos_token_id
        head = self.model.get_output_embeddings()
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        with torch.no_grad():
            for start in range(0, len(order), batch_rows):
                chunk = order[start:start + batch_rows]
                width = max(len(rows[i]) for i in chunk)
                ids = torch.full((len(chunk), width), pad, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                for r, i in enumerate(chunk):
                    ids[r, :len(rows[i])] = torch.tensor(rows[i])
                    mask[r, :len(rows[i])] = 1
                # Run the transformer body once, then apply the output layer only at the positions that predict
                # the candidate's tokens and the end marker: a full (rows x width x vocab) logits tensor would
                # need gigabytes for a 50k-token vocabulary.
                hidden = self.model.base_model(input_ids=ids, attention_mask=mask)[0]
                row_idx, pos_idx, owners = [], [], []
                for r, i in enumerate(chunk):
                    p, c = spans[i]
                    for pos in range(p - 1, p + c):          # c positions for the tokens, 1 for the end marker
                        row_idx.append(r)
                        pos_idx.append(pos)
                    owners.append((i, len(pos_idx) - c - 1, p, c))
                logp = torch.log_softmax(head(hidden[row_idx, pos_idx]).float(), dim=-1)
                for i, first, p, c in owners:
                    toks = rows[i][p:p + c]
                    total = sum(float(logp[first + j, t]) for j, t in enumerate(toks))
                    total += float(torch.logsumexp(logp[first + c, self.end_ids], dim=0))   # the action ends here
                    scores[i] = total
        return scores.reshape(len(contexts), len(self.candidate_ids))

    def predict_proba(self, contexts: Sequence[tuple[str, Sequence[str]]], **kwargs) -> np.ndarray:
        """Probabilities over the action vocabulary (the candidate scores, renormalised)."""
        scores = self.score_batch(contexts, **kwargs)
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)

    # -- training --------------------------------------------------------------
    def _encode(self, possessions) -> list[list[int]]:
        eos = self.tokenizer.eos_token_id
        return [self.tokenizer.encode(possession_text(p.team, p.actions)) + [eos] for p in possessions]

    def _batch(self, examples):
        torch = _torch()
        width = max(len(e) for e in examples)
        ids = torch.full((len(examples), width), self.tokenizer.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(examples), width), dtype=torch.float32)
        for r, e in enumerate(examples):
            ids[r, :len(e)] = torch.tensor(e)
            mask[r, :len(e)] = 1
        return ids, mask

    def _loss(self, ids, mask):
        """Mean next-token cross-entropy over real (non-padding) tokens.

        The output layer is applied only at real token positions, so padding never costs a vocab-sized row.
        """
        torch = _torch()
        hidden = self.model.base_model(input_ids=ids, attention_mask=mask.long())[0][:, :-1]
        real = mask[:, 1:].bool()
        logits = self.model.get_output_embeddings()(hidden[real]).float()
        return torch.nn.functional.cross_entropy(logits, ids[:, 1:][real])

    def validation_loss(self, possessions, batch_size: int = 16) -> float:
        torch = _torch()
        examples = self._encode(possessions)
        total = count = 0.0
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                ids, mask = self._batch(examples[start:start + batch_size])
                n = float(mask[:, 1:].sum())
                total += float(self._loss(ids, mask)) * n
                count += n
        return total / count

    def pretrain(self, possessions, val_possessions=None, steps: int = 300, batch_size: int = 16,
                 learning_rate: float = 1e-4, warmup: int = 30, eval_every: int = 50, seed: int = 0,
                 log: Callable[[str], None] = print) -> list[dict]:
        """Continue training on `possessions` (next-token objective). Returns the loss history."""
        torch = _torch()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        examples = self._encode(possessions)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        def rate(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
        history, recent, started, order, cursor = [], [], time.time(), rng.permutation(len(examples)), 0
        if val_possessions is not None:
            history.append({"step": 0, "loss": None, "val_loss": self.validation_loss(val_possessions), "seconds": 0.0})
            log(f"step 0  val loss {history[-1]['val_loss']:.3f}")
        for step in range(1, steps + 1):
            if cursor + batch_size > len(order):
                order, cursor = rng.permutation(len(examples)), 0
            chosen = order[cursor:cursor + batch_size]
            cursor += batch_size
            self.model.train()
            loss = self._loss(*self._batch([examples[i] for i in chosen]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            recent.append(loss.item())
            if step % eval_every == 0 or step == steps:
                val = self.validation_loss(val_possessions) if val_possessions is not None else None
                history.append({"step": step, "loss": float(np.mean(recent)), "val_loss": val,
                                "seconds": round(time.time() - started, 1)})
                recent = []
                log(f"step {step}/{steps}  train loss {history[-1]['loss']:.3f}"
                    + (f"  val loss {val:.3f}" if val is not None else "") + f"  ({history[-1]['seconds']:.0f}s)")
        self.model.eval()
        return history
