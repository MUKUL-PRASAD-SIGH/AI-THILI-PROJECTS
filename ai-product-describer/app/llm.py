"""Milestone 5: the "LLM" of this project is our own GPT, trained from scratch.

``Describer`` loads the trained tokenizer + checkpoint and turns product
details into descriptions.
"""
from __future__ import annotations

from pathlib import Path

import torch

from app import storage
from app.core import END, LENGTHS, Product, finish_text
from app.train import pick_device

# Token budget per requested length (a sentence is ~12-20 tokens).
MAX_TOKENS = {"short": 60, "medium": 110, "long": 170}


class Describer:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None):
        self.dir = storage.model_dir(model_dir)
        self.device = pick_device(device)
        self.tok = storage.load_tokenizer(self.dir)
        self.model, self.meta = storage.load_checkpoint(self.dir, self.device)
        self.end_id = self.tok.special[END]

    def complete(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
                 top_k: int | None = 40, n: int = 1, seed: int | None = None) -> list[str]:
        """Continue raw prompt text; returns only the new text for each sample."""
        ids = self.tok.encode(prompt)
        x = torch.tensor([ids] * n, dtype=torch.long, device=self.device)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        out = self.model.generate(x, max_new_tokens, temperature, top_k,
                                  stop_id=self.end_id, generator=gen)
        return [self.tok.decode(row[len(ids):].tolist()) for row in out]

    def describe(self, product: Product, n: int = 1, temperature: float = 0.8,
                 top_k: int | None = 40, seed: int | None = None,
                 max_new_tokens: int | None = None) -> list[str]:
        """Generate ``n`` distinct description variants for one product."""
        if product.length not in LENGTHS:
            raise ValueError(f"length must be one of {LENGTHS}")
        budget = max_new_tokens or MAX_TOKENS[product.length]
        results: list[str] = []
        attempts = 0
        while len(results) < n and attempts < 4:
            raw = self.complete(product.prompt(), budget, temperature, top_k,
                                n=(n - len(results)) * 2 if attempts else n,
                                seed=None if seed is None else seed + attempts)
            for text in raw:
                text = finish_text(text)
                if text and text not in results:
                    results.append(text)
            attempts += 1
        return results[:n]
