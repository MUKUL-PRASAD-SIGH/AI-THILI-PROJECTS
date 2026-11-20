"""Milestone 2: GPT-2 with an extended tokenizer for the category tokens.

* The 8 category tokens (``<|venue_inquiry|>`` ...) plus a padding token are
  added to GPT-2's tokenizer, and the embedding matrix is resized to match.
* A brand-new token starts with a random vector. Instead, each category
  token is initialised to the average embedding of the words in its name
  ("venue", " inquiry"), so GPT-2's existing knowledge gives it a head start.
* Classification = one forward pass: take the logits at the last position of
  ``Request: ...\\nCategory:`` and softmax over **only** the 8 category tokens.
  That yields a label and a confidence score, and can never produce text
  that is not a category.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core import LABEL_TOKENS, LABELS, pretty, prompt, token_label

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "openai-community/gpt2"
PAD_TOKEN = "<|pad|>"


def _env(name: str, default: str) -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    return os.environ.get(name, default)


def base_model_name() -> str:
    return _env("BASE_MODEL", DEFAULT_BASE_MODEL)


def model_dir(path: str | Path | None = None) -> Path:
    return Path(path) if path else Path(_env("MODEL_DIR", str(PROJECT_ROOT / "models" / "sorter")))


def pick_device(requested: str | None = None) -> str:
    requested = requested or os.environ.get("DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def extend_tokenizer(tokenizer):
    """Add the category tokens and a pad token. Returns the number added."""
    added = tokenizer.add_special_tokens({
        "pad_token": PAD_TOKEN,
        "additional_special_tokens": LABEL_TOKENS,
    })
    tokenizer.padding_side = "right"
    return added


def build_model_and_tokenizer(base: str | None = None):
    """Load GPT-2, extend its tokenizer and resize + initialise embeddings."""
    base = base or base_model_name()
    tokenizer = AutoTokenizer.from_pretrained(base)
    old_vocab = len(tokenizer)
    extend_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(base)
    model.resize_token_embeddings(len(tokenizer))

    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for tok in LABEL_TOKENS:
            word_ids = tokenizer.encode(" " + pretty(token_label(tok)), add_special_tokens=False)
            word_ids = [i for i in word_ids if i < old_vocab]
            emb[tokenizer.convert_tokens_to_ids(tok)] = emb[word_ids].mean(dim=0)
        emb[tokenizer.pad_token_id] = emb[:old_vocab].mean(dim=0)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def label_token_ids(tokenizer) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(LABEL_TOKENS)
    if any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError("tokenizer has not been extended with the category tokens")
    return ids


@dataclass
class Prediction:
    text: str
    label: str
    confidence: float
    scores: dict[str, float]

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: -kv[1])[:k]


@torch.no_grad()
def classify_batch(model, tokenizer, texts: list[str], device: str = "cpu",
                   batch_size: int = 16) -> list[Prediction]:
    """The inference pipeline: prompt -> GPT-2 -> scores over category tokens."""
    model.eval()
    cand = torch.tensor(label_token_ids(tokenizer), device=device)
    out: list[Prediction] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer([prompt(t) for t in chunk], return_tensors="pt", padding=True,
                        truncation=True, max_length=model.config.n_positions - 1)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits                              # (B, T, vocab)
        last = enc["attention_mask"].sum(dim=1) - 1                # last real token (right padding)
        final = logits[torch.arange(len(chunk), device=device), last]
        probs = torch.softmax(final[:, cand], dim=-1).cpu()
        for text, p in zip(chunk, probs):
            scores = {label: round(float(v), 4) for label, v in zip(LABELS, p)}
            best = int(p.argmax())
            out.append(Prediction(text, LABELS[best], float(p[best]), scores))
    return out


class Sorter:
    """Loads the fine-tuned model once and classifies requests."""

    def __init__(self, directory: str | Path | None = None, device: str | None = None):
        self.dir = model_dir(directory)
        if not (self.dir / "config.json").exists():
            raise FileNotFoundError(
                f"No fine-tuned model in {self.dir}. Run `python -m app.cli train` first.")
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir)
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(self.dir).to(self.device)

    def classify(self, text: str) -> Prediction:
        return self.classify_many([text])[0]

    def classify_many(self, texts: list[str]) -> list[Prediction]:
        return classify_batch(self.model, self.tokenizer, texts, self.device)
