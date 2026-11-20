"""Load the FAQ model (pre-trained, base + LoRA adapter, or merged) and generate answers.

Everything runs locally with GPT-2; no API key or hosted model is involved.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from app import core, lora, storage

DEFAULT_BASE = "openai-community/gpt2"
STOP_STRINGS = ("\nQuestion:", "\n", "Question:")


def base_model_name() -> str:
    return os.getenv("BASE_MODEL", DEFAULT_BASE)


def pick_device(requested: str | None = None) -> str:
    requested = requested or os.getenv("DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def adapter_dir() -> Path:
    return storage.model_dir() / "adapter"


def merged_dir() -> Path:
    return storage.model_dir() / "merged"


def load_tokenizer(name: str | Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(name))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so batched generation continues right after each prompt
    return tok


def load_base(name: str | Path | None = None, device: str | None = None):
    """The pre-trained model with no adaptation. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM

    name = name or base_model_name()
    model = AutoModelForCausalLM.from_pretrained(str(name))
    model.to(pick_device(device)).eval()
    return model, load_tokenizer(name)


def load_adapted(path: str | Path | None = None, device: str | None = None):
    """Pre-trained model with the trained LoRA adapter injected (not merged)."""
    path = Path(path or adapter_dir())
    meta = lora.read_adapter_config(path)
    model, tok = load_base(meta.get("base_model", base_model_name()), device="cpu")
    lora.load_adapter(model, path)
    model.to(pick_device(device)).eval()
    return model, tok


def load_merged(path: str | Path | None = None, device: str | None = None):
    """The deployable model: adapter folded into the weights, plain GPT-2 architecture."""
    return load_base(Path(path or merged_dir()), device)


def available_models() -> list[str]:
    kinds = []
    if (merged_dir() / "config.json").exists():
        kinds.append("merged")
    if (adapter_dir() / lora.ADAPTER_WEIGHTS).exists():
        kinds.append("adapter")
    kinds.append("base")
    return kinds


def load_model(kind: str = "auto", device: str | None = None):
    """Load by name: 'merged', 'adapter', 'base', or 'auto' (best one that exists)."""
    if kind == "auto":
        kind = available_models()[0]
    loaders = {"merged": load_merged, "adapter": load_adapted, "base": load_base}
    if kind not in loaders:
        raise ValueError(f"unknown model {kind!r}; choose from {', '.join(loaders)}")
    model, tok = loaders[kind](device=device)
    return model, tok, kind


def clean_answer(text: str) -> str:
    """Cut a generation at the first sign the model has moved on from the answer."""
    cut = len(text)
    for stop in STOP_STRINGS:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


@torch.no_grad()
def generate_answers(model, tokenizer, questions: list[str], max_new_tokens: int = 80,
                     temperature: float = 0.0, top_p: float = 0.9, batch_size: int = 16,
                     seed: int | None = 0) -> list[str]:
    """Answer several questions. temperature 0 means greedy decoding (deterministic)."""
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    answers: list[str] = []
    for i in range(0, len(questions), batch_size):
        prompts = [core.format_prompt(q) for q in questions[i: i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        sample = temperature > 0
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=top_p if sample else None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0,
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for row in new_tokens:
            answers.append(clean_answer(tokenizer.decode(row, skip_special_tokens=True)))
    return answers


def generate_answer(model, tokenizer, question: str, **kwargs) -> str:
    return generate_answers(model, tokenizer, [question], **kwargs)[0]
