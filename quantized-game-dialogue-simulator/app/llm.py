"""Load the trained dialogue GPT and make NPCs talk."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from app import core, storage, train
from app.model import GPT, GPTConfig, generate


def set_threads() -> int:
    n = int(os.getenv("TORCH_THREADS", "0") or 0)
    if n > 0:
        torch.set_num_threads(n)
    return torch.get_num_threads()


def checkpoint_path() -> Path:
    return storage.model_dir() / train.CHECKPOINT


def load_checkpoint(path: str | Path | None = None) -> tuple[GPT, core.CharTokenizer, dict]:
    path = Path(path or checkpoint_path())
    if not path.exists():
        raise SystemExit(f"no trained model at {storage.display(path)}: run `python -m app.cli train` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, core.CharTokenizer(ckpt["chars"]), ckpt.get("history", {})


def npc_reply(model, tok: core.CharTokenizer, role: str, player_line: str, max_new_tokens: int = 140,
              temperature: float = 0.8, top_k: int | None = 20, seed: int | None = 0) -> str:
    """The NPC's one-line answer to the player (generation stops at the end of the line)."""
    return npc_replies(model, tok, [(role, player_line)], max_new_tokens, temperature, top_k, seed)[0]


def npc_replies(model, tok: core.CharTokenizer, prompts: list[tuple[str, str]], max_new_tokens: int = 140,
                temperature: float = 0.8, top_k: int | None = 20, seed: int | None = 0) -> list[str]:
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    newline = tok.stoi["\n"]
    out = []
    for role, line in prompts:
        ids = torch.tensor([tok.encode(core.format_prompt(role, line))], dtype=torch.long)
        result = generate(model, ids, max_new_tokens, temperature, top_k, stop_token=newline, generator=gen,
                          block_size=model.cfg.block_size)
        text = tok.decode(result[0, ids.size(1):])
        out.append(text.split("\n")[0].strip())
    return out
