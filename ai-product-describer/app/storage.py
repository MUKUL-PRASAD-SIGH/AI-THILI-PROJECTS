"""Where things live on disk: model checkpoints, the tokenizer, and the text
files users save generated descriptions to."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch

from app.model import GPT, GPTConfig
from app.tokenizer import BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = "describer.pt"
TOKENIZER = "tokenizer.json"
TRAIN_LOG = "train_log.json"


def model_dir(path: str | Path | None = None) -> Path:
    """MODEL_DIR from the argument, the environment/.env, or ./models."""
    if path is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        path = os.environ.get("MODEL_DIR", PROJECT_ROOT / "models")
    return Path(path)


def save_checkpoint(directory: Path, model: GPT, extra: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / CHECKPOINT
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(),
                **(extra or {})}, out)
    return out


def load_checkpoint(directory: Path, device: str = "cpu") -> tuple[GPT, dict]:
    ckpt_path = directory / CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained model in {directory}. Run `python -m app.cli train` first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, {k: v for k, v in ckpt.items() if k not in ("config", "state_dict")}


def save_tokenizer(directory: Path, tok: BPETokenizer) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tok.save(directory / TOKENIZER)
    return directory / TOKENIZER


def load_tokenizer(directory: Path) -> BPETokenizer:
    path = directory / TOKENIZER
    if not path.exists():
        raise FileNotFoundError(
            f"No tokenizer in {directory}. Run `python -m app.cli train` first.")
    return BPETokenizer.load(path)


def save_log(directory: Path, log: dict) -> None:
    (directory / TRAIN_LOG).write_text(json.dumps(log, indent=2), encoding="utf-8")


def save_descriptions(path: str | Path, product_header: str, descriptions: list[str]) -> Path:
    """Append generated descriptions to a plain text file a shop owner can
    paste from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# {product_header}  ({stamp})"]
    for i, d in enumerate(descriptions, 1):
        lines.append(f"{i}. {d}" if len(descriptions) > 1 else d)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    return path
