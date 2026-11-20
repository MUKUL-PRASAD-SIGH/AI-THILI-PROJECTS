"""Small file helpers: JSON Lines datasets, JSON reports and the answer-feedback log."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def model_dir() -> Path:
    """Where adapters, merged models and reports live (MODEL_DIR, default ``models/``)."""
    path = Path(os.getenv("MODEL_DIR", "models"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {n} is not valid JSON ({exc.msg})") from None
    return rows


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feedback_path() -> Path:
    return model_dir() / "feedback.jsonl"


def record_feedback(question: str, answer: str, rating: int, comment: str = "",
                    model: str = "", path: str | Path | None = None) -> dict:
    """Log an administrator's 1-5 rating of a generated answer."""
    if not 1 <= int(rating) <= 5:
        raise ValueError("rating must be between 1 and 5")
    row = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "rating": int(rating),
        "comment": comment.strip(),
        "model": model,
    }
    append_jsonl(path or feedback_path(), row)
    return row


def read_feedback(path: str | Path | None = None) -> list[dict]:
    path = Path(path or feedback_path())
    return read_jsonl(path) if path.exists() else []
