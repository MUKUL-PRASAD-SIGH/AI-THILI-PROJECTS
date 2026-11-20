"""Milestone 5: save classified requests (and corrections) for review.

Results are appended to a structured file. The format follows the
extension: ``.jsonl`` (default, one JSON object per line) or ``.csv``
(opens straight in a spreadsheet). Each row records when, what, which
category, how confident, and the runner-up categories.

Corrections are kept in their own JSONL file so they can be fed back into
training with ``python -m app.cli train --with-corrections``.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core import Example, load_examples, normalise, resolve_label
from app.llm import Prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_RESULTS = OUTPUT_DIR / "classified.jsonl"
DEFAULT_CORRECTIONS = OUTPUT_DIR / "corrections.jsonl"

FIELDS = ["timestamp", "request", "category", "confidence", "runner_up",
          "runner_up_confidence", "source"]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def to_row(pred: Prediction, source: str = "cli") -> dict:
    top = pred.top(2)
    runner = top[1] if len(top) > 1 else ("", 0.0)
    return {"timestamp": _now(), "request": pred.text, "category": pred.label,
            "confidence": round(pred.confidence, 4), "runner_up": runner[0],
            "runner_up_confidence": round(runner[1], 4), "source": source}


def save_predictions(path: str | Path, preds: list[Prediction], source: str = "cli") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [to_row(p, source) for p in preds]
    if path.suffix.lower() == ".csv":
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    else:
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_saved(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_requests(path: str | Path) -> list[str]:
    """Batch input: a .txt file (one request per line) or a .csv with a
    'text' or 'request' column."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            col = next((c for c in (reader.fieldnames or []) if c.lower() in ("text", "request")),
                       None)
            if col is None:
                raise ValueError(f"{path} needs a 'text' or 'request' column")
            texts = [row[col] for row in reader]
    else:
        texts = path.read_text(encoding="utf-8").splitlines()
    return [normalise(t) for t in texts if t.strip() and not t.lstrip().startswith("#")]


def save_correction(text: str, predicted: str, corrected: str,
                    path: str | Path = DEFAULT_CORRECTIONS) -> Path:
    corrected = resolve_label(corrected)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": _now(), "request": normalise(text),
                            "predicted": predicted, "corrected": corrected},
                           ensure_ascii=False) + "\n")
    return path


def load_corrections(path: str | Path = DEFAULT_CORRECTIONS) -> list[Example]:
    """Corrections as training examples; the latest correction for a text wins,
    and texts already in the labelled dataset are skipped."""
    latest: dict[str, Example] = {}
    for row in load_saved(path):
        latest[row["request"].lower()] = Example(row["request"], row["corrected"])
    known = {e.text.lower() for e in load_examples()}
    return [e for k, e in latest.items() if k not in known]
