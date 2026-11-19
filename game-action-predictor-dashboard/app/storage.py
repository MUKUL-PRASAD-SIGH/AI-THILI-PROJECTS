"""Where files live, and reading / writing them."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "play_by_play.csv"
SEQUENCES_PATH = ROOT / "data" / "possessions.jsonl"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "action-lm"


def write_raw(df: pd.DataFrame, path: Path | str = RAW_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"team": "string", "opponent": "string"})


def write_sequences(possessions, path: Path | str = SEQUENCES_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in possessions:
            f.write(json.dumps({"game_id": p.game_id, "team": p.team, "number": p.number, "actions": p.actions}) + "\n")
    return path


def read_sequences(path: Path | str = SEQUENCES_PATH):
    from app.core import Possession

    with Path(path).open(encoding="utf-8") as f:
        return [Possession(**json.loads(line)) for line in f if line.strip()]


def write_report(name: str, text: str, data: dict | None = None) -> Path:
    """Write reports/<name>.md (and reports/<name>.json when `data` is given)."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    if data is not None:
        (REPORTS / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
