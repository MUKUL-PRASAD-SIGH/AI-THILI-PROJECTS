"""Reading raw transcripts and writing / reading the anonymized corpus."""
from __future__ import annotations

from pathlib import Path

from app.core import anonymize, preprocess

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw_interactions.txt"
CORPUS_PATH = ROOT / "data" / "corpus.txt"
ANON_TRANSCRIPT_PATH = ROOT / "data" / "anonymized_interactions.txt"


def read_turns(path: Path | str = RAW_PATH) -> list[tuple[str, str]]:
    """Read a transcript file into (speaker, text) pairs. Lines starting with # are skipped."""
    turns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        turns.append((speaker.strip().lower(), text.strip()))
    return turns


def build_corpus(turns: list[tuple[str, str]], speakers: tuple[str, ...] = ("agent",)) -> list[str]:
    """Anonymize and tokenize turns; return one space-joined sentence per line.

    By default only agent turns are kept, since suggestions should sound like an agent.
    Pass an empty tuple to keep every speaker.
    """
    lines = []
    for speaker, text in turns:
        if speakers and speaker not in speakers:
            continue
        for sentence in preprocess(text):
            lines.append(" ".join(sentence))
    return lines


def write_corpus(lines: list[str], path: Path | str = CORPUS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_corpus(path: Path | str = CORPUS_PATH) -> list[list[str]]:
    """Load a prepared corpus: one tokenized sentence per line."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.split() for line in text.splitlines() if line.strip()]


def write_anonymized_transcript(turns: list[tuple[str, str]], path: Path | str = ANON_TRANSCRIPT_PATH) -> Path:
    """Write the full transcript with PII removed (human-readable, for review)."""
    path = Path(path)
    path.write_text("\n".join(f"{s.capitalize()}: {anonymize(t)}" for s, t in turns) + "\n", encoding="utf-8")
    return path
