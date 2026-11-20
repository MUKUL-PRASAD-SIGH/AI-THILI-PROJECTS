"""The FAQ dataset: loading, validation, prompt format and train / held-out splits.

Each FAQ record (one JSON object per line in ``data/faqs.jsonl``) has:

* ``id``          unique id such as ``dmv-01``
* ``department``  the department whose guidelines the answer comes from
* ``question``    the canonical wording of the question
* ``paraphrases`` other ways residents ask the same thing (at least one)
* ``answer``      the approved answer, taken from the department's guidelines
* ``facts``       short strings the answer must contain to be correct
                  (fees, deadlines, phone numbers...). Used for evaluation.

Training sees the canonical question and every paraphrase except the last one.
The last paraphrase of every FAQ is held out, so evaluation measures whether the
model answers a wording it has never seen, not whether it can echo its training set.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from app import storage

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "faqs.jsonl"

PROMPT_HEADER = "Riverton City Services FAQ"
REQUIRED = ("id", "department", "question", "paraphrases", "answer", "facts")


@dataclass(frozen=True)
class FAQ:
    id: str
    department: str
    question: str
    answer: str
    paraphrases: tuple[str, ...] = field(default_factory=tuple)
    facts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_questions(self) -> tuple[str, ...]:
        return (self.question, *self.paraphrases)


@dataclass(frozen=True)
class Example:
    """One (question, answer) pair to train on or evaluate."""

    faq_id: str
    department: str
    question: str
    answer: str
    facts: tuple[str, ...] = ()


def clean_text(text: str) -> str:
    """Collapse whitespace and trim; keeps the casing and punctuation of the guideline."""
    return re.sub(r"\s+", " ", str(text)).strip()


def validate_record(record: dict) -> FAQ:
    """Turn a raw dict into an FAQ, raising ValueError with a readable reason if it is malformed."""
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"FAQ {record.get('id', '?')} is missing {', '.join(missing)}")
    faq_id = clean_text(record["id"])
    if not faq_id:
        raise ValueError("FAQ id is empty")
    for key in ("department", "question", "answer"):
        if not clean_text(record[key]):
            raise ValueError(f"FAQ {faq_id} has an empty {key}")
    paraphrases = record["paraphrases"]
    facts = record["facts"]
    if not isinstance(paraphrases, list) or not [p for p in paraphrases if clean_text(p)]:
        raise ValueError(f"FAQ {faq_id} needs at least one paraphrase (the held-out test wording)")
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"FAQ {faq_id} needs at least one fact to check answers against")
    answer = clean_text(record["answer"])
    for fact in facts:
        if clean_text(fact).lower() not in answer.lower():
            raise ValueError(f"FAQ {faq_id}: fact {fact!r} does not appear in its answer")
    return FAQ(
        id=faq_id,
        department=clean_text(record["department"]),
        question=clean_text(record["question"]),
        answer=answer,
        paraphrases=tuple(clean_text(p) for p in paraphrases if clean_text(p)),
        facts=tuple(clean_text(f) for f in facts),
    )


def load_faqs(path: str | Path = DEFAULT_DATA) -> list[FAQ]:
    """Load and validate every FAQ. Duplicate ids or duplicate question wordings are errors."""
    faqs = [validate_record(r) for r in storage.read_jsonl(path)]
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for faq in faqs:
        if faq.id in seen_ids:
            raise ValueError(f"duplicate FAQ id {faq.id}")
        seen_ids.add(faq.id)
        for q in faq.all_questions:
            key = q.lower()
            if key in seen_questions:
                raise ValueError(f"question {q!r} appears in both {seen_questions[key]} and {faq.id}")
            seen_questions[key] = faq.id
    return faqs


def format_prompt(question: str) -> str:
    """The text the model sees before it writes an answer."""
    return f"{PROMPT_HEADER}\nQuestion: {clean_text(question)}\nAnswer:"


def format_target(answer: str) -> str:
    """The text the model learns to write after the prompt (the EOS token is appended at tokenisation)."""
    return " " + clean_text(answer)


def split_examples(faqs: list[FAQ]) -> tuple[list[Example], list[Example]]:
    """Train on the canonical question plus all but the last paraphrase; hold out the last paraphrase."""
    train: list[Example] = []
    held_out: list[Example] = []
    for faq in faqs:
        *seen, unseen = faq.all_questions
        for q in seen:
            train.append(Example(faq.id, faq.department, q, faq.answer, faq.facts))
        held_out.append(Example(faq.id, faq.department, unseen, faq.answer, faq.facts))
    return train, held_out


def seen_examples(faqs: list[FAQ]) -> list[Example]:
    """The canonical wording of every FAQ (part of training), to check the answers were learnt."""
    return [Example(f.id, f.department, f.question, f.answer, f.facts) for f in faqs]


def shuffled(items: list, seed: int) -> list:
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def department_counts(faqs: list[FAQ]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for faq in faqs:
        counts[faq.department] = counts.get(faq.department, 0) + 1
    return counts
