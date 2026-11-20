"""Milestone 1: the request categories and the labelled dataset.

Each category gets its own special token, e.g. ``<|venue_inquiry|>``. GPT-2
is trained to read a request and then emit exactly one of those tokens:

    Request: Is the rooftop terrace free on March 14th?
    Category: <|venue_inquiry|><|endoftext|>
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "requests.csv"

# label -> what it covers (shown in the CLI)
CATEGORIES: dict[str, str] = {
    "venue_inquiry": "venues: availability, capacity, site visits, access",
    "catering_request": "food & drink: menus, dietary needs, tastings, bar",
    "decor_request": "decor & theme: flowers, colours, lighting, props",
    "entertainment_booking": "entertainment: music, DJs, speakers, activities",
    "photo_video": "photography, video and livestreaming",
    "budget_quote": "money: quotes, deposits, invoices, discounts",
    "schedule_change": "dates & timing: reschedules, postponements, cancellations",
    "guest_logistics": "guests: transport, hotels, RSVPs, registration, signage",
}
LABELS = list(CATEGORIES)


def label_token(label: str) -> str:
    return f"<|{label}|>"


LABEL_TOKENS = [label_token(l) for l in LABELS]


def token_label(token: str) -> str:
    return token.removeprefix("<|").removesuffix("|>")


def pretty(label: str) -> str:
    """'venue_inquiry' -> 'venue inquiry'"""
    return label.replace("_", " ")


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def prompt(text: str) -> str:
    """What the model sees; it must continue with a category token."""
    return f"Request: {normalise(text)}\nCategory:"


def resolve_label(name: str) -> str:
    """Accept 'venue inquiry', 'Venue_Inquiry', '<|venue_inquiry|>' or a
    unique prefix like 'venue'."""
    key = token_label(name.strip()).lower().replace(" ", "_").replace("-", "_")
    if key in CATEGORIES:
        return key
    matches = [l for l in LABELS if key and l.startswith(key)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"unknown category {name!r}; choose from: {', '.join(LABELS)}")


@dataclass(frozen=True)
class Example:
    text: str
    label: str


def load_examples(path: Path = DATA_FILE) -> list[Example]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out, seen = [], set()
    for row in rows:
        text, label = normalise(row["text"]), row["label"].strip()
        if label not in CATEGORIES:
            raise ValueError(f"unknown label {label!r} in {path}")
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(Example(text, label))
    return out


def stratified_split(examples: list[Example], test_frac: float = 0.2,
                     seed: int = 13) -> tuple[list[Example], list[Example]]:
    """Hold out the same share of every category for testing."""
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)
    rng = random.Random(seed)
    train, test = [], []
    for label in LABELS:
        group = by_label.get(label, [])
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_frac)) if len(group) > 1 else 0
        test += group[:n_test]
        train += group[n_test:]
    rng.shuffle(train)
    return train, test
