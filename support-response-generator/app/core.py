"""Core logic: text preprocessing / anonymization and the n-gram language model."""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Milestone 1: preprocessing and anonymization
# ---------------------------------------------------------------------------

# Placeholder tokens that replace personal data. They survive tokenization
# as single tokens so the model can still learn "sent to <EMAIL>" etc.
PLACEHOLDERS = ("<EMAIL>", "<URL>", "<CARD>", "<PHONE>", "<ORDER_ID>", "<TRACKING>", "<ACCOUNT>", "<NAME>")

_NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"

# Order matters: more specific patterns run first.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "<EMAIL>"),
    (re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I), "<URL>"),
    (re.compile(r"\b(?:\d[ -]?){12,15}\d\b"), "<CARD>"),
    (re.compile(r"\b(card ending (?:in )?)\d{4}\b", re.I), r"\1<CARD>"),
    (re.compile(r"\b1Z[0-9A-Z]{16}\b"), "<TRACKING>"),
    (re.compile(r"(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)\s?\d{3}[ -]?\d{4}|\b\d{3}[ -]\d{3}[ -]\d{4}\b|\b\d{5}[ -]\d{5}\b)"),
     "<PHONE>"),
    (re.compile(r"#\d{4,}\b|\bORD-\d+\b", re.I), "<ORDER_ID>"),
    (re.compile(r"\b(account(?: number)?:?\s+)\d{6,}\b", re.I), r"\1<ACCOUNT>"),
    # Names introduced by the speaker: "my name is Priya Sharma", "I'm Arjun".
    (re.compile(rf"\b(my name is|this is|I'm|I am)\s+{_NAME}\b(?=[\s,.!?])"), r"\1 <NAME>"),
    (re.compile(rf"\b(Thanks|Thank you),\s+{_NAME}(?= here\b)"), r"\1, <NAME>"),
    # Greetings addressed to a person: "Hi Priya," / "Dear Fatima,"
    (re.compile(rf"\b(Hi|Hello|Hey|Dear)\s+(?!there\b|team\b|all\b){_NAME}(?=\s*[,!.])"), r"\1 <NAME>"),
]


def anonymize(text: str) -> str:
    """Replace personal data (emails, phones, cards, ids, names) with placeholders."""
    for pattern, repl in _PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


_PLACEHOLDER_RE = re.compile(r"<[A-Z_]+>")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]")
SENTENCE_END = {".", "!", "?"}
PUNCTUATION = {".", ",", "!", "?", ";", ":"}


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word, number, punctuation and placeholder tokens."""
    tokens: list[str] = []
    pos = 0
    for m in _PLACEHOLDER_RE.finditer(text):
        tokens.extend(_WORD_RE.findall(text[pos:m.start()].lower().replace("’", "'")))
        if m.group(0) in PLACEHOLDERS:
            tokens.append(m.group(0))
        pos = m.end()
    tokens.extend(_WORD_RE.findall(text[pos:].lower().replace("’", "'")))
    return tokens


def split_sentences(tokens: list[str]) -> list[list[str]]:
    """Split a token list into sentences at . ! ? (punctuation is kept)."""
    sentences, current = [], []
    for tok in tokens:
        current.append(tok)
        if tok in SENTENCE_END:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def preprocess(text: str) -> list[list[str]]:
    """Clean, anonymize and tokenize one utterance into sentences of tokens."""
    text = re.sub(r"\s+", " ", text).strip()
    return split_sentences(tokenize(anonymize(text)))


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into readable text (no space before punctuation)."""
    out = ""
    for tok in tokens:
        if not out:
            out = tok
        elif tok in PUNCTUATION:
            out += tok
        else:
            out += " " + tok
    return out
