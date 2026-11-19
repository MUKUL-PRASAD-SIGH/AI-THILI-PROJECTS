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


# ---------------------------------------------------------------------------
# Milestone 2: n-gram language model with smoothing
# ---------------------------------------------------------------------------

BOS = "<s>"
EOS = "</s>"


class NGramModel:
    """An n-gram language model with add-k smoothing and linear interpolation.

    For every order i = 1..n the add-k estimate is

        P_i(w | h) = (count(h, w) + k) / (count(h) + k * V)

    and the final probability mixes the orders whose history was seen in training:

        P(w | h) = sum_i lambda_i * P_i(w | h_i)      (lambdas renormalised)

    so an unseen trigram history falls back to the bigram and unigram estimates
    instead of collapsing to a uniform guess.
    """

    def __init__(self, n: int = 3, k: float = 0.1, lambdas: list[float] | None = None):
        if n < 1:
            raise ValueError("n must be >= 1")
        if k <= 0:
            raise ValueError("k must be > 0 (add-k smoothing)")
        self.n = n
        self.k = k
        # Default weights favour longer histories: e.g. n=3 -> [0.1, 0.3, 0.6] for uni/bi/tri.
        self.lambdas = lambdas or [i / sum(range(1, n + 1)) for i in range(1, n + 1)]
        if len(self.lambdas) != n:
            raise ValueError("need one lambda per order")
        # counts[i][history_tuple][word] for history length i (0 = unigram).
        self.counts: list[dict[tuple, dict[str, int]]] = [dict() for _ in range(n)]
        self.context_totals: list[dict[tuple, int]] = [dict() for _ in range(n)]
        self.vocab: set[str] = set()

    # -- training -----------------------------------------------------------
    def fit(self, sentences: list[list[str]]) -> "NGramModel":
        for sent in sentences:
            padded = [BOS] * (self.n - 1) + list(sent) + [EOS]
            for idx in range(self.n - 1, len(padded)):
                word = padded[idx]
                self.vocab.add(word)
                for order in range(self.n):
                    history = tuple(padded[idx - order:idx]) if order else ()
                    table = self.counts[order].setdefault(history, {})
                    table[word] = table.get(word, 0) + 1
                    self.context_totals[order][history] = self.context_totals[order].get(history, 0) + 1
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -- probabilities ------------------------------------------------------
    def _history(self, context: list[str]) -> list[str]:
        """Last n-1 tokens of the current sentence, left-padded with <s>."""
        return ([BOS] * (self.n - 1) + list(context))[-(self.n - 1):] if self.n > 1 else []

    def addk_prob(self, word: str, history: tuple) -> float:
        order = len(history)
        total = self.context_totals[order].get(history, 0)
        count = self.counts[order].get(history, {}).get(word, 0)
        return (count + self.k) / (total + self.k * self.vocab_size)

    def prob(self, word: str, context: list[str]) -> float:
        """Smoothed, interpolated P(word | context)."""
        if not self.vocab:
            raise RuntimeError("model is not trained")
        hist = self._history(context)
        weighted, weight_sum = 0.0, 0.0
        for order in range(self.n):
            history = tuple(hist[len(hist) - order:]) if order else ()
            if order and history not in self.context_totals[order]:
                continue  # unseen history: skip this order and renormalise
            weighted += self.lambdas[order] * self.addk_prob(word, history)
            weight_sum += self.lambdas[order]
        return weighted / weight_sum

    def distribution(self, context: list[str]) -> dict[str, float]:
        """P(w | context) for every word in the vocabulary (sums to 1)."""
        return {w: self.prob(w, context) for w in self.vocab}

    def perplexity(self, sentences: list[list[str]]) -> float:
        import math

        log_sum, count = 0.0, 0
        for sent in sentences:
            context: list[str] = []
            for word in list(sent) + [EOS]:
                log_sum += math.log(self.prob(word, context))
                context.append(word)
                count += 1
        return math.exp(-log_sum / max(count, 1))

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n": self.n, "k": self.k, "lambdas": self.lambdas,
            "counts": [[[list(h), t] for h, t in table.items()] for table in self.counts],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(n=data["n"], k=data["k"], lambdas=data["lambdas"])
        for order, table in enumerate(data["counts"]):
            for history, words in table:
                history = tuple(history)
                model.counts[order][history] = dict(words)
                model.context_totals[order][history] = sum(words.values())
                model.vocab.update(words)
        return model
