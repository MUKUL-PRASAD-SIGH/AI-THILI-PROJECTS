from collections import Counter

import pytest

from app.core import (CATEGORIES, LABEL_TOKENS, LABELS, load_examples, prompt,
                      resolve_label, stratified_split)


def test_dataset_covers_every_category():
    examples = load_examples()
    counts = Counter(e.label for e in examples)
    assert set(counts) == set(LABELS)
    assert min(counts.values()) >= 15
    assert len({e.text for e in examples}) == len(examples)


def test_stratified_split_keeps_every_category_in_test():
    train, test = stratified_split(load_examples())
    assert {e.label for e in test} == set(LABELS)
    assert not {e.text for e in train} & {e.text for e in test}
    assert 0.15 < len(test) / (len(train) + len(test)) < 0.25


def test_label_tokens_and_prompt():
    assert LABEL_TOKENS[0] == "<|venue_inquiry|>" and len(LABEL_TOKENS) == len(CATEGORIES)
    assert prompt("  Hi   there ’ ") == "Request: Hi there '\nCategory:"


@pytest.mark.parametrize("given", ["venue inquiry", "Venue_Inquiry", "<|venue_inquiry|>", "venue"])
def test_resolve_label(given):
    assert resolve_label(given) == "venue_inquiry"


@pytest.mark.parametrize("given", ["weather", ""])
def test_resolve_label_rejects_unknown(given):
    with pytest.raises(ValueError):
        resolve_label(given)
