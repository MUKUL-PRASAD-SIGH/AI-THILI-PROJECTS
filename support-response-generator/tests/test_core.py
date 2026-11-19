from app import storage
from app.core import anonymize, detokenize, preprocess, tokenize


# --- Milestone 1: preprocessing / anonymization -----------------------------

def test_anonymize_removes_pii():
    raw = ("Hi, my name is Priya Sharma. Email priya.sharma@example.com, call +1 (415) 555-0132, "
           "card 4111 1111 1111 1111, order #458921 and ORD-77213, see https://x.example.com/a")
    out = anonymize(raw)
    for secret in ["Priya", "Sharma", "priya.sharma", "555-0132", "4111", "458921", "77213", "x.example.com"]:
        assert secret not in out
    for tag in ["<NAME>", "<EMAIL>", "<PHONE>", "<CARD>", "<ORDER_ID>", "<URL>"]:
        assert tag in out


def test_greeting_names_are_removed_but_generic_greetings_kept():
    assert anonymize("Hi Rahul, thanks") == "Hi <NAME>, thanks"
    assert anonymize("Hi there, thanks") == "Hi there, thanks"


def test_tokenize_keeps_placeholders_and_punctuation():
    assert tokenize("Sent to <EMAIL>. I'm here!") == ["sent", "to", "<EMAIL>", ".", "i'm", "here", "!"]


def test_preprocess_splits_sentences():
    assert preprocess("Thank you.   Let me check!") == [["thank", "you", "."], ["let", "me", "check", "!"]]


def test_detokenize():
    assert detokenize(["thank", "you", ",", "rahul", "."]) == "thank you, rahul."


def test_prepared_corpus_has_no_raw_pii():
    lines = storage.build_corpus(storage.read_turns())
    text = "\n".join(lines)
    for secret in ["@", "priya", "rahul", "fatima", "4111", "458921"]:
        assert secret not in text
    assert len(lines) > 50


# --- Milestone 2: n-gram model with smoothing ------------------------------

import math

import pytest

from app.core import NGramModel

TOY = [["thank", "you", "for", "reaching", "out", "."],
       ["thank", "you", "for", "your", "patience", "."],
       ["thank", "you", "for", "your", "patience", "."]]


def test_distribution_sums_to_one():
    model = NGramModel(n=3, k=0.1).fit(TOY)
    for ctx in ([], ["thank"], ["thank", "you", "for"], ["never", "seen"]):
        assert math.isclose(sum(model.distribution(ctx).values()), 1.0, rel_tol=1e-9)


def test_smoothing_gives_unseen_words_nonzero_probability():
    model = NGramModel(n=3, k=0.1).fit(TOY)
    assert model.prob("reaching", ["thank", "you"]) > 0  # unseen after "thank you"
    assert model.prob("your", ["you", "for"]) > model.prob("reaching", ["you", "for"])


def test_unseen_history_backs_off_to_lower_orders():
    model = NGramModel(n=3, k=0.1).fit(TOY)
    # "zzz for" was never seen, but bigram "for -> your" was, so "your" should still win.
    dist = model.distribution(["zzz", "for"])
    assert max(dist, key=dist.get) == "your"


def test_perplexity_lower_on_training_data():
    model = NGramModel(n=3, k=0.1).fit(TOY)
    assert model.perplexity(TOY) < model.perplexity([["patience", "out", "reaching", "thank"]])


def test_round_trip_serialisation():
    model = NGramModel(n=3, k=0.5).fit(TOY)
    clone = NGramModel.from_dict(model.to_dict())
    assert clone.prob("your", ["you", "for"]) == model.prob("your", ["you", "for"])


def test_invalid_parameters():
    with pytest.raises(ValueError):
        NGramModel(n=0)
    with pytest.raises(ValueError):
        NGramModel(k=0)
