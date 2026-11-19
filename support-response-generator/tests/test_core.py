import math

import pytest

from app import storage
from app.core import (NGramModel, anonymize, append_words, complete, current_context, detokenize, generate,
                      preprocess, suggest_next, tokenize)


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


# --- Milestone 3: suggestions and continuation ------------------------------

@pytest.fixture(scope="module")
def corpus_model():
    return NGramModel().fit(storage.load_corpus())


def test_current_context_uses_only_the_open_sentence():
    assert current_context("Hi Priya, thank you. Let me") == ["let", "me"]
    assert current_context("Thank you.") == []
    assert current_context("Hi Priya,") == ["hi", "<NAME>", ","]


def test_suggest_next_returns_ranked_top_k(corpus_model):
    sugg = suggest_next(corpus_model, ["thank", "you", "for"], k=3)
    assert len(sugg) == 3
    assert [w for w, _ in sugg][0] in {"reaching", "contacting", "your"}
    assert sugg[0][1] >= sugg[1][1] >= sugg[2][1]


def test_suggestions_are_accurate_on_common_phrases(corpus_model):
    cases = {
        "Let me check the": "status",
        "The refund will be processed within 5 to 7 business": "days",
        "Is there anything else I can help you": "with",
        "Thank you for your": "patience",
    }
    for draft, expected in cases.items():
        assert suggest_next(corpus_model, current_context(draft), k=1)[0][0] == expected


def test_generate_greedy_completes_a_sentence(corpus_model):
    words = generate(corpus_model, current_context("Let me check the"), max_words=15)
    assert words[0] == "status" and words[-1] == "."


def test_generate_sampling_is_reproducible_with_seed(corpus_model):
    a = generate(corpus_model, [], strategy="sample", seed=7)
    b = generate(corpus_model, [], strategy="sample", seed=7)
    assert a == b and len(a) > 0


def test_generate_respects_max_words(corpus_model):
    assert len(generate(corpus_model, [], max_words=3, stop_at_sentence_end=False)) <= 3


def test_append_words_spacing():
    assert append_words("Thank you", ["for", "waiting", "."]) == "Thank you for waiting."
    assert append_words("", ["hello", ","]) == "Hello,"
    assert append_words("Done.", ["i'm", "sorry", ",", "i", "will", "help", "<NAME>"]) == "Done. I'm sorry, I will help <NAME>"


def test_complete_extends_draft(corpus_model):
    out = complete(corpus_model, "Please check your inbox and")
    assert out.startswith("Please check your inbox and your spam folder")


def test_default_smoothing_beats_heavier_smoothing_on_held_out_data():
    import random

    sents = storage.load_corpus()
    random.Random(0).shuffle(sents)
    held_out, train = sents[:14], sents[14:]
    tuned = NGramModel().fit(train).perplexity(held_out)
    laplace = NGramModel(k=1.0).fit(train).perplexity(held_out)
    assert tuned < laplace
