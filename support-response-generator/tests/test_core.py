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
