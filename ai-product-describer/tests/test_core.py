from app.core import (DESCRIPTION, END, PRODUCT, Product, finish_text,
                      length_bucket, parse_notes, parse_record)


def test_record_round_trip():
    p = Product(name="Midnight Fig", notes=["black fig", "cedarwood"], size="10 oz",
                length="short", description="Ripe fig. Dry cedar.")
    rec = p.to_record()
    assert rec.startswith(PRODUCT) and rec.endswith(END) and DESCRIPTION in rec
    back = parse_record(rec)
    assert back == p


def test_prompt_ends_at_description_token():
    assert Product(name="X").prompt().endswith(DESCRIPTION)


def test_length_bucket():
    assert length_bucket("One. Two.") == "short"
    assert length_bucket("One. Two. Three.") == "medium"
    assert length_bucket("A. B. C. D. E.") == "long"


def test_parse_notes():
    assert parse_notes("Lavender, vanilla & musk") == ["lavender", "vanilla", "musk"]


def test_finish_text_cuts_special_tokens_and_half_sentences():
    assert finish_text("Warm amber. Soft wood and<|end|>junk") == "Warm amber."
    assert finish_text("Fresh  pine.  Cozy and") == "Fresh pine."
