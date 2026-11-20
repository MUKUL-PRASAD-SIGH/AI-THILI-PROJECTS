import random

from app.core import END, parse_record
from app.corpus import build_corpus, clean_text, make_product, parse_seed_file


def test_clean_text_strips_html_prices_and_promo():
    raw = "<p>Our <b>Lavender</b> candle’s lovely.</p> Only $24.99!!! SHOP NOW - limited stock!"
    assert clean_text(raw) == "Our Lavender candle's lovely."


def test_seed_file_is_cleaned_and_deduplicated():
    seeds = parse_seed_file()
    names = [p.name for p in seeds]
    assert len(names) == len(set(names))          # the duplicate listing is dropped
    for p in seeds:
        assert "<" not in p.description and "$" not in p.description
        assert p.notes and p.size


def test_generated_products_are_well_formed():
    rng = random.Random(0)
    for _ in range(50):
        p = make_product(rng)
        assert p.description.endswith((".", "!"))
        assert p.length in ("short", "medium", "long")


def test_build_corpus_is_reproducible(tmp_path):
    a = build_corpus(n_generated=40, out=tmp_path / "a.txt")
    b = build_corpus(n_generated=40, out=tmp_path / "b.txt")
    assert (tmp_path / "a.txt").read_text() == (tmp_path / "b.txt").read_text()
    text = (tmp_path / "a.txt").read_text(encoding="utf-8")
    records = [r + END for r in text.split(END) if r.strip()]
    assert len(records) == len(a)
    assert parse_record(records[0].strip()).name
