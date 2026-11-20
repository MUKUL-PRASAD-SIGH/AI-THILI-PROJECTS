import pytest

from app import core, corpus


def test_parse_and_format_roundtrip():
    exs = [core.Exchange("GUARD", "Halt?", "Who goes there."), core.Exchange("BARD", "Sing.", "La la.")]
    assert core.parse_corpus(core.format_corpus(exs)) == exs


def test_parse_skips_malformed_blocks():
    text = "PLAYER: Hi\nGUARD: Halt.\n\nonly one line\n\nGUARD: wrong order\nPLAYER: x\n"
    assert core.parse_corpus(text) == [core.Exchange("GUARD", "Hi", "Halt.")]


def test_prompt_format():
    assert core.format_prompt("guard", " Let me in. ") == "PLAYER: Let me in.\nGUARD:"


def test_clean_line_normalises_text():
    assert corpus.clean_line("“Hello” — traveller…   <b>") == '"Hello" - traveller... b'
    assert corpus.clean_line("50% (off) #1") == "50 off 1"


def test_generator_is_seeded_and_covers_every_role():
    a, b = corpus.generate_exchanges(80, seed=3), corpus.generate_exchanges(80, seed=3)
    assert a == b
    assert {e.role for e in a} == set(core.ROLES)
    assert all("{" not in e.player + e.reply for e in a)


def test_preprocess_drops_duplicates_and_unknown_roles():
    ex = core.Exchange("GUARD", "Hi", "Halt.")
    out = corpus.preprocess([ex, ex, core.Exchange("DRAGON", "Hi", "Roar."), core.Exchange("GUARD", "", "x")])
    assert out == [ex]


def test_seed_file_parses():
    seeds = corpus.load_seed()
    assert len(seeds) >= 30 and {e.role for e in seeds} == set(core.ROLES)


def test_committed_corpus_is_clean():
    exs = core.parse_corpus(corpus.CORPUS_FILE.read_text(encoding="utf-8"))
    assert len(exs) >= 2500
    assert corpus.preprocess(exs) == exs  # already clean and de-duplicated


def test_split_keeps_exchanges_whole():
    exs = corpus.generate_exchanges(100)
    train, val = core.split_exchanges(exs, 0.2)
    assert len(core.parse_corpus(train)) + len(core.parse_corpus(val)) == 100


def test_tokenizer_roundtrip_and_unknown(tmp_path):
    tok = core.CharTokenizer.from_text("PLAYER: hi\nGUARD: halt\n")
    assert tok.decode(tok.encode("GUARD: hi")) == "GUARD: hi"
    tok.save(tmp_path / "t.json")
    assert core.CharTokenizer.load(tmp_path / "t.json").chars == tok.chars
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tok.encode("zebra")
