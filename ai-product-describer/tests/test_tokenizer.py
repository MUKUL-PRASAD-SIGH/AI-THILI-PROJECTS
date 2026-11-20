import torch

from app.core import DESCRIPTION, END, PRODUCT
from app.model import Embeddings, GPTConfig
from app.tokenizer import BPETokenizer

TEXT = ("Warm vanilla and soft sandalwood. Warm vanilla candle. "
        "Hand-poured candle with warm vanilla. ") * 20


def test_round_trip_any_text():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    for s in [TEXT[:80], "café crème ☕ — \U0001F56F", "", "  spaces\n\nand lines"]:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress_frequent_words():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    assert len(tok.encode(" vanilla")) == 1       # a frequent word becomes one token
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 3


def test_special_tokens_are_atomic():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    ids = tok.encode(f"{PRODUCT}name: X\n{DESCRIPTION}Warm vanilla.{END}")
    for s in (PRODUCT, DESCRIPTION, END):
        assert ids.count(tok.special[s]) == 1
    assert tok.vocab_size <= 300 and max(ids) < tok.vocab_size


def test_save_and_load(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=280)
    tok.save(tmp_path / "tok.json")
    again = BPETokenizer.load(tmp_path / "tok.json")
    assert again.encode(TEXT) == tok.encode(TEXT)
    assert again.special == tok.special


def test_embeddings_shape_and_positions():
    cfg = GPTConfig(vocab_size=50, block_size=8, n_embd=16, dropout=0.0)
    emb = Embeddings(cfg)
    x = torch.tensor([[3, 3, 3]])
    out = emb(x)
    assert out.shape == (1, 3, 16)
    # same token at different positions gets different vectors
    assert not torch.allclose(out[0, 0], out[0, 1])
