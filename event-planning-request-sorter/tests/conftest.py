import pytest
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from app.llm import DEFAULT_BASE_MODEL


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """A tiny, randomly initialised GPT-2 with the real GPT-2 tokenizer, so
    tests exercise the full pipeline in seconds without the 500 MB weights."""
    path = tmp_path_factory.mktemp("tiny-gpt2")
    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=32, n_layer=2, n_head=2)
    GPT2LMHeadModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)
