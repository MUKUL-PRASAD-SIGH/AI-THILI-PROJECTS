import pytest
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from app.core import LABELS, Example
from app.llm import DEFAULT_BASE_MODEL
from app.train import train


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """A tiny, randomly initialised GPT-2 with the real GPT-2 tokenizer, so
    tests exercise the full pipeline in seconds without the 500 MB weights."""
    path = tmp_path_factory.mktemp("tiny-gpt2")
    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=64, n_layer=2, n_head=4)
    GPT2LMHeadModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_trained(tiny_base, tmp_path_factory):
    out = tmp_path_factory.mktemp("sorter")
    # a few, very separable examples so a tiny random model can learn them
    examples = [Example(f"note {i}, please help with {w}", label)
                for label, w in zip(LABELS, ["venue", "food", "flowers", "band", "photos",
                                             "price", "reschedule", "shuttle"])
                for i in range(6)]
    metrics = train(base=tiny_base, out_dir=out, epochs=40, lr=3e-3, batch_size=8,
                    device="cpu", examples=examples, log=lambda *_: None)
    return out, metrics
