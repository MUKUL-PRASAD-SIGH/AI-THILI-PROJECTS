"""A tiny model trained for a few seconds on a slice of the corpus, shared by the model/quantization tests."""
import pytest
import torch

from app import core, corpus, train
from app.model import GPTConfig


@pytest.fixture(scope="session")
def tiny_data():
    text = corpus.CORPUS_FILE.read_text(encoding="utf-8")
    tok = core.CharTokenizer.from_text(text)
    train_text, val_text = core.split_exchanges(core.parse_corpus(text)[:400], 0.2)
    ids = lambda t: torch.tensor(tok.encode(t), dtype=torch.long)  # noqa: E731
    return tok, ids(train_text), ids(val_text)


@pytest.fixture(scope="session")
def tiny_trained(tiny_data):
    mcfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=32, dropout=0.0)
    tcfg = train.TrainConfig(steps=150, batch_size=16, lr=3e-3, warmup=10, eval_every=50)
    model, history = train.train(mcfg, tcfg, data=tiny_data, log=lambda s: None)
    return model, tiny_data[0], history
