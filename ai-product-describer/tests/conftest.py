import pytest

from app.corpus import build_corpus
from app.train import TrainConfig, train


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A small GPT trained for a few seconds - enough to exercise generation."""
    root = tmp_path_factory.mktemp("model")
    corpus = root / "corpus.txt"
    build_corpus(n_generated=80, out=corpus)
    cfg = TrainConfig(vocab_size=320, block_size=48, n_embd=32, n_head=2, n_layer=2,
                      dropout=0.0, batch_size=16, max_steps=120, lr=3e-3, min_lr=3e-4,
                      warmup=5, eval_every=60, eval_batches=2)
    train(cfg, corpus=corpus.read_text(encoding="utf-8"), out_dir=root, device="cpu",
          log=lambda *_: None)
    return root
