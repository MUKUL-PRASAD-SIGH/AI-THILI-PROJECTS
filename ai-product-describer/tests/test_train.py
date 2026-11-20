import random

import pytest

from app import storage
from app.corpus import build_corpus
from app.train import TrainConfig, lr_at, train


def tiny_cfg(**kw):
    base = dict(vocab_size=300, block_size=32, n_embd=32, n_head=2, n_layer=1,
                dropout=0.0, batch_size=8, max_steps=60, lr=3e-3, min_lr=3e-4,
                warmup=5, eval_every=30, eval_batches=2)
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("c") / "corpus.txt"
    build_corpus(n_generated=60, out=path)
    return path.read_text(encoding="utf-8")


def test_lr_schedule_warms_up_then_decays():
    cfg = tiny_cfg(max_steps=100, warmup=10)
    assert lr_at(0, cfg) < lr_at(9, cfg) == pytest.approx(cfg.lr)
    assert lr_at(100, cfg) == pytest.approx(cfg.min_lr)


def test_training_lowers_loss_and_saves(tmp_path, tiny_corpus):
    result = train(tiny_cfg(), corpus=tiny_corpus, out_dir=tmp_path, device="cpu",
                   log=lambda *_: None)
    h = result["history"]
    assert h[-1]["train_loss"] < h[0]["train_loss"] - 1.0
    model, meta = storage.load_checkpoint(tmp_path)
    tok = storage.load_tokenizer(tmp_path)
    assert model.cfg.vocab_size == tok.vocab_size
    assert "val_loss" in meta and (tmp_path / storage.TRAIN_LOG).exists()


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        storage.load_checkpoint(tmp_path)
