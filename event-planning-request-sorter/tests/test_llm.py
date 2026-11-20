import pytest
import torch

from app.core import LABEL_TOKENS, LABELS, pretty, prompt
from app.llm import PAD_TOKEN, build_model_and_tokenizer, classify_batch, label_token_ids


@pytest.fixture(scope="module")
def mt(tiny_base):
    return build_model_and_tokenizer(tiny_base)


def test_category_tokens_are_single_tokens(mt):
    model, tok = mt
    for t in LABEL_TOKENS:
        assert len(tok.encode(t)) == 1
    assert tok.pad_token == PAD_TOKEN
    assert model.get_input_embeddings().weight.shape[0] == len(tok)
    assert model.config.pad_token_id == tok.pad_token_id


def test_new_tokens_start_from_their_words(mt):
    model, tok = mt
    emb = model.get_input_embeddings().weight
    tid = tok.convert_tokens_to_ids("<|venue_inquiry|>")
    words = tok.encode(" " + pretty("venue_inquiry"))
    assert torch.allclose(emb[tid], emb[words].mean(0), atol=1e-6)


def test_prompt_tokenizes_without_category(mt):
    _, tok = mt
    ids = tok.encode(prompt("Is the hall free?"))
    assert not set(ids) & set(label_token_ids(tok))


def test_classify_batch_returns_distribution(mt):
    model, tok = mt
    preds = classify_batch(model, tok, ["short one", "a much longer request about the venue"])
    for p in preds:
        assert p.label in LABELS
        assert abs(sum(p.scores.values()) - 1) < 1e-3
        assert p.confidence == pytest.approx(max(p.scores.values()), abs=1e-4)


def test_padding_does_not_change_scores(mt):
    model, tok = mt
    alone = classify_batch(model, tok, ["short one"])[0]
    batched = classify_batch(model, tok, ["short one", "a much longer request " * 5])[0]
    assert alone.scores == pytest.approx(batched.scores, abs=1e-3)
