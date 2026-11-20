import torch

from app.core import END, SPECIAL_TOKENS, Product
from app.llm import MAX_TOKENS, Describer
from app.model import GPT, GPTConfig


def test_generate_appends_tokens_and_respects_block_size():
    model = GPT(GPTConfig(vocab_size=30, block_size=8, n_embd=16, n_head=2, n_layer=1))
    out = model.generate(torch.zeros(2, 3, dtype=torch.long), max_new_tokens=20)
    assert out.shape == (2, 23)          # longer than block_size: context is cropped


def test_generate_stops_at_stop_token():
    model = GPT(GPTConfig(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=1))
    with torch.no_grad():                # force the model to always predict token 7
        model.head.weight.zero_()
        model.ln_f.bias.zero_()
        model.head.weight[7] = 1.0
        model.ln_f.weight.fill_(1.0)
        model.ln_f.bias[0] = 5.0
    out = model.generate(torch.zeros(1, 2, dtype=torch.long), 50, temperature=0, stop_id=7)
    assert out.shape[1] == 3 and out[0, -1] == 7


def test_describer_generates_clean_variants(tiny_model_dir):
    d = Describer(tiny_model_dir, device="cpu")
    p = Product(name="Midnight Fig", notes=["black fig", "cedarwood"], size="10 oz",
                length="short")
    texts = d.describe(p, n=3, seed=1)
    assert 1 <= len(texts) <= 3 and len(set(texts)) == len(texts)
    for t in texts:
        assert t and not any(s in t for s in SPECIAL_TOKENS)


def test_seed_makes_output_repeatable(tiny_model_dir):
    d = Describer(tiny_model_dir, device="cpu")
    p = Product(name="Sea Breeze", notes=["sea salt"], length="medium")
    assert d.describe(p, seed=5) == d.describe(p, seed=5)


def test_raw_completion_is_bounded(tiny_model_dir):
    d = Describer(tiny_model_dir, device="cpu")
    out = d.complete(Product(name="X", length="long").prompt(), max_new_tokens=10, seed=0)[0]
    assert len(d.tok.encode(out)) <= 10
    assert MAX_TOKENS["short"] < MAX_TOKENS["medium"] < MAX_TOKENS["long"]
