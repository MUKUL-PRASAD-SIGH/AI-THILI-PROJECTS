import torch

from app.model import GPT, Block, CausalSelfAttention, GPTConfig

CFG = GPTConfig(vocab_size=40, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(1, 10, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 4, 32)          # change only the "future"
    y2 = attn(x2)
    assert torch.allclose(y1[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y1[:, 6:], y2[:, 6:])


def test_attention_weights_are_lower_triangular_and_sum_to_one():
    attn = CausalSelfAttention(CFG).eval()
    _, w = attn(torch.randn(2, 7, 32), return_weights=True)
    assert w.shape == (2, 4, 7, 7)
    assert torch.all(torch.triu(w[0, 0], diagonal=1) == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 7))


def test_block_keeps_shape():
    assert Block(CFG)(torch.randn(3, 5, 32)).shape == (3, 5, 32)


def test_gpt_forward_and_loss():
    model = GPT(CFG)
    idx = torch.randint(0, 40, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 40)
    # an untrained model is close to uniform: loss ~ ln(vocab)
    assert abs(loss.item() - torch.log(torch.tensor(40.0)).item()) < 0.5


def test_whole_model_is_causal():
    torch.manual_seed(1)
    model = GPT(CFG).eval()
    a = torch.randint(0, 40, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 40
    assert torch.allclose(model(a)[0][:, :8], model(b)[0][:, :8], atol=1e-5)
