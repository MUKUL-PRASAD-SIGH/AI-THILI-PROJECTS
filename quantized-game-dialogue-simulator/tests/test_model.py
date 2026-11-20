import math

import pytest
import torch

from app import llm, train
from app.model import GPT, GPTConfig, LinearSlot, generate


def _model(**kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(**{"vocab_size": 20, "block_size": 16, "n_layer": 2, "n_head": 2, "n_embd": 16,
                            "dropout": 0.0, **kw})).eval()


def test_shapes_and_initial_loss():
    m = _model()
    x = torch.randint(0, 20, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 20)
    assert abs(float(loss) - math.log(20)) < 0.3  # near-uniform at init


def test_attention_is_causal():
    m = _model()
    x = torch.randint(0, 20, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 20  # change only future tokens
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_every_projection_is_a_linear_slot():
    m = _model()
    slots = [n for n, mod in m.named_modules() if isinstance(mod, LinearSlot)]
    assert len(slots) == 2 * 4 + 1  # qkv, attn proj, fc, mlp proj per layer + head
    assert sum(isinstance(mod, torch.nn.Linear) for mod in m.modules()) == len(slots)


def test_too_long_input_raises():
    with pytest.raises(ValueError, match="block_size"):
        _model()(torch.zeros(1, 17, dtype=torch.long))


def test_generate_greedy_is_deterministic_and_stops():
    m = _model()
    start = torch.zeros(1, 3, dtype=torch.long)
    a = generate(m, start, 10, temperature=0)
    assert torch.equal(a, generate(m, start, 10, temperature=0))
    assert a.shape == (1, 13)
    stop = int(a[0, 3])
    assert generate(m, start, 10, temperature=0, stop_token=stop).shape == (1, 4)


def test_training_learns_dialogue(tiny_trained):
    _, _, history = tiny_trained
    first, best = history["evals"][0]["val_loss"], history["best_val_loss"]
    assert best < first - 1.0


def test_checkpoint_roundtrip_and_reply(tiny_trained, tmp_path):
    model, tok, history = tiny_trained
    path = tmp_path / "m.pt"
    train.save_checkpoint(model, tok, history, path)
    loaded, tok2, _ = llm.load_checkpoint(path)
    x = torch.tensor([tok.encode("PLAYER: Hello.\nGUARD:")])
    assert torch.allclose(loaded(x)[0], model(x)[0])
    reply = llm.npc_reply(loaded, tok2, "GUARD", "Hello.", max_new_tokens=40, seed=1)
    assert "\n" not in reply and reply == llm.npc_reply(model, tok, "GUARD", "Hello.", max_new_tokens=40, seed=1)
