import pytest
import torch
from torch import nn

from app import core, lora, train


def _ids():
    return torch.randint(1, 256, (2, 12), generator=torch.Generator().manual_seed(1))


def test_lora_linear_starts_as_identity_update():
    base = nn.Linear(10, 6)
    layer = lora.LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 10)
    assert torch.allclose(layer(x), base(x))
    assert layer.lora_A.shape == (4, 10) and layer.lora_B.shape == (6, 4)
    assert not base.weight.requires_grad


def test_lora_linear_math_matches_formula():
    base = nn.Linear(5, 3)
    layer = lora.LoRALinear(base, r=2, alpha=6)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(4, 5)
    expected = base(x) + (6 / 2) * x @ layer.lora_A.T @ layer.lora_B.T
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_inject_wraps_every_gpt2_projection(tiny_gpt2):
    names = lora.inject_lora(tiny_gpt2, lora.LoRAConfig(r=4, alpha=8))
    assert len(names) == 2 * 4  # 2 layers x (c_attn, attn.c_proj, c_fc, mlp.c_proj)
    trainable, total = lora.count_parameters(tiny_gpt2)
    expected = sum(m.lora_A.numel() + m.lora_B.numel() for m in lora.lora_layers(tiny_gpt2).values())
    assert trainable == expected < total
    assert all(("lora_" in n) == p.requires_grad for n, p in tiny_gpt2.named_parameters())


def test_injected_model_output_unchanged_before_training(tiny_gpt2):
    ids = _ids()
    before = tiny_gpt2(ids).logits
    lora.inject_lora(tiny_gpt2, lora.LoRAConfig(r=4, alpha=8, dropout=0.0))
    assert torch.allclose(tiny_gpt2(ids).logits, before, atol=1e-6)


def test_bad_target_raises(tiny_gpt2):
    with pytest.raises(ValueError, match="no layers matched"):
        lora.inject_lora(tiny_gpt2, lora.LoRAConfig(target_modules=["q_proj"]))


def _perturb(model):
    with torch.no_grad():
        for m in lora.lora_layers(model).values():
            m.lora_B.normal_(0, 0.05)


def test_merge_gives_same_outputs_and_plain_architecture(tiny_gpt2):
    lora.inject_lora(tiny_gpt2, lora.LoRAConfig(r=4, alpha=8, dropout=0.0))
    _perturb(tiny_gpt2)
    tiny_gpt2.eval()
    ids = _ids()
    adapted = tiny_gpt2(ids).logits
    lora.merge_lora(tiny_gpt2)
    assert not lora.lora_layers(tiny_gpt2)
    assert torch.allclose(tiny_gpt2(ids).logits, adapted, atol=1e-5)


def test_unmerge_restores_base_weight():
    base = nn.Linear(4, 4)
    w0 = base.weight.detach().clone()
    layer = lora.LoRALinear(base, r=2, alpha=4)
    nn.init.normal_(layer.lora_B)
    layer.merge()
    assert not torch.allclose(base.weight, w0)
    layer.unmerge()
    assert torch.allclose(base.weight, w0, atol=1e-6)


def test_adapter_save_load_roundtrip(tiny_gpt2, tmp_path):
    from transformers import GPT2LMHeadModel

    fresh = GPT2LMHeadModel(tiny_gpt2.config)
    fresh.load_state_dict(tiny_gpt2.state_dict())
    cfg = lora.LoRAConfig(r=4, alpha=8, dropout=0.0)
    lora.inject_lora(tiny_gpt2, cfg)
    _perturb(tiny_gpt2)
    lora.save_adapter(tiny_gpt2, cfg, tmp_path)
    lora.load_adapter(fresh, tmp_path)
    ids = _ids()
    tiny_gpt2.eval(), fresh.eval()
    assert torch.allclose(fresh(ids).logits, tiny_gpt2(ids).logits, atol=1e-6)


def test_encode_masks_prompt(tokenizer):
    ids, labels = train.encode_example(tokenizer, "Q?", "Yes.")
    n_prompt = len(tokenizer.encode(core.format_prompt("Q?")))
    assert labels[:n_prompt] == [train.IGNORE] * n_prompt
    assert labels[n_prompt:] == ids[n_prompt:]
    assert ids[-1] == tokenizer.eos_token_id


def test_training_lowers_loss_and_only_moves_lora(tiny_gpt2, tokenizer):
    examples = [core.Example("a", "D", "Fee?", "It costs $5."), core.Example("b", "D", "Hours?", "9 to 5.")]
    frozen = {n: p.detach().clone() for n, p in tiny_gpt2.named_parameters()}
    cfg = train.TrainConfig(epochs=40, batch_size=2, lr=3e-2, lora=lora.LoRAConfig(r=8, alpha=16, dropout=0.0))
    hist = train.train_lora(tiny_gpt2, tokenizer, examples, examples, cfg, log=lambda s: None)
    assert hist["best_val_loss"] < hist["initial_val_loss"] * 0.7
    for name, p in tiny_gpt2.named_parameters():
        if "lora_" not in name:
            assert torch.equal(p, frozen[name.replace(".base.", ".")]), name
