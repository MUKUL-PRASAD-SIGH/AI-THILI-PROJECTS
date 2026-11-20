import pytest
import torch
from torch.ao.nn.quantized import Linear as QLinear
from torch.ao.nn.quantized.dynamic import Linear as DynLinear

from app import llm, quantize, train


@pytest.mark.parametrize("bits", [8, 6, 4, 3, 2])
def test_pack_unpack_roundtrip(bits):
    codes = torch.randint(0, 2 ** bits, (7, 13), dtype=torch.uint8)
    data = quantize.pack_bits(codes, bits)
    assert len(data) == -(-codes.numel() * bits // 8)  # ceil(n * bits / 8) bytes
    assert torch.equal(quantize.unpack_bits(data, bits, codes.shape), codes)


@pytest.mark.parametrize("bits", [8, 4, 2])
def test_quantize_tensor_error_bound(bits):
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    qt = quantize.quantize_tensor(w, bits)
    assert int(qt.codes.max()) <= 2 ** bits - 1
    err = (qt.dequantize() - w).abs()
    assert torch.all(err <= qt.scale / 2 + 1e-5)  # at most half a step per row
    assert torch.equal(quantize.quantize_tensor(torch.zeros(2, 3), bits).dequantize(), torch.zeros(2, 3))


def test_more_bits_less_error():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    errs = [(quantize.quantize_tensor(w, b).dequantize() - w).pow(2).mean() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_dynamic_and_static_use_int8_kernels(tiny_trained, tiny_data):
    model, tok, _ = tiny_trained
    dyn = quantize.dynamic_quantize(model)
    assert sum(isinstance(m, DynLinear) for m in dyn.modules()) == 9
    calib = quantize.calibration_batches(tiny_data[1], model.cfg.block_size, n_windows=16)
    stat = quantize.static_quantize(model, calib)
    assert sum(isinstance(m, QLinear) for m in stat.modules()) == 9
    assert not any(isinstance(m, torch.nn.Linear) for m in stat.modules())
    assert any(isinstance(m, torch.nn.Linear) for m in model.modules())  # original untouched


def test_quantized_models_stay_close_to_float(tiny_trained, tiny_data):
    model, tok, _ = tiny_trained
    val = tiny_data[2]
    bs = model.cfg.block_size
    base = train.eval_loss(model, val, bs)
    calib = quantize.calibration_batches(tiny_data[1], bs, n_windows=16)
    for q in (quantize.dynamic_quantize(model), quantize.static_quantize(model, calib),
              quantize.kbit_quantize(model, 8)[0]):
        assert abs(train.eval_loss(q, val, bs) - base) < 0.05
    assert train.eval_loss(quantize.kbit_quantize(model, 2)[0], val, bs) > base


def test_quantized_model_generates(tiny_trained):
    model, tok, _ = tiny_trained
    reply = llm.npc_reply(quantize.dynamic_quantize(model), tok, "GUARD", "Hello.", max_new_tokens=20)
    assert isinstance(reply, str) and "\n" not in reply


def test_build_all_sizes_and_kbit_reload(tiny_trained, tiny_data, tmp_path):
    model, tok, _ = tiny_trained
    v = quantize.build_all(model, tok, tiny_data[1], tmp_path, log=lambda s: None)
    sizes = {k: d["bytes"] for k, d in v.items()}
    assert sizes["dynamic"] < sizes["fp32"] and sizes["static"] < sizes["fp32"]
    assert sizes["int8"] > sizes["int4"] > sizes["int2"]
    # the k-bit file is the packed weights plus float extras, with little container overhead
    packed = sum(-(-m.weight.numel() * 4 // 8) for m in quantize.linear_layers(model).values())
    extras = quantize.size_breakdown(model)["total"] - quantize.size_breakdown(model)["linear_weights"]
    assert sizes["int4"] < (packed + extras) * 1.2 + 20_000
    reloaded, _ = quantize.load_kbit(tmp_path / "int4.pt")
    x = torch.tensor([tok.encode("PLAYER: Hi.\nGUARD:")])
    assert torch.allclose(reloaded(x)[0], v["int4"]["model"](x)[0], atol=1e-5)


def test_size_breakdown_adds_up(tiny_trained):
    model, _, _ = tiny_trained
    b = quantize.size_breakdown(model)
    assert b["linear_weights"] + b["linear_biases"] + b["embeddings"] + b["layernorms"] == b["total"]
    assert b["total"] == model.num_params() * 4


def test_build_all_never_overwrites_training_checkpoint(tiny_trained, tiny_data, tmp_path, monkeypatch):
    model, tok, history = tiny_trained
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    ckpt = tmp_path / train.CHECKPOINT
    train.save_checkpoint(model, tok, history, ckpt)
    before = ckpt.read_bytes()
    quantize.build_all(model, tok, tiny_data[1], log=lambda s: None)
    assert ckpt.read_bytes() == before
    assert (tmp_path / "variants" / "fp32.pt").exists()
