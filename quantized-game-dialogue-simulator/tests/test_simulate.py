import pytest

from app import quantize, report, simulate, storage, train


def test_fewer_bits_never_slower_and_smaller(tiny_trained):
    model, _, _ = tiny_trained
    rows = simulate.simulate_devices(model, {"fp32": 32, "int8": 8, "int4": 4, "int2": 2})
    for dev in simulate.DEVICES:
        r = {x["variant"]: x for x in rows if x["device"] == dev.name}
        assert r["fp32"]["model_mb"] > r["int8"]["model_mb"] > r["int4"]["model_mb"] > r["int2"]["model_mb"]
        assert r["fp32"]["ms_per_token"] >= r["int8"]["ms_per_token"] >= r["int4"]["ms_per_token"]


def test_roofline_matches_hand_calculation(tiny_trained):
    model, _, _ = tiny_trained
    p = simulate.variant_profile(model, 32, model.cfg.block_size)
    n_linear = sum(m.weight.numel() for m in quantize.linear_layers(model).values())
    assert p["linear_ops"] == 2 * n_linear
    dev = simulate.DEVICES[0]
    row = next(r for r in simulate.simulate_devices(model, {"fp32": 32}) if r["device"] == dev.name)
    mem = p["bytes_per_token"] / (dev.bandwidth_gbs * 1e9)
    comp = p["linear_ops"] / (dev.fp32_gflops * 1e9) + p["attn_ops"] / (dev.fp32_gflops * 1e9)
    assert row["ms_per_token"] == pytest.approx(max(mem, comp) * 1e3, abs=1e-4)


def test_benchmarks_cover_every_shape(tiny_trained, tiny_data):
    model, tok, _ = tiny_trained
    calib = quantize.calibration_batches(tiny_data[1], model.cfg.block_size, n_windows=8)
    variants = {"fp32": model, "dynamic": quantize.dynamic_quantize(model),
                "static": quantize.static_quantize(model, calib)}
    ops = simulate.benchmark_ops(variants, model.cfg.block_size, repeats=3)
    shapes = {(o["in"], o["out"]) for o in ops}
    assert len(shapes) == 5 and len(ops) == 10  # qkv, attn proj, fc, mlp proj, head; decode + prefill
    assert sum(o["per_model"] for o in ops if o["tokens"] == 1) == 9
    assert all(o[f"{k}_us"] > 0 for o in ops for k in variants)
    gen = simulate.benchmark_generation(variants, tok, [("GUARD", "Hi.")], new_tokens=5)
    assert set(gen) == set(variants) and all(g["tokens"] == 5 for g in gen.values())


def test_full_report_end_to_end(tiny_trained, tmp_path, monkeypatch):
    model, tok, history = tiny_trained
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(storage, "REPORTS", tmp_path / "reports")
    train.save_checkpoint(model, tok, history, tmp_path / "models" / train.CHECKPOINT)
    out = report.run(log=lambda s: None, samples_per_prompt=1, op_repeats=2, prompts=report.PROMPTS[15:18],
                     max_new_tokens=20, bench_tokens=5)
    v = out["variants"]
    assert list(v) == quantize.variant_names()
    assert v["fp32"]["greedy_same_as_fp32"] == 1.0 and v["fp32"]["size_ratio"] == 1.0
    assert v["int8"]["size_ratio"] > 1.5 and v["int2"]["val_loss"] >= v["int8"]["val_loss"]
    for f in ("results.json", "quantization.md", "samples.md", "tradeoff.png"):
        assert (tmp_path / "reports" / f).stat().st_size > 0
    assert "Simulated devices" in (tmp_path / "reports" / "quantization.md").read_text(encoding="utf-8")
