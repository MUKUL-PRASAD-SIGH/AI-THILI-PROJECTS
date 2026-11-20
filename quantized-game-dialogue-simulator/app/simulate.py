"""Inference speed: measured on this machine, and simulated for resource-constrained devices.

1. ``benchmark_ops``  times the operation that dominates a GPT forward pass, ``y = x W^T + b``,
   for every Linear shape in the model: the float32 weight vs PyTorch's int8 dynamic and
   int8 static kernels. Both decode-shaped (1 token) and prefill-shaped (a full context)
   inputs are timed, since they stress memory and compute differently.
2. ``benchmark_generation``  times whole NPC replies (tokens per second, batch 1).
3. ``simulate_devices``  estimates per-token decode latency on target devices with a
   roofline model. Generating one token reads every weight once and does ~2 FLOPs per
   weight, so

       time/token = max(weight bytes / memory bandwidth,  ops / compute throughput)

   with int8 throughput for int8 (and lower-bit, unpacked to int8) weights and float32
   throughput for float weights. Device numbers are stated assumptions, not measurements;
   change ``DEVICES`` to match real hardware.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from app import llm
from app.model import LinearSlot


def _time(fn, repeats: int = 30, warmup: int = 5, inner: int = 10) -> float:
    """Median seconds per call."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(inner):
                fn()
            samples.append((time.perf_counter() - start) / inner)
    return statistics.median(samples)


def _slots(model: nn.Module) -> dict[str, nn.Module]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LinearSlot) or type(m).__name__ == "LinearSlot"}


def benchmark_ops(variants: dict[str, nn.Module], block_size: int, repeats: int = 30) -> list[dict]:
    """Per Linear shape: microseconds per call for each variant, at decode and prefill sizes."""
    fp32_slots = _slots(variants["fp32"])
    seen: dict[tuple[int, int], str] = {}
    for name, slot in fp32_slots.items():
        shape = (slot.linear.in_features, slot.linear.out_features)
        seen.setdefault(shape, name)
    rows = []
    for (n_in, n_out), name in seen.items():
        count = sum((s.linear.in_features, s.linear.out_features) == (n_in, n_out) for s in fp32_slots.values())
        for tokens in (1, block_size):
            x = torch.randn(1, tokens, n_in)
            row = {"layer": name.split(".", 2)[-1] if name.startswith("blocks") else name, "in": n_in, "out": n_out,
                   "per_model": count, "tokens": tokens}
            for vname, model in variants.items():
                slot = _slots(model)[name]
                row[f"{vname}_us"] = _time(lambda: slot(x), repeats) * 1e6
            rows.append(row)
    return rows


def benchmark_generation(variants: dict[str, nn.Module], tok, prompts: list[tuple[str, str]],
                         new_tokens: int = 60) -> dict[str, dict]:
    """Greedy replies of a fixed length (no early stop) so every variant does the same work."""
    from app.model import generate
    from app import core

    out = {}
    for name, model in variants.items():
        ids = [torch.tensor([tok.encode(core.format_prompt(r, l))]) for r, l in prompts]
        generate(model, ids[0], 5, temperature=0, block_size=model.cfg.block_size)  # warm-up
        start = time.perf_counter()
        for x in ids:
            generate(model, x, new_tokens, temperature=0, block_size=model.cfg.block_size)
        seconds = time.perf_counter() - start
        total = new_tokens * len(ids)
        out[name] = {"tokens": total, "seconds": round(seconds, 3), "tokens_per_s": round(total / seconds, 1),
                     "ms_per_token": round(1000 * seconds / total, 3)}
    return out


@dataclass
class Device:
    name: str
    bandwidth_gbs: float      # sustained memory bandwidth, GB/s
    fp32_gflops: float        # sustained float32 throughput
    int8_gops: float          # sustained int8 multiply-accumulate throughput (x2 ops)
    budget_mb: float          # memory you can give the dialogue model
    note: str


DEVICES = [
    Device("Handheld console", 25.0, 40.0, 160.0, 16.0,
           "mid-range mobile SoC, a few cores spared from rendering"),
    Device("Budget phone (little core)", 4.0, 4.0, 16.0, 4.0,
           "one efficiency core, model kept small so the game has the RAM"),
    Device("Microcontroller (Cortex-M7 class)", 0.4, 0.2, 0.8, 1.0,
           "e.g. a smart toy or e-reader; ~1 MB usable for the model"),
]


def variant_profile(model: nn.Module, bits: int, block_size: int) -> dict:
    """Bytes read and operations per generated token for a weight bit width."""
    from app.quantize import linear_layers

    linear = list(linear_layers(model).values())
    n_linear = sum(m.weight.numel() for m in linear)
    n_rows = sum(m.weight.shape[0] for m in linear)
    n_other = sum(p.numel() for p in model.parameters()) - n_linear
    # weights at `bits`; per-row scale+zero (3 bytes) for sub-float formats; everything else float32
    weight_bytes = n_linear * bits / 8 + (n_rows * 3 if bits < 32 else 0)
    n_emb = model.pos_emb.weight.numel() + model.tok_emb.weight.numel()
    cfg = model.cfg
    # per token only one token row and one position row of the embeddings are read
    other_bytes = 4 * (n_other - n_emb) + 4 * 2 * cfg.n_embd
    ctx = block_size // 2  # average context length while replying
    attn_ops = 2 * 2 * cfg.n_layer * ctx * cfg.n_embd  # q.k and att.v
    return {"bytes_per_token": weight_bytes + other_bytes, "linear_ops": 2 * n_linear, "attn_ops": attn_ops,
            "model_bytes": weight_bytes + 4 * n_other}


def simulate_devices(fp32_model: nn.Module, variants: dict[str, int], reply_tokens: int = 60) -> list[dict]:
    """variants: {name: weight bits}. Returns one row per (device, variant)."""
    rows = []
    for dev in DEVICES:
        for name, bits in variants.items():
            p = variant_profile(fp32_model, bits, fp32_model.cfg.block_size)
            mem_s = p["bytes_per_token"] / (dev.bandwidth_gbs * 1e9)
            lin_tput = (dev.fp32_gflops if bits == 32 else dev.int8_gops) * 1e9
            comp_s = p["linear_ops"] / lin_tput + p["attn_ops"] / (dev.fp32_gflops * 1e9)
            t = max(mem_s, comp_s)
            rows.append({"device": dev.name, "variant": name, "bits": bits,
                         "model_mb": round(p["model_bytes"] / 1e6, 3), "fits": p["model_bytes"] / 1e6 <= dev.budget_mb,
                         "ms_per_token": round(t * 1e3, 4), "tokens_per_s": round(1 / t, 1),
                         "reply_ms": round(t * reply_tokens * 1e3, 1),
                         "bound": "memory" if mem_s >= comp_s else "compute"})
    return rows


def devices_table() -> list[dict]:
    return [asdict(d) for d in DEVICES]


def threads() -> int:
    return llm.set_threads()
