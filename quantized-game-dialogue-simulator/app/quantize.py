"""Post-training quantization of the dialogue GPT, and model size measurement.

Three families of quantized models are built from the trained float32 checkpoint:

* ``dynamic``  PyTorch dynamic PTQ: every ``nn.Linear`` weight is stored as int8
               (per-tensor affine); activations are quantized on the fly at run time
               from their observed min/max. No calibration data needed.
* ``static``   PyTorch static PTQ: int8 weights *and* int8 activations at the input
               of every Linear, with activation scales fixed in advance by running
               calibration text through observers (``LinearSlot`` stubs in model.py).
* ``int8`` ... ``int2``   from-scratch weight quantization at any bit width: each
               Linear weight row gets its own scale and zero point
               (asymmetric, per output channel), weights are rounded to ``bits``-bit
               integers and bit-packed. Used for the bit-width study (stretch goal);
               the model runs on the dequantized weights ("fake quantization"), so
               quality is exact while speed is simulated (see ``simulate.py``).

Embeddings and LayerNorms stay float32 in every variant; they are reported separately.
"""
from __future__ import annotations

import copy
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao")
from torch.ao.quantization import convert, get_default_qconfig, prepare, quantize_dynamic  # noqa: E402

from app import core, storage  # noqa: E402
from app.model import GPT, LinearSlot  # noqa: E402

ENGINE = "x86" if "x86" in torch.backends.quantized.supported_engines else "qnnpack"
BIT_WIDTHS = (8, 6, 4, 3, 2)


def _engine() -> None:
    torch.backends.quantized.engine = ENGINE


# --- PyTorch dynamic and static PTQ -------------------------------------------------------

def dynamic_quantize(model: GPT) -> nn.Module:
    _engine()
    return quantize_dynamic(copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)


def calibration_batches(ids: torch.Tensor, block_size: int, n_windows: int = 64, batch_size: int = 16):
    """Evenly spaced windows of training text for the static-quantization observers."""
    starts = torch.linspace(0, len(ids) - block_size - 1, n_windows).long()
    windows = torch.stack([ids[s: s + block_size] for s in starts])
    return list(windows.split(batch_size))


def static_quantize(model: GPT, calib: list[torch.Tensor]) -> nn.Module:
    """Observers on each LinearSlot's input/output, calibrate, then convert to int8 kernels."""
    _engine()
    q = copy.deepcopy(model).eval()
    q.qconfig = None
    for module in q.modules():
        if isinstance(module, LinearSlot):
            module.qconfig = get_default_qconfig(ENGINE)
    prepare(q, inplace=True)
    with torch.no_grad():
        for batch in calib:
            q(batch)
    convert(q, inplace=True)
    return q


# --- from-scratch k-bit weight quantization ---------------------------------------------------

@dataclass
class QTensor:
    """Integer codes in [0, 2^bits - 1] with one scale / zero point per output row."""

    codes: torch.Tensor     # uint8 holding the unpacked codes, shape like the weight
    scale: torch.Tensor     # (rows, 1) float32
    zero_point: torch.Tensor  # (rows, 1) float32
    bits: int

    def dequantize(self) -> torch.Tensor:
        return (self.codes.float() - self.zero_point) * self.scale


def quantize_tensor(w: torch.Tensor, bits: int) -> QTensor:
    """Asymmetric per-row affine quantization: w ~= (q - z) * s, q in [0, 2^bits - 1]."""
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")
    qmax = 2 ** bits - 1
    w = w.detach().float()
    lo = w.min(dim=1, keepdim=True).values.clamp(max=0.0)  # include 0 so zero stays exact
    hi = w.max(dim=1, keepdim=True).values.clamp(min=0.0)
    # scales are stored as float16 on disk; round them now so the evaluated model == the saved model
    scale = ((hi - lo) / qmax).clamp(min=1e-4).half().float()
    zero = torch.round(-lo / scale).clamp(0, qmax)
    codes = torch.clamp(torch.round(w / scale) + zero, 0, qmax).to(torch.uint8)
    return QTensor(codes, scale, zero, bits)


def pack_bits(codes: torch.Tensor, bits: int) -> bytes:
    """Store each code in exactly ``bits`` bits (e.g. two 4-bit codes per byte)."""
    flat = codes.flatten().numpy().astype(np.uint8)
    bit_matrix = np.unpackbits(flat[:, None], axis=1)[:, 8 - bits:]
    return np.packbits(bit_matrix.flatten()).tobytes()


def unpack_bits(data: bytes, bits: int, shape: tuple[int, ...]) -> torch.Tensor:
    n = int(np.prod(shape))
    bit_matrix = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: n * bits].reshape(n, bits)
    padded = np.concatenate([np.zeros((n, 8 - bits), dtype=np.uint8), bit_matrix], axis=1)
    return torch.from_numpy(np.packbits(padded, axis=1).reshape(shape).copy())


def linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def kbit_quantize(model: GPT, bits: int) -> tuple[GPT, dict[str, QTensor]]:
    """Copy of the model whose Linear weights are replaced by their k-bit reconstruction."""
    q = copy.deepcopy(model).eval()
    qtensors = {}
    with torch.no_grad():
        for name, layer in linear_layers(q).items():
            qt = quantize_tensor(layer.weight, bits)
            layer.weight.copy_(qt.dequantize())
            qtensors[name] = qt
    return q, qtensors


def save_kbit(model: GPT, qtensors: dict[str, QTensor], tok: core.CharTokenizer, path: str | Path) -> int:
    """Deployable k-bit file: packed codes + per-row scales (fp16) + every other tensor in fp32."""
    linear_weights = {f"{n}.weight" for n in qtensors}
    payload = {
        "bits": next(iter(qtensors.values())).bits,
        "model_config": model.cfg.to_dict(),
        "chars": tok.chars,
        # packed codes go in a uint8 tensor: torch.save pickles raw `bytes` as latin-1 text, which inflates them ~1.5x
        "packed": {n: (torch.frombuffer(bytearray(pack_bits(qt.codes, qt.bits)), dtype=torch.uint8), tuple(qt.codes.shape),
                       qt.scale.half(), qt.zero_point.to(torch.uint8))
                   for n, qt in qtensors.items()},
        "float": {k: v for k, v in model.state_dict().items() if k not in linear_weights},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path.stat().st_size


def load_kbit(path: str | Path) -> tuple[GPT, core.CharTokenizer]:
    from app.model import GPTConfig

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**payload["model_config"]))
    state = dict(payload["float"])
    for name, (data, shape, scale, zero) in payload["packed"].items():
        codes = unpack_bits(data.numpy().tobytes(), payload["bits"], shape)
        state[f"{name}.weight"] = (codes.float() - zero.float()) * scale.float()
    model.load_state_dict(state)
    return model.eval(), core.CharTokenizer(payload["chars"])


# --- size measurement ---------------------------------------------------------------------

def serialized_size(obj) -> int:
    """Bytes ``torch.save`` writes for a model's state_dict (or any object)."""
    buf = io.BytesIO()
    torch.save(obj.state_dict() if isinstance(obj, nn.Module) else obj, buf)
    return buf.getbuffer().nbytes


def size_breakdown(model: GPT) -> dict[str, int]:
    """Float32 bytes by part of the model: what quantization can and cannot shrink."""
    linear = sum(m.weight.numel() * 4 for m in linear_layers(model).values())
    biases = sum(m.bias.numel() * 4 for m in linear_layers(model).values() if m.bias is not None)
    emb = (model.tok_emb.weight.numel() + model.pos_emb.weight.numel()) * 4
    total = sum(p.numel() * 4 for p in model.parameters())
    return {"linear_weights": linear, "linear_biases": biases, "embeddings": emb,
            "layernorms": total - linear - biases - emb, "total": total}


# --- building and loading all variants ---------------------------------------------------------

def variant_names() -> list[str]:
    return ["fp32", "dynamic", "static"] + [f"int{b}" for b in BIT_WIDTHS]


def build_all(model: GPT, tok: core.CharTokenizer, calib_ids: torch.Tensor, out_dir: str | Path | None = None,
              log=print) -> dict[str, dict]:
    """Every variant, its file on disk and its measured size. Returns {name: {"model", "bytes", "kind"}}."""
    out_dir = Path(out_dir or storage.model_dir() / "variants")  # never the training checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict] = {}

    def save_torch(name: str, m: nn.Module, kind: str):
        path = out_dir / f"{name}.pt"
        torch.save(m.state_dict(), path)
        variants[name] = {"model": m, "bytes": path.stat().st_size, "kind": kind, "file": path.name}
        log(f"{name:>8}: {path.stat().st_size / 1e6:.3f} MB  ({kind})")

    save_torch("fp32", model, "float32 baseline")
    save_torch("dynamic", dynamic_quantize(model), "PyTorch dynamic PTQ, int8 weights")
    calib = calibration_batches(calib_ids, model.cfg.block_size)
    save_torch("static", static_quantize(model, calib), "PyTorch static PTQ, int8 weights + activations")
    for bits in BIT_WIDTHS:
        qmodel, qt = kbit_quantize(model, bits)
        path = out_dir / f"int{bits}.pt"
        size = save_kbit(qmodel, qt, tok, path)
        variants[f"int{bits}"] = {"model": qmodel, "bytes": size, "kind": f"from-scratch {bits}-bit weights, per-row scale",
                                  "file": path.name}
        log(f"{'int' + str(bits):>8}: {size / 1e6:.3f} MB  (from-scratch {bits}-bit packed weights)")
    return variants


def load_variant(name: str):
    """(model, tokenizer) for any variant name, rebuilt from the float checkpoint."""
    from app import llm, train

    model, tok, _ = llm.load_checkpoint()
    if name == "fp32":
        return model, tok
    if name == "dynamic":
        return dynamic_quantize(model), tok
    if name == "static":
        _, train_ids, _ = train.load_data()
        return static_quantize(model, calibration_batches(train_ids, model.cfg.block_size)), tok
    if name.startswith("int") and name[3:].isdigit():
        return kbit_quantize(model, int(name[3:]))[0], tok
    raise SystemExit(f"unknown model {name!r}; choose from {', '.join(variant_names())}")
