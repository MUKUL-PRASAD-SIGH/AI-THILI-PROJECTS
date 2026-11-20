"""LoRA (Low-Rank Adaptation) written from scratch in plain PyTorch.

A frozen weight matrix W (out x in) gets a trainable low-rank update:

    h = W x + (alpha / r) * B A x          A: r x in,  B: out x r

A starts random (Kaiming uniform) and B starts at zero, so at step 0 the adapted
model is exactly the pre-trained model. Only A and B are trained: for GPT-2 with
r=16 on every attention and MLP projection that is about 2.4M parameters, ~2% of
the 124M in the base model.

After training the update can be folded into the base weight,

    W' = W + (alpha / r) * B A

which removes the extra matmuls at inference: the merged model has the original
architecture, size and speed (see ``merge_lora``).

GPT-2 stores its projections as ``transformers`` ``Conv1D`` layers whose weight is
(in x out), the transpose of ``nn.Linear``. ``LoRALinear`` handles both layouts.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_CONFIG = "adapter_config.json"


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    # Module names are matched on their dotted suffix, e.g. "attn.c_attn" matches
    # "transformer.h.0.attn.c_attn". These four are every projection in a GPT-2 block.
    target_modules: list[str] = field(
        default_factory=lambda: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    )

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LoRAConfig":
        known = {k: data[k] for k in ("r", "alpha", "dropout", "target_modules") if k in data}
        return cls(**known)


def _is_conv1d(layer: nn.Module) -> bool:
    # transformers.pytorch_utils.Conv1D: weight (in, out), forward is x @ W + b
    return type(layer).__name__ == "Conv1D" and hasattr(layer, "nf")


def _in_out(layer: nn.Module) -> tuple[int, int]:
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    if _is_conv1d(layer):
        n_in, n_out = layer.weight.shape
        return n_in, n_out
    raise TypeError(f"LoRA supports nn.Linear and GPT-2 Conv1D, not {type(layer).__name__}")


class LoRALinear(nn.Module):
    """Wraps a frozen linear layer and adds a trainable low-rank update."""

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        self.in_features, self.out_features = _in_out(base)
        self.fan_in_fan_out = _is_conv1d(base)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        weight = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=weight.dtype, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=weight.dtype, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        """The update (alpha/r) * B A, in the same layout as the base layer's weight."""
        delta = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
        return delta.T if self.fan_in_fan_out else delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        # x @ A^T @ B^T: two thin matmuls instead of materialising the full delta
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + update * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


def _matches(name: str, targets: list[str]) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Replace every target layer with a LoRALinear wrapper and freeze everything else.

    Returns the names of the wrapped layers. Raises if nothing matched, which almost
    always means a typo in ``target_modules``.
    """
    names = [
        n for n, m in model.named_modules()
        if _matches(n, config.target_modules) and not isinstance(m, LoRALinear)
        and (isinstance(m, nn.Linear) or _is_conv1d(m))
    ]
    if not names:
        raise ValueError(f"no layers matched target_modules={config.target_modules}")
    for name in names:
        _set_submodule(model, name, LoRALinear(model.get_submodule(name), config.r, config.alpha, config.dropout))
    mark_only_lora_trainable(model)
    return names


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts. Tied weights are counted once."""
    seen: set[int] = set()
    trainable = total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
            if "lora_A" in n or "lora_B" in n}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    expected = {n for n in params if "lora_A" in n or "lora_B" in n}
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(f"adapter does not fit this model (missing {len(missing)}, unexpected {len(unexpected)})")
    with torch.no_grad():
        for name, tensor in state.items():
            if params[name].shape != tensor.shape:
                raise ValueError(f"{name}: adapter shape {tuple(tensor.shape)} != model {tuple(params[name].shape)}")
            params[name].copy_(tensor.to(params[name].device, params[name].dtype))


def save_adapter(model: nn.Module, config: LoRAConfig, out_dir: str | Path, extra: dict | None = None) -> Path:
    """Write only the LoRA weights (a few MB) plus the config needed to re-inject them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), str(out / ADAPTER_WEIGHTS))
    meta = {"lora": config.to_dict(), **(extra or {})}
    (out / ADAPTER_CONFIG).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def read_adapter_config(adapter_dir: str | Path) -> dict:
    return json.loads((Path(adapter_dir) / ADAPTER_CONFIG).read_text(encoding="utf-8"))


def load_adapter(model: nn.Module, adapter_dir: str | Path) -> LoRAConfig:
    """Inject LoRA layers into a fresh base model and load trained adapter weights into them."""
    config = LoRAConfig.from_dict(read_adapter_config(adapter_dir)["lora"])
    inject_lora(model, config)
    load_lora_state_dict(model, load_file(str(Path(adapter_dir) / ADAPTER_WEIGHTS)))
    return config


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight and put the plain layer back.

    The result has no LoRA modules left: same architecture and parameter count as
    the original pre-trained model, so it can be saved with ``save_pretrained`` and
    loaded anywhere without this code.
    """
    for name, layer in lora_layers(model).items():
        layer.merge()
        _set_submodule(model, name, layer.base)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
