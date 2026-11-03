"""LoRA checkpoint key normalization and exponential moving average."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def _remap_lora_keys(values: dict[str, torch.Tensor], expected: set[str]) -> dict[str, torch.Tensor]:
    if set(values) == expected:
        return values
    remapped = {}
    for key, tensor in values.items():
        name = key
        for prefix in ("base_model.model.", "base_model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        remapped[name] = tensor
    if set(remapped) != expected:
        raise KeyError("LoRA key inventory differs from PEFT runtime")
    return remapped


class LoRAEMA:
    """Solar-style trainable-only EMA: θ ← d·θ + (1-d)·θ_live, d=0.999."""

    def __init__(self, lora, *, decay: float = 0.999) -> None:
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError("EMA decay must be in [0, 1]")
        self.decay = float(decay)
        self.shadow = {
            key: tensor.detach().float().cpu().clone()
            for key, tensor in lora.state_dict().items()
        }

    def load_shadow(self, path: Path, expected: set[str]) -> None:
        raw = load_file(str(path), device="cpu")
        remapped = _remap_lora_keys(raw, expected)
        for key, shadow in self.shadow.items():
            shadow.copy_(remapped[key].detach().float())

    @torch.no_grad()
    def update(self, lora) -> None:
        live = lora.state_dict()
        for key, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(live[key].detach().float().cpu(), alpha=1.0 - self.decay)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {key: value.to(dtype=torch.bfloat16) for key, value in self.shadow.items()}
