"""Portable path defaults shared by command-line entry points.

Explicit command-line arguments take precedence over these environment defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


def path_default(name: str) -> str:
    models = Path(os.environ.get("MODEL_DIR", "models")).expanduser()
    data = Path(os.environ.get("DATA_ROOT", "data")).expanduser()
    weights = models / "ToucheWorld"
    defaults = {
        "TRANSFORMER": models / "ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "VAE": models / "ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors",
        "GEMMA": models / "ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "CAMERA": models / "solarwm/camera_adapter.safetensors",
        "JSONL": data / "train/train.jsonl",
        "CACHE_DIR": Path("cache"),
        "BENCH_ROOT": data / "bench",
        "RUN_DIR": weights / "teacher27",
        "HAND_RUN": weights / "causal27",
        "DISTILL_RUN": weights / "causal27",
    }
    if name not in defaults:
        raise KeyError(f"Unknown path setting: {name}")
    return str(Path(os.environ.get(name) or defaults[name]).expanduser())
