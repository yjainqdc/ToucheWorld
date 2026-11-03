"""Subsample the official 30-step LTX sigma grid for block-AR few-step denoise.

Training matches each student state to GT x0 on the current chunk, then Euler
to the next sigma. Inference uses the same grid without touching SolarWM's
frozen 30-step InferencePlan.
"""

from __future__ import annotations

import numpy as np
import torch

from solarwm.backends.ltx25.inference import native_sigma_schedule


def fewstep_sigma_schedule(steps: int) -> np.ndarray:
    """Return ``steps + 1`` descending sigmas ending at 0, taken from the 30-step grid."""

    if isinstance(steps, bool) or int(steps) < 1:
        raise ValueError("few-step schedule needs at least 1 denoise step")
    full = native_sigma_schedule(30)
    count = int(steps) + 1
    index = np.linspace(0, full.shape[0] - 1, count)
    index = np.round(index).astype(np.int64)
    index[0] = 0
    index[-1] = full.shape[0] - 1
    chosen = full[index]
    if chosen.shape[0] != count:
        raise ValueError("few-step subsample collapsed")
    if np.any(chosen[:-1] <= chosen[1:]):
        raise ValueError("few-step sigmas must be strictly descending")
    if float(chosen[-1]) != 0.0:
        raise ValueError("few-step schedule must finish at 0")
    return chosen.astype(np.float32, copy=True)


def fewstep_sigma_tensor(steps: int, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(fewstep_sigma_schedule(steps), device=device, dtype=torch.float32)


def blend_block_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    first: torch.Tensor,
    sigma: torch.Tensor,
    *,
    start: int,
    end: int,
) -> torch.Tensor:
    """History stays clean; current block is the RF interpolate at ``sigma``."""

    sigma_b = sigma.view(-1, 1, 1, 1, 1).to(dtype=torch.float32)
    mixed = clean.float().clone()
    mixed[:, :, start:end] = (1.0 - sigma_b) * clean[:, :, start:end].float() + sigma_b * noise[
        :, :, start:end
    ].float()
    mixed[:, :, :1] = first.float()
    return mixed.to(dtype=clean.dtype)


def shortcut_velocity(latent: torch.Tensor, clean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Exact v that jumps from ``latent`` to ``clean`` in one Euler step: (x - x0) / σ."""

    scale = sigma.view(-1, 1, 1, 1, 1).to(dtype=torch.float32).clamp_min(1.0e-6)
    return ((latent.float() - clean.float()) / scale).to(dtype=latent.dtype)


def euler_commit_block(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    start: int,
    end: int,
    first: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    updated = latent.float() + velocity.float() * (sigma_next.float() - sigma.float())
    out = latent.clone()
    out[:, :, start:end] = updated[:, :, start:end].to(dtype=latent.dtype)
    out[:, :, :start] = history[:, :, :start]
    out[:, :, :1] = first
    return out
