"""Block-level teacher forcing for causal LTX-2.5 (history clean, current block noisy)."""

from __future__ import annotations

import torch

from pipelines.ltx25_solar.causal_attn import denoise_block_ranges
from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY
from solarwm.backends.ltx25.torch_flow import Objective


def sample_denoise_span(
    *,
    block_frames: int,
    latent_frames: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    frames = int(STABLE_GEOMETRY.latent_frames if latent_frames is None else latent_frames)
    ranges = denoise_block_ranges(frames, int(block_frames))
    pick = int(torch.randint(len(ranges), (1,), generator=generator).item())
    return ranges[pick]


def prepare_block_objective(
    video_latent: torch.Tensor,
    first_frame_latent: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    *,
    start: int,
    end: int,
) -> tuple[Objective, torch.Tensor]:
    """Noise only ``[start, end)``. Return objective plus ``[B, T]`` t=0 context mask."""

    frames = STABLE_GEOMETRY.latent_frames
    if not 0 < start < end <= frames:
        raise ValueError(f"invalid denoise span [{start}, {end}) for {frames} frames")
    batch = video_latent.shape[0]
    clean = video_latent.float().clone()
    clean[:, :, :1] = first_frame_latent.float()
    noise_fp32 = noise.float()
    target = noise_fp32 - clean
    expanded = sigma.view(batch, 1, 1, 1, 1).to(dtype=torch.float32)
    noisy = clean.clone()
    span = slice(start, end)
    noisy[:, :, span] = (1.0 - expanded) * clean[:, :, span] + expanded * noise_fp32[:, :, span]
    noisy[:, :, :1] = first_frame_latent.float()
    context = torch.ones((batch, frames), device=video_latent.device, dtype=torch.bool)
    context[:, start:end] = False
    return Objective(noisy.to(dtype=video_latent.dtype), target, sigma), context


def zero_future_hand(
    control: torch.Tensor,
    motion: torch.Tensor,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop hand signals after the current block so conv/injection cannot see the future."""

    control = control.clone()
    motion = motion.clone()
    control[:, :, end:] = 0
    motion[:, end:] = 0
    return control, motion


def block_loss_mask(start: int, end: int, like: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(like, dtype=torch.float32)
    mask[:, :, start:end] = 1.0
    mask[:, :, 0] = 0.0
    return mask
