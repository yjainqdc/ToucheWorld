"""Hand-conditioning sidecar for LTX-2.5, Generated Reality-style injection.

The conditioning enters the DiT as

    x = patchify_proj(patchify(z_rgb)) + control_proj(patchify(VAE(skeleton)))
      + E_motion(H)

``patchify_proj`` stays the base Linear.  ``control_proj`` and the motion
encoder's last layer are bias-free and zero-initialised, so all-zero
keypoints are an exact no-op through untrained weights.
"""

from __future__ import annotations

import torch
from torch import nn

HAND_STATE_PREFIX = "hand_conditioning."


def is_hand_state_key(name: str) -> bool:
    return name.startswith(HAND_STATE_PREFIX) or f".{HAND_STATE_PREFIX}" in name


class HandMotionEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, inner_dim: int):
        super().__init__()
        if min(feature_dim, hidden_dim, inner_dim) <= 0:
            raise ValueError("hand motion encoder dimensions must be positive")
        self.in_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False
        )
        self.act = nn.GELU()
        self.out_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.causal: bool = False
        self.reset_output_projections()

    def reset_output_projections(self) -> None:
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.in_proj(features))
        if self.causal:
            padded = torch.nn.functional.pad(hidden.transpose(1, 2), (2, 0))
            conv = torch.nn.functional.conv1d(padded, self.temporal_conv.weight)
        else:
            conv = self.temporal_conv(hidden.transpose(1, 2))
        hidden = hidden + self.act(conv.transpose(1, 2))
        return self.out_proj(hidden)


class HandConditioning(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        inner_dim: int,
        motion_feature_dim: int,
        motion_hidden_dim: int,
    ):
        super().__init__()
        if min(latent_channels, inner_dim) <= 0:
            raise ValueError("hand conditioning dimensions must be positive")
        self.control_proj = nn.Linear(latent_channels, inner_dim, bias=False)
        self.motion_encoder = HandMotionEncoder(
            motion_feature_dim, motion_hidden_dim, inner_dim
        )
        self.reset_output_projections()

    def reset_output_projections(self) -> None:
        nn.init.zeros_(self.control_proj.weight)
        self.motion_encoder.reset_output_projections()

    def forward(
        self,
        control_tokens: torch.Tensor | None,
        motion_features: torch.Tensor | None,
        tokens_per_frame: int,
        motion_history: int = 0,
    ) -> torch.Tensor:
        if control_tokens is None and motion_features is None:
            raise ValueError(
                "hand conditioning called with neither control nor motion input"
            )
        residual: torch.Tensor | None = None
        if control_tokens is not None:
            residual = self.control_proj(control_tokens)
        if motion_features is not None:
            motion = self.motion_encoder(motion_features)
            if motion_history:
                motion = motion[:, motion_history:]
            motion = motion.repeat_interleave(tokens_per_frame, dim=1)
            residual = motion if residual is None else residual + motion
        return residual
