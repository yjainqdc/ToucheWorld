"""Block-autoregressive Euler rollout for causal LTX-2.5.

History frames stay at the previously committed x0 (timestep 0). Only the
current latent chunk is denoised. Future chunks keep their reserved noise and
are invisible to attention via the block-causal mask.

Optional sliding-window mask + KV cache: current queries attend frame 0 and
the last ``window_frames`` latent frames; after the first Euler step of a
chunk, history K/V are reused.
"""

from __future__ import annotations

import itertools
from contextlib import nullcontext

import torch

from pipelines.ltx25_solar.block_tf import zero_future_hand
from pipelines.ltx25_solar.causal_attn import denoise_block_ranges, set_causal_window_frames
from pipelines.ltx25_solar.fewstep import fewstep_sigma_schedule
from pipelines.ltx25_solar.kv_cache import AttnKVCache, install_attn_kv_cache
from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY
from solarwm.backends.ltx25.torch_inference import (
    LTX25Sampler,
    _guided_x0,
    _patchified_noise,
    _perturbation,
    _x0,
)
from solarwm.errors import BackendContractError
from solarwm.inference.engine import InferenceCase


class BlockARSampler(LTX25Sampler):
    def __init__(
        self,
        *args,
        block_frames: int = 4,
        window_frames: int = 8,
        kv_cache: bool = True,
        denoise_steps: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_frames = int(block_frames)
        self.window_frames = int(window_frames)
        self.denoise_steps = int(denoise_steps)
        self.kv: AttnKVCache | None = None
        set_causal_window_frames(self.window_frames)
        if kv_cache:
            self.kv = AttnKVCache()
            wrapped = install_attn_kv_cache(self.model, self.kv)
            print(
                f"[block-ar] window={self.window_frames} kv-cache attn1={wrapped} "
                f"denoise_steps={self.denoise_steps}",
                flush=True,
            )

    def _velocity_block(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor,
        viewmats: torch.Tensor,
        camera_k: torch.Tensor,
        *,
        caption: torch.Tensor,
        mask: torch.Tensor,
        context_mask: torch.Tensor,
        perturbed: bool,
        branch: str,
    ) -> torch.Tensor:
        if self.kv is not None:
            self.kv.branch = branch
        result = self.model(
            video_latent=latent,
            sigma=sigma.reshape(1),
            caption_embedding=caption,
            first_frame_mask=context_mask,
            camera={"viewmats": viewmats, "K": camera_k},
            caption_mask=mask,
            perturbations=_perturbation(
                perturbed=perturbed,
                blocks=self.plan.guidance.stg_blocks,
                device=self.device,
            ),
        )
        if tuple(result.shape) != (1, *STABLE_GEOMETRY.latent_shape):
            raise BackendContractError("LTX inference velocity geometry differs")
        if not bool(torch.isfinite(result).all()):
            raise BackendContractError("LTX inference velocity contains NaN or Inf")
        return result

    def sample(self, case: InferenceCase) -> torch.Tensor:
        batch = self.source.get(case.sample_id)
        first = batch.first_frame_latent.to(self.device, dtype=torch.bfloat16)
        caption = batch.video_prompt_embeds.to(self.device, dtype=torch.bfloat16)
        mask = batch.prompt_attention_mask.to(self.device, dtype=torch.int64)
        viewmats = batch.relative_w2c.to(self.device, dtype=torch.float32)
        camera_k = batch.camera_k.to(self.device, dtype=torch.float32)
        generator = torch.Generator(device=self.device).manual_seed(case.noise_seed)
        latent = _patchified_noise(generator=generator, device=self.device)
        latent[:, :, :1].copy_(first)
        prefix = int((case.metadata or {}).get("prefix_latents") or 1)
        prefix = max(1, min(prefix, STABLE_GEOMETRY.latent_frames - 1))
        history = batch.video_latent.to(self.device, dtype=torch.bfloat16)
        if prefix > 1:
            latent[:, :, :prefix].copy_(history[:, :, :prefix])
            print(
                f"[block-ar] {case.sample_id} pin GT prefix_latents={prefix}",
                flush=True,
            )
        if self.denoise_steps == 30:
            sigma_values = self.plan.sigmas.copy()
        else:
            sigma_values = fewstep_sigma_schedule(self.denoise_steps)
        sigmas = torch.tensor(sigma_values, device=self.device, dtype=torch.float32)
        ranges = denoise_block_ranges(STABLE_GEOMETRY.latent_frames, self.block_frames)
        base_control = getattr(self.model, "control", None)
        base_motion = getattr(self.model, "motion", None)
        set_causal_window_frames(self.window_frames)

        was_training = self.model.training
        self.model.eval()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        try:
            with torch.inference_mode(), autocast:
                ranges = tuple(
                    (max(start, prefix), end)
                    for start, end in ranges
                    if end > prefix
                )
                for start, end in ranges:
                    context = torch.ones(
                        (1, STABLE_GEOMETRY.latent_frames),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    context[:, start:end] = False
                    if base_control is not None and base_motion is not None:
                        control, motion = zero_future_hand(base_control, base_motion, end)
                        self.model.control = control
                        self.model.motion = motion
                    if self.kv is not None:
                        self.kv.begin_block(start, end, self.window_frames)
                    print(
                        f"[block-ar] {case.sample_id} denoise [{start},{end}) "
                        f"window={self.window_frames} kv={self.kv is not None}",
                        flush=True,
                    )
                    for step, (sigma, sigma_next) in enumerate(itertools.pairwise(sigmas)):
                        if self.kv is not None:
                            self.kv.mode = "fill" if step == 0 else "hit"
                        kwargs = dict(
                            viewmats=viewmats,
                            camera_k=camera_k,
                            context_mask=context,
                        )
                        conditioned = self._velocity_block(
                            latent,
                            sigma,
                            caption=caption,
                            mask=mask,
                            perturbed=False,
                            branch="cond",
                            **kwargs,
                        )
                        unconditioned = self._velocity_block(
                            latent,
                            sigma,
                            caption=self.negative_caption,
                            mask=self.negative_mask,
                            perturbed=False,
                            branch="uncond",
                            **kwargs,
                        )
                        perturbed = self._velocity_block(
                            latent,
                            sigma,
                            caption=caption,
                            mask=mask,
                            perturbed=True,
                            branch="stg",
                            **kwargs,
                        )
                        timestep = sigma.expand(
                            1, 1, STABLE_GEOMETRY.latent_frames, 1, 1
                        ).clone()
                        timestep[:, :, context[0]] = 0.0
                        denoised = _guided_x0(
                            _x0(latent, conditioned, timestep),
                            _x0(latent, unconditioned, timestep),
                            _x0(latent, perturbed, timestep),
                            self.plan,
                        )
                        denoised[:, :, :prefix].copy_(history[:, :, :prefix])
                        denoised[:, :, :start] = latent[:, :, :start]
                        velocity = ((latent.float() - denoised.float()) / sigma.float()).to(
                            latent.dtype
                        )
                        updated = (
                            latent.float()
                            + velocity.float() * (sigma_next.float() - sigma.float())
                        ).to(latent.dtype)
                        latent[:, :, start:end] = updated[:, :, start:end]
                        latent[:, :, :prefix].copy_(history[:, :, :prefix])
                        if not bool(torch.isfinite(latent).all()):
                            raise BackendContractError("block-AR Euler became non-finite")
                    if self.kv is not None:
                        self.kv.mode = "off"
                    if base_control is not None:
                        self.model.control = base_control
                        self.model.motion = base_motion
        finally:
            set_causal_window_frames(0)
            if self.kv is not None:
                self.kv.mode = "off"
            self.model.train(was_training)
        if not torch.equal(latent[:, :, :prefix], history[:, :, :prefix]):
            raise BackendContractError("block-AR changed the clean prefix latents")
        return latent.contiguous()
