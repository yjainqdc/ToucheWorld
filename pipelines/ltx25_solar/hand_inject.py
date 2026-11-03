"""Attach the hand sidecar after Solar ``patchify_proj`` and inject a frozen-camera + trainable-hand LoRA pair."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from pipelines.ltx25_solar.hand_conditioning import HandConditioning
from solarwm.backends.ltx25.adapter import lora_target_modules
from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY
from solarwm.backends.ltx25.torch_distributed import token_bounds
from solarwm.backends.ltx25.torch_model import LoRARuntime
from solarwm.errors import BackendContractError


class HandPatchifyProj(nn.Module):
    """``patchify_proj`` plus an optional zero-init sidecar residual."""

    def __init__(self, linear: nn.Module, hand: HandConditioning) -> None:
        super().__init__()
        self.linear = linear
        self.hand = hand
        self.control_tokens: torch.Tensor | None = None
        self.motion_features: torch.Tensor | None = None

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(latent)
        if self.control_tokens is None and self.motion_features is None:
            return hidden
        residual = self.hand(
            self.control_tokens,
            self.motion_features,
            STABLE_GEOMETRY.tokens_per_latent,
        )
        return hidden + residual.to(dtype=hidden.dtype)


def peft_core(module: nn.Module) -> nn.Module:
    if hasattr(module, "get_base_model"):
        return module.get_base_model()
    return module


def install_hand_patchify(core: nn.Module, hook: HandPatchifyProj) -> None:
    core.patchify_proj = hook
    preprocessor = getattr(core, "video_args_preprocessor", None)
    if preprocessor is not None and hasattr(preprocessor, "patchify_proj"):
        preprocessor.patchify_proj = hook
    nested = getattr(preprocessor, "simple_preprocessor", None)
    if nested is not None and hasattr(nested, "patchify_proj"):
        nested.patchify_proj = hook


class HandAwareBackbone(nn.Module):
    """Slice control tokens the same way SP slices RGB tokens, then forward."""

    def __init__(self, backbone: nn.Module, hook: HandPatchifyProj) -> None:
        super().__init__()
        self.backbone = backbone
        # Same module is already registered as core.patchify_proj; do not
        # register it again or AdamW would see duplicate Parameter objects.
        object.__setattr__(self, "_hook", hook)

    def forward(
        self,
        video_latent: torch.Tensor,
        sigma: torch.Tensor,
        caption_embedding: torch.Tensor,
        first_frame_mask: torch.Tensor,
        camera: dict[str, torch.Tensor],
        *,
        caption_mask: torch.Tensor,
        hand_control_latent: torch.Tensor | None = None,
        hand_motion_features: torch.Tensor | None = None,
        perturbations: Any = None,
    ) -> torch.Tensor:
        from ltx_core.components.patchifiers import VideoLatentPatchifier

        control_tokens = None
        motion = None
        if hand_control_latent is not None:
            if tuple(hand_control_latent.shape[1:]) != tuple(video_latent.shape[1:]):
                raise BackendContractError("hand control latent must match RGB latent shape")
            control_tokens = VideoLatentPatchifier(patch_size=1).patchify(hand_control_latent)
            start, stop = token_bounds(int(control_tokens.shape[1]))
            control_tokens = control_tokens[:, start:stop].contiguous()
        if hand_motion_features is not None:
            expected = (
                video_latent.shape[0],
                STABLE_GEOMETRY.latent_frames,
                182,
            )
            if tuple(hand_motion_features.shape) != expected:
                raise BackendContractError(
                    f"hand motion must be {expected}, got {tuple(hand_motion_features.shape)}"
                )
            start, stop = token_bounds(STABLE_GEOMETRY.video_tokens)
            tpf = STABLE_GEOMETRY.tokens_per_latent
            motion = hand_motion_features[:, start // tpf : stop // tpf].contiguous()
        hook = self._hook
        hook.control_tokens = control_tokens
        hook.motion_features = motion
        try:
            return self.backbone(
                video_latent=video_latent,
                sigma=sigma,
                caption_embedding=caption_embedding,
                first_frame_mask=first_frame_mask,
                camera=camera,
                caption_mask=caption_mask,
                perturbations=perturbations,
            )
        finally:
            hook.control_tokens = None
            hook.motion_features = None


def _adapter_runtime(core: nn.Module, adapter_name: str, *, zero_output: bool) -> LoRARuntime:
    import peft

    if zero_output:
        for name, parameter in core.named_parameters():
            if f"lora_B.{adapter_name}" in name:
                parameter.data.zero_()
    names = list(core.peft_config.keys())
    # PeftModel.set_adapter() in this PEFT only takes one name; the tuner
    # can keep several adapters on at once (camera + hand + distill).
    core.base_model.set_adapter(names)
    for parameter in core.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    state = peft.get_peft_model_state_dict(
        core,
        state_dict=core.state_dict(keep_vars=True),
        adapter_name=adapter_name,
        save_embedding_layers=False,
    )
    parameter_by_key: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.nn.Parameter):
            raise BackendContractError(f"{adapter_name} LoRA state did not retain Parameters")
        parameter_by_key[key] = value
    return LoRARuntime(core, peft, parameter_by_key)


def inject_named_adapter(
    core: nn.Module,
    name: str,
    *,
    rank: int,
    zero_output: bool = True,
) -> LoRARuntime:
    """Add a PEFT adapter on the same 480 LTX targets as camera/hand."""

    import peft

    if name in core.peft_config:
        raise BackendContractError(f"adapter {name!r} already exists")
    configuration = peft.LoraConfig(
        r=int(rank),
        lora_alpha=int(rank),
        lora_dropout=0.0,
        target_modules=list(lora_target_modules()),
        bias="none",
        init_lora_weights=True,
    )
    core.add_adapter(name, configuration)
    return _adapter_runtime(core, name, zero_output=zero_output)


def inject_hand_adapter(core: nn.Module, *, rank: int = 32) -> LoRARuntime:
    """Add a second PEFT adapter named ``hand``; leave ``default`` (camera) in place."""

    return inject_named_adapter(core, "hand", rank=rank, zero_output=False)


def inject_distill_adapter(core: nn.Module, *, rank: int = 64) -> LoRARuntime:
    """Add a zero-init ``distill`` adapter; keep camera + hand adapters enabled."""

    return inject_named_adapter(core, "distill", rank=rank, zero_output=True)


def freeze_camera_train_hand(model: nn.Module) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_sidecar = ".patchify_proj.hand." in name
        is_hand_lora = ("lora_A.hand" in name) or ("lora_B.hand" in name)
        parameter.requires_grad_(is_sidecar or is_hand_lora)
        if parameter.requires_grad and id(parameter) not in {id(item) for item in trainable}:
            trainable.append(parameter)
    if not trainable:
        raise BackendContractError("no trainable hand LoRA / sidecar parameters")
    return trainable


def freeze_all_train_adapter(model: nn.Module, adapter_name: str = "distill") -> list[torch.nn.Parameter]:
    """Freeze camera LoRA, hand LoRA, and sidecar; train only ``adapter_name``."""

    token_a = f"lora_A.{adapter_name}"
    token_b = f"lora_B.{adapter_name}"
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        keep = (token_a in name) or (token_b in name)
        parameter.requires_grad_(keep)
        if parameter.requires_grad and id(parameter) not in {id(item) for item in trainable}:
            trainable.append(parameter)
    if not trainable:
        raise BackendContractError(f"no trainable {adapter_name} LoRA parameters")
    return trainable
