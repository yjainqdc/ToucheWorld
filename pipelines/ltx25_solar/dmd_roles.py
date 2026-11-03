"""Role switching and gradient contracts for shared-backbone Solar-Hand DMD."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from pipelines.ltx25_solar.causal_attn import (
    causal_attention_enabled,
    set_causal_attention_enabled,
)


@dataclass(frozen=True)
class DMDRole:
    name: str
    adapters: tuple[str, ...]
    causal: bool


TEACHER = DMDRole("teacher", ("default", "hand"), False)
STUDENT = DMDRole("student", ("default", "hand", "distill"), True)
CRITIC = DMDRole("critic", ("default", "hand", "critic"), False)
ROLES = {role.name: role for role in (TEACHER, STUDENT, CRITIC)}


def _set_adapters(core: torch.nn.Module, names: tuple[str, ...]) -> None:
    available = set(getattr(core, "peft_config", {}))
    missing = set(names) - available
    if missing:
        raise RuntimeError(f"missing DMD adapters: {sorted(missing)}")
    tuner = getattr(core, "base_model", None)
    setter = getattr(tuner, "set_adapter", None)
    if setter is None:
        raise RuntimeError("PEFT backbone has no multi-adapter setter")
    setter(list(names))
    # PEFT's set_adapter() marks every activated adapter trainable. Restore the
    # DMD contract after every role switch: camera/hand always stay frozen.
    for parameter_name, parameter in core.named_parameters():
        trainable = (
            "lora_A.distill" in parameter_name
            or "lora_B.distill" in parameter_name
            or "lora_A.critic" in parameter_name
            or "lora_B.critic" in parameter_name
        )
        parameter.requires_grad_(trainable)


@contextmanager
def use_dmd_role(
    core: torch.nn.Module,
    hand: torch.nn.Module,
    role: DMDRole,
) -> Iterator[None]:
    """Activate exactly one DMD role on a shared frozen backbone."""

    previous = tuple(getattr(core, "active_adapters", ()))
    previous_causal = causal_attention_enabled()
    encoder = getattr(hand, "motion_encoder", None)
    if encoder is None or not hasattr(encoder, "causal"):
        raise RuntimeError("hand sidecar lacks causal mode")
    previous_hand_causal = bool(encoder.causal)
    _set_adapters(core, role.adapters)
    set_causal_attention_enabled(role.causal)
    encoder.causal = role.causal
    try:
        yield
    finally:
        if previous:
            _set_adapters(core, previous)
        set_causal_attention_enabled(previous_causal)
        encoder.causal = previous_hand_causal


def partition_dmd_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Freeze shared weights and return disjoint student/critic parameters."""

    student: list[torch.nn.Parameter] = []
    critic: list[torch.nn.Parameter] = []
    seen_student: set[int] = set()
    seen_critic: set[int] = set()
    for name, parameter in model.named_parameters():
        is_student = "lora_A.distill" in name or "lora_B.distill" in name
        is_critic = "lora_A.critic" in name or "lora_B.critic" in name
        if is_student and is_critic:
            raise RuntimeError(f"parameter belongs to two DMD roles: {name}")
        parameter.requires_grad_(is_student or is_critic)
        if is_student and id(parameter) not in seen_student:
            student.append(parameter)
            seen_student.add(id(parameter))
        if is_critic and id(parameter) not in seen_critic:
            critic.append(parameter)
            seen_critic.add(id(parameter))
    if not student or not critic:
        raise RuntimeError("DMD needs non-empty distill and critic adapters")
    if seen_student & seen_critic:
        raise RuntimeError("student and critic parameters overlap")
    return student, critic


def assert_role_gradients(
    model: torch.nn.Module,
    *,
    expected: str,
) -> None:
    """Fail a smoke run if gradients escaped the selected adapter."""

    token = f".{expected}"
    found = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        found += 1
        if token not in name:
            raise RuntimeError(f"{expected} update leaked gradient into {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"{expected} gradient is non-finite: {name}")
    if found <= 0:
        raise RuntimeError(f"{expected} update produced no gradients")
