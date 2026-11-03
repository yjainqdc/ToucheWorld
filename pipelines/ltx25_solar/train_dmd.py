"""Four-step Solar-Hand DMD with a shared LTX-2.5 backbone.

The frozen bidirectional teacher, causal generator, and bidirectional fake
score share base/camera/hand weights.  PEFT role switching keeps only one 22B
backbone resident per GPU while the student ``distill`` and ``critic`` LoRAs
remain disjoint trainable parameter sets.
"""

from __future__ import annotations

from pipelines.common.paths import path_default

import argparse
import functools
import itertools
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader, DistributedSampler

from pipelines.common.dmd_algo import calc_kl_gradient, dmd_fake_score_loss, dmd_student_loss
from pipelines.ltx25_solar.block_tf import (
    block_loss_mask,
    prepare_block_objective,
    zero_future_hand,
)
from pipelines.ltx25_solar.causal_attn import (
    denoise_block_ranges,
    install_frame_causal_attention,
    set_causal_window_frames,
)
from pipelines.ltx25_solar.dmd_roles import (
    CRITIC,
    STUDENT,
    TEACHER,
    assert_role_gradients,
    partition_dmd_parameters,
    use_dmd_role,
)
from pipelines.ltx25_solar.fewstep import fewstep_sigma_tensor
from pipelines.ltx25_solar.hand_conditioning import HandConditioning
from pipelines.ltx25_solar.hand_inject import (
    HandAwareBackbone,
    HandPatchifyProj,
    inject_distill_adapter,
    inject_hand_adapter,
    inject_named_adapter,
    install_hand_patchify,
    peft_core,
)
from pipelines.common.lora import LoRAEMA, _remap_lora_keys
from pipelines.ltx25_solar.train_hand import HandLatentCache, hand_region_weight
from solarwm.backends.ltx25.checkpoint import inspect_base_checkpoint
from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY
from solarwm.backends.ltx25.torch_distributed import (
    broadcast_sp_tensor,
    clip_replicated_gradient_norm,
    initialize,
    synchronize_replicated_gradients,
)
from solarwm.backends.ltx25.torch_flow import sample_shifted_logit_normal
from solarwm.backends.ltx25.torch_inference import _perturbation
from solarwm.backends.ltx25.torch_model import (
    LTX25SequenceParallelModel,
    SolarLTX25VideoTransformerBlock,
    inject_lora,
    load_strict_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument(
        "--jsonl",
        default=path_default("JSONL"),
    )
    p.add_argument("--roles", default="follow,grow,still_empty,still_hands")
    p.add_argument("--still-empty-limit", type=int, default=20)
    p.add_argument("--still-hands-limit", type=int, default=20)
    p.add_argument("--hand-run-dir", required=True)
    p.add_argument("--hand-step", type=int, default=8000)
    p.add_argument("--causal-run-dir", required=True)
    p.add_argument("--causal-step", type=int, default=2000)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--adapter",
        default=(
            path_default("CAMERA")
        ),
    )
    p.add_argument(
        "--transformer",
        default=(
            path_default("TRANSFORMER")
        ),
    )
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--student-steps", type=int, default=4)
    p.add_argument("--block-frames", type=int, default=4)
    p.add_argument("--window-frames", type=int, default=8)
    p.add_argument("--hand-rank", type=int, default=32)
    p.add_argument("--distill-rank", type=int, default=64)
    p.add_argument("--critic-rank", type=int, default=64)
    p.add_argument("--generator-lr", type=float, default=2e-5)
    p.add_argument("--critic-lr", type=float, default=2e-5)
    p.add_argument("--critic-warmup", type=int, default=20)
    p.add_argument("--gen-update-ratio", type=int, default=5)
    p.add_argument("--sft-weight", type=float, default=4.0)
    p.add_argument("--fkl-weight", type=float, default=1.0)
    p.add_argument("--real-fkl-weight", type=float, default=0.5)
    p.add_argument("--teacher-cfg", type=float, default=3.0)
    p.add_argument("--teacher-stg", type=float, default=1.0)
    p.add_argument("--teacher-rescale", type=float, default=0.7)
    p.add_argument("--score-sigma-min", type=float, default=0.02)
    p.add_argument("--score-sigma-max", type=float, default=0.98)
    p.add_argument("--hand-dropout", type=float, default=0.1)
    p.add_argument("--hand-region-weight", type=float, default=0.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--sp-size", type=int, default=1)
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-contract", action="store_true")
    return p.parse_args()


def _flow_x0(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    first: torch.Tensor,
) -> torch.Tensor:
    scale = sigma.reshape(-1, 1, 1, 1, 1).to(torch.float32)
    x0 = noisy.float() - scale * velocity.float()
    x0[:, :, :1] = first.float()
    return x0.to(noisy.dtype)


def _mix_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    first: torch.Tensor,
) -> torch.Tensor:
    scale = sigma.reshape(-1, 1, 1, 1, 1).to(torch.float32)
    mixed = (1.0 - scale) * clean.float() + scale * noise.float()
    mixed[:, :, :1] = first.float()
    return mixed.to(clean.dtype)


def _guided_velocity(
    conditioned: torch.Tensor,
    unconditioned: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    cfg: float,
    stg: float,
    rescale: float,
) -> torch.Tensor:
    cond = conditioned.float()
    guided = cond + (float(cfg) - 1.0) * (cond - unconditioned.float())
    guided = guided + float(stg) * (cond - perturbed.float())
    if not math.isclose(float(rescale), 0.0):
        std = guided.std().clamp_min(1.0e-8)
        guided = guided * (float(rescale) * cond.std() / std + 1.0 - float(rescale))
    return guided.to(conditioned.dtype)


def _batch_to_device(batch: dict, device: torch.device, hand_dropout: float) -> dict:
    result = {
        "video": batch["video_latent"].to(device=device, dtype=torch.bfloat16),
        "first": batch["first_frame_latent"].to(device=device, dtype=torch.bfloat16),
        "caption": batch["video_prompt_embeds"].to(device=device, dtype=torch.bfloat16),
        "mask": batch["prompt_attention_mask"].to(device=device, dtype=torch.int64),
        "viewmats": batch["relative_w2c"].to(device=device, dtype=torch.float32),
        "camera_k": batch["camera_K"].to(device=device, dtype=torch.float32),
        "control": batch["hand_control_latent"].to(device=device, dtype=torch.bfloat16),
        "motion": batch["hand_motion_features"].to(device=device, dtype=torch.bfloat16),
    }
    if float(torch.rand((), device=device)) < float(hand_dropout):
        result["control"] = torch.zeros_like(result["control"])
        result["motion"] = torch.zeros_like(result["motion"])
    return result


def _context(batch_size: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros(
        batch_size,
        STABLE_GEOMETRY.latent_frames,
        device=device,
        dtype=torch.bool,
    )
    value[:, :1] = True
    return value


def _forward(
    model: torch.nn.Module,
    core: torch.nn.Module,
    hand: torch.nn.Module,
    role,
    data: dict,
    latent: torch.Tensor,
    sigma: torch.Tensor,
    context: torch.Tensor,
    *,
    control: torch.Tensor | None = None,
    motion: torch.Tensor | None = None,
    caption: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    perturbed: bool = False,
) -> torch.Tensor:
    with use_dmd_role(core, hand, role):
        return model(
            video_latent=latent,
            sigma=sigma.reshape(latent.shape[0]),
            caption_embedding=data["caption"] if caption is None else caption,
            first_frame_mask=context,
            camera={"viewmats": data["viewmats"], "K": data["camera_k"]},
            caption_mask=data["mask"] if mask is None else mask,
            hand_control_latent=data["control"] if control is None else control,
            hand_motion_features=data["motion"] if motion is None else motion,
            perturbations=_perturbation(
                perturbed=perturbed,
                blocks=(28,),
                device=latent.device,
            ),
        )


def _student_rollout(
    model: torch.nn.Module,
    core: torch.nn.Module,
    hand: torch.nn.Module,
    data: dict,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    block_frames: int,
    grad_block: int | None,
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """Full block-AR rollout; only one sampled block retains a graph."""

    state = noise.detach().clone()
    state[:, :, :1] = data["first"]
    pieces = [data["first"]]
    grad_span = None
    ranges = denoise_block_ranges(STABLE_GEOMETRY.latent_frames, block_frames)
    for block_index, (start, end) in enumerate(ranges):
        current = state[:, :, start:end]
        context = torch.ones(
            state.shape[0],
            STABLE_GEOMETRY.latent_frames,
            device=state.device,
            dtype=torch.bool,
        )
        context[:, start:end] = False
        control, motion = zero_future_hand(data["control"], data["motion"], end)
        for hop, (sigma, sigma_next) in enumerate(itertools.pairwise(sigmas)):
            keep_graph = grad_block == block_index and hop == len(sigmas) - 2
            grad_context = torch.enable_grad() if keep_graph else torch.no_grad()
            full = state.detach().clone()
            full[:, :, start:end] = current
            with grad_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                velocity = _forward(
                    model,
                    core,
                    hand,
                    STUDENT,
                    data,
                    full,
                    sigma.expand(full.shape[0]),
                    context,
                    control=control,
                    motion=motion,
                )
                updated = current.float() + velocity[:, :, start:end].float() * (
                    sigma_next.float() - sigma.float()
                )
                current = updated.to(state.dtype)
            if not keep_graph:
                current = current.detach()
        if grad_block == block_index:
            grad_span = (start, end)
            pieces.append(current)
        else:
            pieces.append(current.detach())
        state[:, :, start:end] = current.detach()
    generated = torch.cat(pieces, dim=2)
    if tuple(generated.shape) != tuple(data["video"].shape):
        raise RuntimeError(f"DMD rollout shape mismatch: {tuple(generated.shape)}")
    return generated, grad_span


@torch.no_grad()
def _teacher_and_fake_x0(
    model: torch.nn.Module,
    core: torch.nn.Module,
    hand: torch.nn.Module,
    data: dict,
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy = _mix_noise(clean, noise, sigma, data["first"])
    context = _context(noisy.shape[0], noisy.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cond = _forward(model, core, hand, TEACHER, data, noisy, sigma, context)
        uncond = _forward(
            model,
            core,
            hand,
            TEACHER,
            data,
            noisy,
            sigma,
            context,
            caption=torch.zeros_like(data["caption"]),
            mask=torch.zeros_like(data["mask"]),
        )
        stg = _forward(
            model,
            core,
            hand,
            TEACHER,
            data,
            noisy,
            sigma,
            context,
            perturbed=True,
        )
        real_velocity = _guided_velocity(
            cond,
            uncond,
            stg,
            cfg=args.teacher_cfg,
            stg=args.teacher_stg,
            rescale=args.teacher_rescale,
        )
        fake_velocity = _forward(model, core, hand, CRITIC, data, noisy, sigma, context)
    return (
        _flow_x0(noisy, real_velocity, sigma, data["first"]),
        _flow_x0(noisy, fake_velocity, sigma, data["first"]),
        noisy,
    )


def _masked_mse(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(device=left.device, dtype=torch.float32)
    error = (left.float() - right.float()).square() * weight
    return error.sum() / weight.sum().clamp_min(1.0)


def _sft_anchor(
    model: torch.nn.Module,
    core: torch.nn.Module,
    hand: torch.nn.Module,
    data: dict,
    start: int,
    end: int,
) -> torch.Tensor:
    noise = torch.randn_like(data["video"])
    sigma = sample_shifted_logit_normal(
        data["video"].shape[0],
        device=data["video"].device,
        std=1.0,
        epsilon=1.0e-3,
        uniform_probability=0.1,
    )
    objective, context = prepare_block_objective(
        data["video"], data["first"], noise, sigma, start=start, end=end
    )
    control, motion = zero_future_hand(data["control"], data["motion"], end)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = _forward(
            model,
            core,
            hand,
            STUDENT,
            data,
            objective.noisy,
            sigma,
            context,
            control=control,
            motion=motion,
        )
    region = hand_region_weight(control, 0.0)
    loss_mask = block_loss_mask(start, end, prediction) * region.to(
        prediction.device, torch.float32
    )
    return _masked_mse(prediction, objective.target_velocity, loss_mask)


def _fkl_anchor(
    model: torch.nn.Module,
    core: torch.nn.Module,
    hand: torch.nn.Module,
    data: dict,
    clean: torch.Tensor,
    sigma: torch.Tensor,
    score_noise: torch.Tensor,
    real_x0: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    noisy = _mix_noise(clean.detach(), score_noise, sigma, data["first"])
    context = _context(noisy.shape[0], noisy.device)
    control, motion = zero_future_hand(data["control"], data["motion"], end)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        velocity = _forward(
            model,
            core,
            hand,
            STUDENT,
            data,
            noisy,
            sigma,
            context,
            control=control,
            motion=motion,
        )
    student_x0 = _flow_x0(noisy, velocity, sigma, data["first"])
    return _masked_mse(
        student_x0,
        real_x0.detach(),
        block_loss_mask(start, end, student_x0),
    )


def _sync_and_clip(parameters: list[torch.nn.Parameter]) -> None:
    synchronize_replicated_gradients(tuple(parameters))
    clip_replicated_gradient_norm(tuple(parameters), 1.0)


def _save(
    out: Path,
    step: int,
    student_lora,
    critic_lora,
    student_ema: LoRAEMA,
) -> None:
    save_file(student_lora.state_dict(), str(out / f"generator-live-step{step:05d}.safetensors"))
    save_file(student_ema.tensors(), str(out / f"generator-ema-step{step:05d}.safetensors"))
    save_file(critic_lora.state_dict(), str(out / f"critic-live-step{step:05d}.safetensors"))


def main() -> None:
    args = parse_args()
    if args.student_steps != 4:
        raise ValueError("Solar-Hand DMD production contract is four student steps")
    if not 0.0 < args.score_sigma_min < args.score_sigma_max < 1.0:
        raise ValueError("score sigma range must lie strictly inside (0,1)")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hand_root = Path(args.hand_run_dir)
    causal_root = Path(args.causal_run_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ.setdefault("LOCAL_WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE") or str(world))
    runtime = initialize(
        {
            "distributed": {
                "world_size": world,
                "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
                "sequence_parallel_size": int(args.sp_size),
            },
            "train": {"fsdp": {"sharding_strategy": "FULL_SHARD"}},
        }
    )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + runtime.dp_rank)
    set_causal_window_frames(int(args.window_frames))

    dataset = HandLatentCache(
        Path(args.cache_dir),
        jsonl=Path(args.jsonl),
        still_empty_limit=args.still_empty_limit,
        still_hands_limit=args.still_hands_limit,
        seed=args.seed,
        roles=tuple(x.strip() for x in args.roles.split(",") if x.strip()),
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=runtime.dp_world_size,
            rank=runtime.dp_rank,
            shuffle=True,
            seed=args.seed,
        )
        if runtime.dp_world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        generator=torch.Generator().manual_seed(args.seed) if sampler is None else None,
        num_workers=0,
        pin_memory=True,
    )

    inspection = inspect_base_checkpoint(args.transformer)
    loaded = load_strict_model(
        inspection,
        device=device,
        camera_translation_transform="linear",
        attention_backend="cudnn",
    )
    loaded.backbone.requires_grad_(False)
    core, camera_lora = inject_lora(loaded.core)
    raw = load_file(args.adapter, device="cpu")
    camera_lora.load_state_dict(_remap_lora_keys(raw, set(camera_lora.keys)), broadcast=True)
    del raw
    hand_lora = inject_hand_adapter(core, rank=args.hand_rank)
    hand_path = hand_root / f"hand-lora-ema-step{args.hand_step:05d}.safetensors"
    side_path = hand_root / f"hand-sidecar-step{args.hand_step:05d}.safetensors"
    raw = load_file(str(hand_path), device="cpu")
    hand_lora.load_state_dict(_remap_lora_keys(raw, set(hand_lora.keys)), broadcast=False)
    del raw
    student_lora = inject_distill_adapter(core, rank=args.distill_rank)
    causal_path = causal_root / f"distill-lora-ema-step{args.causal_step:05d}.safetensors"
    raw = load_file(str(causal_path), device="cpu")
    student_lora.load_state_dict(_remap_lora_keys(raw, set(student_lora.keys)), broadcast=False)
    del raw
    critic_lora = inject_named_adapter(core, "critic", rank=args.critic_rank, zero_output=True)

    inner = peft_core(core)
    install_frame_causal_attention(inner, block_frames=args.block_frames)
    hand = HandConditioning(
        STABLE_GEOMETRY.latent_channels,
        int(inner.inner_dim),
        182,
        512,
    ).to(device=device, dtype=torch.bfloat16)
    hook = HandPatchifyProj(inner.patchify_proj, hand)
    install_hand_patchify(inner, hook)
    side = load_file(str(side_path), device="cpu")
    hook.hand.load_state_dict(
        {key.removeprefix("hand_conditioning."): value for key, value in side.items()}
    )
    loaded.backbone.transformer = LTX25SequenceParallelModel(core)
    model = HandAwareBackbone(loaded.backbone, hook)
    student_params, critic_params = partition_dmd_parameters(model)

    if args.activation_checkpoint:
        blocks = [m for m in model.modules() if isinstance(m, SolarLTX25VideoTransformerBlock)]
        selected = {id(m) for m in blocks}
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda module: id(module) in selected,
        )
    model.train()
    student_optim = torch.optim.AdamW(
        student_params,
        lr=args.generator_lr,
        betas=(0.9, 0.95),
        weight_decay=1.0e-4,
    )
    critic_optim = torch.optim.AdamW(
        critic_params,
        lr=args.critic_lr,
        betas=(0.9, 0.95),
        weight_decay=1.0e-4,
    )
    student_ema = LoRAEMA(student_lora, decay=args.ema_decay)
    if args.start_step:
        for runtime_lora, stem in (
            (student_lora, "generator-live"),
            (critic_lora, "critic-live"),
        ):
            path = out / f"{stem}-step{args.start_step:05d}.safetensors"
            raw = load_file(str(path), device="cpu")
            runtime_lora.load_state_dict(
                _remap_lora_keys(raw, set(runtime_lora.keys)),
                broadcast=False,
            )
        student_ema.load_shadow(
            out / f"generator-ema-step{args.start_step:05d}.safetensors",
            set(student_lora.keys),
        )
    if runtime.rank == 0:
        shutil.copy2(side_path, out / side_path.name)
        print(
            f"[dmd] clips={len(dataset)} roles={dataset.role_counts} world={world} "
            f"teacher=bidir hand={hand_root.name}@{args.hand_step} "
            f"student=causal:{causal_root.name}@{args.causal_step} critic=bidir-zero "
            f"trainable_student={sum(p.numel() for p in student_params)/1e6:.1f}M "
            f"trainable_critic={sum(p.numel() for p in critic_params)/1e6:.1f}M",
            flush=True,
        )

    sigmas = fewstep_sigma_tensor(4, device=device)
    ranges = denoise_block_ranges(STABLE_GEOMETRY.latent_frames, args.block_frames)
    iterator = iter(loader)
    epoch = 0
    step = int(args.start_step)
    generator_updates = 0
    running_critic = 0.0
    running_generator = 0.0
    started = time.time()
    while step < args.max_steps:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            raw_batch = next(iterator)
        data = _batch_to_device(raw_batch, device, args.hand_dropout)
        for value in data.values():
            broadcast_sp_tensor(value)
        rollout_noise = torch.randn_like(data["video"])
        score_noise = torch.randn_like(data["video"])
        broadcast_sp_tensor(rollout_noise)
        broadcast_sp_tensor(score_noise)

        # Critic: learn the score of the current four-step generator distribution.
        student_optim.zero_grad(set_to_none=True)
        critic_optim.zero_grad(set_to_none=True)
        with torch.no_grad():
            generated, _ = _student_rollout(
                model,
                core,
                hook.hand,
                data,
                rollout_noise,
                sigmas,
                block_frames=args.block_frames,
                grad_block=None,
            )
        sigma = torch.empty(data["video"].shape[0], device=device).uniform_(
            args.score_sigma_min, args.score_sigma_max
        )
        noisy = _mix_noise(generated, score_noise, sigma, data["first"])
        context = _context(noisy.shape[0], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            critic_velocity = _forward(
                model, core, hook.hand, CRITIC, data, noisy, sigma, context
            )
            critic_loss = dmd_fake_score_loss(
                critic_velocity[:, :, 1:],
                score_noise[:, :, 1:],
                generated[:, :, 1:],
            )
        # Non-reentrant activation checkpointing recomputes transformer blocks
        # during backward, so the same adapter role must remain active.
        with use_dmd_role(core, hook.hand, CRITIC):
            critic_loss.backward()
        assert_role_gradients(model, expected="critic")
        _sync_and_clip(critic_params)
        critic_optim.step()
        running_critic += float(critic_loss.detach())

        generator_loss_value = 0.0
        update_generator = (
            step + 1 > args.critic_warmup
            and (step + 1 - args.critic_warmup) % args.gen_update_ratio == 0
        )
        if update_generator:
            student_optim.zero_grad(set_to_none=True)
            critic_optim.zero_grad(set_to_none=True)
            grad_block = int(torch.randint(len(ranges), (), device=device).item())
            generated, grad_span = _student_rollout(
                model,
                core,
                hook.hand,
                data,
                rollout_noise,
                sigmas,
                block_frames=args.block_frames,
                grad_block=grad_block,
            )
            if grad_span is None:
                raise RuntimeError("DMD generator rollout retained no gradient block")
            start, end = grad_span
            score_sigma = torch.empty(data["video"].shape[0], device=device).uniform_(
                args.score_sigma_min, args.score_sigma_max
            )
            real_x0, fake_x0, _ = _teacher_and_fake_x0(
                model,
                core,
                hook.hand,
                data,
                generated.detach(),
                score_noise,
                score_sigma,
                args,
            )
            kl_gradient = calc_kl_gradient(
                fake_x0,
                real_x0,
                generated.detach(),
                normalize_mode="per_sample",
            ).detach()
            grad_mask = torch.zeros_like(generated, dtype=torch.bool)
            grad_mask[:, :, start:end] = True
            dmd_loss = dmd_student_loss(generated, kl_gradient, gradient_mask=grad_mask)
            sft_loss = _sft_anchor(model, core, hook.hand, data, start, end)
            fkl_loss = _fkl_anchor(
                model,
                core,
                hook.hand,
                data,
                generated,
                score_sigma,
                score_noise,
                real_x0,
                start,
                end,
            )
            real_noise = torch.randn_like(data["video"])
            real_sigma = torch.empty(data["video"].shape[0], device=device).uniform_(
                args.score_sigma_min, args.score_sigma_max
            )
            real_teacher_x0, _, _ = _teacher_and_fake_x0(
                model,
                core,
                hook.hand,
                data,
                data["video"],
                real_noise,
                real_sigma,
                args,
            )
            real_fkl_loss = _fkl_anchor(
                model,
                core,
                hook.hand,
                data,
                data["video"],
                real_sigma,
                real_noise,
                real_teacher_x0,
                start,
                end,
            )
            generator_loss = (
                dmd_loss
                + args.sft_weight * sft_loss
                + args.fkl_weight * fkl_loss
                + args.real_fkl_weight * real_fkl_loss
            )
            with use_dmd_role(core, hook.hand, STUDENT):
                generator_loss.backward()
            assert_role_gradients(model, expected="distill")
            _sync_and_clip(student_params)
            student_optim.step()
            student_ema.update(student_lora)
            generator_updates += 1
            generator_loss_value = float(generator_loss.detach())
            running_generator += generator_loss_value

        step += 1
        if runtime.rank == 0 and (step == args.start_step + 1 or step % 5 == 0):
            print(
                f"[dmd] step={step}/{args.max_steps} critic={float(critic_loss.detach()):.4f} "
                f"critic_mean={running_critic/(step-args.start_step):.4f} "
                f"generator={generator_loss_value:.4f} gen_updates={generator_updates} "
                f"gen_mean={running_generator/max(1,generator_updates):.4f} "
                f"elapsed={(time.time()-started)/60:.1f}m",
                flush=True,
            )
        should_save = step % args.save_every == 0 or step == args.max_steps
        if runtime.rank == 0 and should_save:
            _save(out, step, student_lora, critic_lora, student_ema)
            print(f"[dmd] saved step={step}", flush=True)
        if dist.is_initialized() and should_save:
            dist.barrier()
        if args.smoke_contract and step >= 2:
            break

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
