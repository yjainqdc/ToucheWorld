"""Block-AR teacher forcing: history clean, denoise one chunk, causal hand conv."""

from __future__ import annotations

from pipelines.common.paths import path_default

import argparse
import functools
import os
import shutil
import time
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

from pipelines.ltx25_solar.block_tf import (
    block_loss_mask,
    prepare_block_objective,
    sample_denoise_span,
    zero_future_hand,
)
from pipelines.ltx25_solar.causal_attn import (
    enable_causal_hand_motion,
    install_frame_causal_attention,
)
from pipelines.ltx25_solar.hand_conditioning import HandConditioning
from pipelines.ltx25_solar.hand_inject import (
    HandAwareBackbone,
    HandPatchifyProj,
    freeze_all_train_adapter,
    inject_distill_adapter,
    inject_hand_adapter,
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
    p.add_argument("--still-empty-limit", type=int, default=20)
    p.add_argument("--still-hands-limit", type=int, default=20)
    p.add_argument("--roles", default="follow,grow,still_empty,still_hands")
    p.add_argument("--hand-region-weight", type=float, default=0.0)
    p.add_argument("--hand-run-dir", required=True)
    p.add_argument("--hand-step", type=int, default=8000)
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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--sp-size", type=int, default=1)
    p.add_argument("--hand-rank", type=int, default=32)
    p.add_argument("--distill-rank", type=int, default=64)
    p.add_argument("--block-frames", type=int, default=4)
    p.add_argument("--hand-dropout", type=float, default=0.1)
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema-decay", type=float, default=0.999)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hand_root = Path(args.hand_run_dir)
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
    torch.manual_seed(args.seed)

    ds = HandLatentCache(
        Path(args.cache_dir),
        jsonl=Path(args.jsonl) if args.jsonl else None,
        still_empty_limit=args.still_empty_limit,
        still_hands_limit=args.still_hands_limit,
        seed=args.seed,
        roles=tuple(role.strip() for role in args.roles.split(",") if role.strip()),
    )
    sampler = None
    if runtime.dp_world_size > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=runtime.dp_world_size,
            rank=runtime.dp_rank,
            shuffle=True,
            seed=args.seed,
        )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        generator=None if sampler is not None else generator,
        num_workers=0,
        pin_memory=True,
    )

    if runtime.rank == 0:
        print(
            f"[causal] clips={len(ds)} roles={ds.role_counts} lr={args.lr} "
            f"steps={args.max_steps} distill_rank={args.distill_rank} "
            f"hand={hand_root.name}@{args.hand_step} causal=block-ar block_frames={args.block_frames}",
            flush=True,
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
    hand_path = hand_root / f"hand-lora-ema-step{int(args.hand_step):05d}.safetensors"
    side_path = hand_root / f"hand-sidecar-step{int(args.hand_step):05d}.safetensors"
    raw_hand = load_file(str(hand_path), device="cpu")
    hand_lora.load_state_dict(_remap_lora_keys(raw_hand, set(hand_lora.keys)), broadcast=False)
    del raw_hand
    distill_lora = inject_distill_adapter(core, rank=args.distill_rank)
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
    mapped = {key.removeprefix("hand_conditioning."): value for key, value in side.items()}
    hook.hand.load_state_dict(mapped)
    enable_causal_hand_motion(hook.hand)
    if runtime.rank == 0:
        dest_side = out / side_path.name
        if not dest_side.exists():
            shutil.copy2(side_path, dest_side)
        print(f"[causal] frozen hand={hand_path.name} sidecar={side_path.name}", flush=True)
    if int(args.start_step) > 0:
        live_path = out / f"distill-lora-live-step{int(args.start_step):05d}.safetensors"
        raw_d = load_file(str(live_path), device="cpu")
        distill_lora.load_state_dict(_remap_lora_keys(raw_d, set(distill_lora.keys)), broadcast=False)
        del raw_d
        if runtime.rank == 0:
            print(f"[causal] resume live={live_path.name}", flush=True)
    loaded.backbone.transformer = LTX25SequenceParallelModel(core)
    model = HandAwareBackbone(loaded.backbone, hook)
    model.train()
    trainable = freeze_all_train_adapter(model, "distill")
    if runtime.rank == 0:
        n = sum(p.numel() for p in trainable)
        print(
            f"[causal] trainable={n/1e6:.1f}M camera_frozen=1 hand_frozen=1 sidecar_frozen=1",
            flush=True,
        )

    if args.activation_checkpoint:
        blocks = [
            module
            for module in model.modules()
            if isinstance(module, SolarLTX25VideoTransformerBlock)
        ]
        if runtime.rank == 0:
            print(f"[causal] activation_ckpt_blocks={len(blocks)}", flush=True)
        selected = {id(module) for module in blocks}
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda module: id(module) in selected,
        )

    use_ddp = runtime.dp_world_size > 1 and runtime.sp_size == 1
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )
        engine = model.module
    else:
        engine = model
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    engine.train()
    ema = LoRAEMA(distill_lora, decay=args.ema_decay)
    if int(args.start_step) > 0:
        ema_path = out / f"distill-lora-ema-step{int(args.start_step):05d}.safetensors"
        ema.load_shadow(ema_path, set(distill_lora.keys))
        if runtime.rank == 0:
            print(f"[causal] resume ema={ema_path.name}", flush=True)

    step = int(args.start_step)
    running = 0.0
    seen = 0
    t0 = time.time()
    data_iter = iter(loader)
    epoch = 0
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                batch = next(data_iter)
            video = batch["video_latent"].to(device=device, dtype=torch.bfloat16)
            first = batch["first_frame_latent"].to(device=device, dtype=torch.bfloat16)
            caption = batch["video_prompt_embeds"].to(device=device, dtype=torch.bfloat16)
            mask = batch["prompt_attention_mask"].to(device=device, dtype=torch.int64)
            viewmats = batch["relative_w2c"].to(device=device, dtype=torch.float32)
            camera_k = batch["camera_K"].to(device=device, dtype=torch.float32)
            control = batch["hand_control_latent"].to(device=device, dtype=torch.bfloat16)
            motion = batch["hand_motion_features"].to(device=device, dtype=torch.bfloat16)
            if float(torch.rand((), generator=None)) < args.hand_dropout:
                control = torch.zeros_like(control)
                motion = torch.zeros_like(motion)
            noise = torch.randn_like(video)
            sigma = sample_shifted_logit_normal(
                video.shape[0],
                device=device,
                std=1.0,
                epsilon=1.0e-3,
                uniform_probability=0.1,
            )
            start, end = sample_denoise_span(block_frames=args.block_frames)
            span = torch.tensor([start, end], device=device, dtype=torch.int64)
            for tensor in (
                video,
                first,
                caption,
                mask,
                viewmats,
                camera_k,
                control,
                motion,
                noise,
                sigma,
                span,
            ):
                broadcast_sp_tensor(tensor)
            start, end = int(span[0].item()), int(span[1].item())
            control, motion = zero_future_hand(control, motion, end)
            objective, first_frame_mask = prepare_block_objective(
                video, first, noise, sigma, start=start, end=end
            )
            warmup = min(1.0, (step + 1) / max(1, args.warmup))
            for group in optim.param_groups:
                group["lr"] = args.lr * warmup
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = engine(
                    video_latent=objective.noisy,
                    sigma=objective.sigma,
                    caption_embedding=caption,
                    first_frame_mask=first_frame_mask,
                    camera={"viewmats": viewmats, "K": camera_k},
                    caption_mask=mask,
                    hand_control_latent=control,
                    hand_motion_features=motion,
                )
            region = hand_region_weight(control, args.hand_region_weight)
            loss_mask = block_loss_mask(start, end, prediction) * region.to(
                device=prediction.device, dtype=torch.float32
            )
            per_element = (prediction.float() - objective.target_velocity.float()).square()
            loss = (per_element * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            (loss / (args.grad_accum * max(1, runtime.sp_size))).backward()
            total_loss += float(loss.detach().item())
        if runtime.sp_size > 1:
            synchronize_replicated_gradients(tuple(trainable))
            clip_replicated_gradient_norm(tuple(trainable), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        ema.update(distill_lora)
        step += 1
        seen += 1
        running += total_loss
        if runtime.rank == 0 and (step % 10 == 0 or step == args.start_step + 1):
            dt = time.time() - t0
            print(
                f"[causal] step={step}/{args.max_steps} loss={total_loss:.4f} "
                f"mean={running/seen:.4f} lr={optim.param_groups[0]['lr']:.2e} "
                f"block=[{start},{end}) elapsed={dt/60:.1f}m",
                flush=True,
            )
        if runtime.rank == 0 and (step % args.save_every == 0 or step == args.max_steps):
            live_path = out / f"distill-lora-live-step{step:05d}.safetensors"
            ema_path = out / f"distill-lora-ema-step{step:05d}.safetensors"
            save_file(distill_lora.state_dict(), str(live_path))
            save_file(ema.tensors(), str(ema_path))
            print(f"[causal] saved {live_path.name} + {ema_path.name}", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
