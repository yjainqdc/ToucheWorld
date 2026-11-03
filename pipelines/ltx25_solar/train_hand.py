"""Train a new LTX-2.5 hand LoRA + sidecar; freeze Solar Stage0.5 camera LoRA."""

from __future__ import annotations

from pipelines.common.paths import path_default

import argparse
import functools
import os
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from pipelines.ltx25_solar.hand_conditioning import HandConditioning
from pipelines.ltx25_solar.hand_geometry import clip_cache_name, iter_jsonl_rows
from pipelines.ltx25_solar.hand_inject import (
    HandAwareBackbone,
    HandPatchifyProj,
    freeze_camera_train_hand,
    inject_hand_adapter,
    install_hand_patchify,
    peft_core,
)
from pipelines.common.lora import LoRAEMA, _remap_lora_keys
from solarwm.backends.ltx25.checkpoint import inspect_base_checkpoint
from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY
from solarwm.backends.ltx25.torch_distributed import (
    broadcast_sp_tensor,
    clip_replicated_gradient_norm,
    initialize,
    synchronize_replicated_gradients,
)
from solarwm.backends.ltx25.torch_flow import (
    prepare_objective,
    sample_shifted_logit_normal,
    velocity_loss,
)
from solarwm.backends.ltx25.torch_model import (
    LTX25SequenceParallelModel,
    SolarLTX25VideoTransformerBlock,
    inject_lora,
    load_strict_model,
)


class HandLatentCache(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        *,
        jsonl: Path | None = None,
        still_empty_limit: int = 0,
        still_hands_limit: int = 0,
        seed: int = 42,
        roles: tuple[str, ...] = ("follow", "grow", "still_empty", "still_hands"),
    ) -> None:
        cache_dir = Path(cache_dir)
        allowed = {str(role) for role in roles}
        if jsonl is None:
            self.paths = sorted(
                path for path in cache_dir.glob("*.safetensors") if not path.name.startswith("_")
            )
            self.role_counts = {"all": len(self.paths)}
        else:
            rows = iter_jsonl_rows(Path(jsonl))
            by_role: dict[str, list[Path]] = {}
            missing: list[str] = []
            for row in rows:
                role = str(row.get("role") or "unknown")
                if role not in allowed:
                    continue
                path = cache_dir / f"{clip_cache_name(str(row['record_id']))}.safetensors"
                if not path.is_file():
                    missing.append(path.name)
                    continue
                by_role.setdefault(role, []).append(path)
            if missing:
                raise FileNotFoundError(f"missing {len(missing)} cache files, e.g. {missing[0]}")
            still_empty = list(by_role.get("still_empty") or [])
            still_hands = list(by_role.get("still_hands") or [])
            if still_empty_limit and len(still_empty) > still_empty_limit:
                rng = torch.Generator().manual_seed(int(seed))
                perm = torch.randperm(len(still_empty), generator=rng).tolist()
                still_empty = [still_empty[i] for i in perm[: int(still_empty_limit)]]
                by_role["still_empty"] = still_empty
            if still_hands_limit and len(still_hands) > still_hands_limit:
                rng = torch.Generator().manual_seed(int(seed) + 1)
                perm = torch.randperm(len(still_hands), generator=rng).tolist()
                still_hands = [still_hands[i] for i in perm[: int(still_hands_limit)]]
                by_role["still_hands"] = still_hands
            self.paths = sorted({path for paths in by_role.values() for path in paths})
            self.role_counts = {role: len(paths) for role, paths in sorted(by_role.items())}
        if not self.paths:
            raise FileNotFoundError(f"no latents in {cache_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tensors = load_file(str(self.paths[index]), device="cpu")
        return {key: tensor.clone() for key, tensor in tensors.items()}


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
    p.add_argument("--resume-dir", default="")
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
    p.add_argument("--hand-dropout", type=float, default=0.1)
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema-decay", type=float, default=0.999)
    return p.parse_args()


def sidecar_tensors(hook: HandPatchifyProj) -> dict[str, torch.Tensor]:
    return {f"hand_conditioning.{key}": value.detach() for key, value in hook.hand.state_dict().items()}


def hand_region_weight(control: torch.Tensor, extra: float) -> torch.Tensor:
    """Boost latent cells where the skeleton-control VAE has energy. Shape [B,1,T,H,W]."""

    if extra <= 0:
        return torch.ones(
            (control.shape[0], 1, control.shape[2], control.shape[3], control.shape[4]),
            device=control.device,
            dtype=torch.float32,
        )
    energy = control.detach().float().abs().mean(dim=1, keepdim=True)
    baseline = energy.mean(dim=(3, 4), keepdim=True).clamp_min(1e-8)
    occupied = (energy > (1.25 * baseline)).to(dtype=torch.float32)
    return 1.0 + float(extra) * occupied


def weighted_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    per_element = (prediction.float() - target.float()).square()
    loss_mask = torch.ones_like(per_element)
    loss_mask[:, :, 0] = 0.0
    loss_mask = loss_mask * region.to(device=prediction.device, dtype=torch.float32)
    return (per_element * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
            f"[hand] clips={len(ds)} roles={ds.role_counts} lr={args.lr} steps={args.max_steps} "
            f"sp={runtime.sp_size} dp={runtime.dp_world_size} rank={args.hand_rank} "
            f"dropout={args.hand_dropout} act_ckpt={args.activation_checkpoint} "
            f"hand_region_w={args.hand_region_weight} roles={args.roles}",
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
    inner = peft_core(core)
    hand = HandConditioning(
        STABLE_GEOMETRY.latent_channels,
        int(inner.inner_dim),
        182,
        512,
    ).to(device=device, dtype=torch.bfloat16)
    hook = HandPatchifyProj(inner.patchify_proj, hand)
    install_hand_patchify(inner, hook)
    if int(args.start_step) > 0:
        resume_root = Path(args.resume_dir) if args.resume_dir else out
        live_path = resume_root / f"hand-lora-live-step{int(args.start_step):05d}.safetensors"
        ema_path = resume_root / f"hand-lora-ema-step{int(args.start_step):05d}.safetensors"
        side_path = resume_root / f"hand-sidecar-step{int(args.start_step):05d}.safetensors"
        raw_hand = load_file(str(live_path), device="cpu")
        hand_lora.load_state_dict(_remap_lora_keys(raw_hand, set(hand_lora.keys)), broadcast=False)
        del raw_hand
        side = load_file(str(side_path), device="cpu")
        mapped = {key.removeprefix("hand_conditioning."): value for key, value in side.items()}
        hook.hand.load_state_dict(mapped)
        if runtime.rank == 0:
            print(f"[hand] resume live={live_path.name} sidecar={side_path.name}", flush=True)
    loaded.backbone.transformer = LTX25SequenceParallelModel(core)
    model = HandAwareBackbone(loaded.backbone, hook)
    model.train()
    trainable = freeze_camera_train_hand(model)
    if runtime.rank == 0:
        n = sum(p.numel() for p in trainable)
        print(f"[hand] trainable={n/1e6:.1f}M camera_lora_frozen=1", flush=True)

    if args.activation_checkpoint:
        blocks = [
            module
            for module in model.modules()
            if isinstance(module, SolarLTX25VideoTransformerBlock)
        ]
        if runtime.rank == 0:
            print(f"[hand] activation_ckpt_blocks={len(blocks)}", flush=True)
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
    ema = LoRAEMA(hand_lora, decay=args.ema_decay)
    if int(args.start_step) > 0:
        resume_root = Path(args.resume_dir) if args.resume_dir else out
        ema_path = resume_root / f"hand-lora-ema-step{int(args.start_step):05d}.safetensors"
        ema.load_shadow(ema_path, set(hand_lora.keys))
        if runtime.rank == 0:
            print(f"[hand] resume ema={ema_path.name}", flush=True)
    if runtime.rank == 0:
        print(f"[hand] ema_decay={args.ema_decay} (hand LoRA only)", flush=True)

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
            ):
                broadcast_sp_tensor(tensor)
            objective = prepare_objective(video, first, noise, sigma)
            first_frame_mask = torch.zeros(
                (video.shape[0], STABLE_GEOMETRY.latent_frames),
                device=device,
                dtype=torch.bool,
            )
            first_frame_mask[:, 0] = True
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
            loss = weighted_velocity_loss(prediction, objective.target_velocity, region)
            (loss / (args.grad_accum * max(1, runtime.sp_size))).backward()
            total_loss += float(loss.detach().item())
        if runtime.sp_size > 1:
            synchronize_replicated_gradients(tuple(trainable))
            clip_replicated_gradient_norm(tuple(trainable), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        ema.update(hand_lora)
        step += 1
        seen += 1
        running += total_loss
        if runtime.rank == 0 and (step % 10 == 0 or step == args.start_step + 1):
            dt = time.time() - t0
            occ = float((region > 1.0).float().mean().detach().item()) if args.hand_region_weight else 0.0
            print(
                f"[hand] step={step}/{args.max_steps} loss={total_loss:.4f} "
                f"mean={running/seen:.4f} lr={optim.param_groups[0]['lr']:.2e} "
                f"hand_occ={occ:.3f} elapsed={dt/60:.1f}m",
                flush=True,
            )
        if runtime.rank == 0 and (step % args.save_every == 0 or step == args.max_steps):
            live_path = out / f"hand-lora-live-step{step:05d}.safetensors"
            ema_path = out / f"hand-lora-ema-step{step:05d}.safetensors"
            side_path = out / f"hand-sidecar-step{step:05d}.safetensors"
            save_file(hand_lora.state_dict(), str(live_path))
            save_file(ema.tensors(), str(ema_path))
            save_file(sidecar_tensors(hook), str(side_path))
            print(f"[hand] saved {live_path.name} + {ema_path.name} + {side_path.name}", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
