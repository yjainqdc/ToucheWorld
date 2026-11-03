"""Causal block-autoregressive inference over a JSONL evaluation manifest."""

from __future__ import annotations

from pipelines.common.paths import path_default

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from eval.sample_solar_hand import OneShotSource, _batch_from_cache, load_eval_model
from pipelines.ltx25_solar.block_rollout import BlockARSampler
from eval.io_utils import write_h264
from pipelines.ltx25_solar.hand_geometry import clip_cache_name
from solarwm.backends.ltx25.inference import GuidanceSpec, InferenceSpec, build_inference_plan
from solarwm.backends.ltx25.official_codec import OfficialDiffVAEDecoder
from solarwm.backends.ltx25.torch_distributed import initialize
from solarwm.inference.engine import InferenceCase

DEFAULT_HAND = Path(path_default("HAND_RUN"))
DEFAULT_DISTILL = Path(path_default("DISTILL_RUN"))
DEFAULT_CAMERA = (
    path_default("CAMERA")
)
DEFAULT_TRANSFORMER = (
    path_default("TRANSFORMER")
)
DEFAULT_VAE = path_default("VAE")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=path_default("BENCH_ROOT"))
    p.add_argument("--jsonl", default="")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--hand-run", default=str(DEFAULT_HAND))
    p.add_argument("--hand-step", type=int, default=27307)
    p.add_argument("--distill-run", default=str(DEFAULT_DISTILL))
    p.add_argument("--distill-step", type=int, default=10240)
    p.add_argument("--distill-rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--stg", type=float, default=1.0)
    p.add_argument("--block-frames", type=int, default=4)
    p.add_argument("--window-frames", type=int, default=8)
    p.add_argument("--denoise-steps", type=int, default=30)
    p.add_argument("--shard", default="0/1")
    p.add_argument(
        "--prefix-latents",
        type=int,
        default=None,
        help="Override jsonl prefix. 1=first frame only (HOI4D Q1/Q4). "
        "Unset uses the row (FinalBench Q3 is 8 ≈ 2s GT).",
    )
    p.add_argument(
        "--claim",
        action="store_true",
        help="Ignore shard split; exclusively claim the next missing mp4 (multi-GPU).",
    )
    p.add_argument("--camera-adapter", default=DEFAULT_CAMERA)
    p.add_argument("--transformer", default=DEFAULT_TRANSFORMER)
    p.add_argument("--vae", default=DEFAULT_VAE)
    args = p.parse_args()

    root = Path(args.root)
    jsonl = Path(args.jsonl) if args.jsonl else root / "eval.jsonl"
    cache = Path(args.cache_dir) if args.cache_dir else root / "cache"
    out = Path(args.out_dir) if args.out_dir else root / f"gens_causal_{args.distill_step}"
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    shard_i, shard_n = (int(v) for v in str(args.shard).split("/"))
    if not args.claim:
        rows = [row for i, row in enumerate(rows) if i % shard_n == shard_i]
    pending = []
    for row in rows:
        name = clip_cache_name(row["record_id"])
        dest = out / f"{name}.mp4"
        cache_path = cache / f"{name}.safetensors"
        if dest.is_file():
            continue
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        pending.append((name, row, cache_path, dest))
    print(
        f"[bench-causal] pending={len(pending)}/{len(rows)} out={out} claim={int(args.claim)}",
        flush=True,
    )
    if not pending:
        return

    for key, value in (("LOCAL_RANK", "0"), ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "1")):
        os.environ.setdefault(key, value)
    initialize({"distributed": {"world_size": 1, "local_world_size": 1, "sequence_parallel_size": 1}})
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = load_eval_model(
        run=Path(args.hand_run),
        step=args.hand_step,
        camera_adapter=args.camera_adapter,
        transformer=args.transformer,
        hand_rank=32,
        device=device,
        distill_run=Path(args.distill_run),
        distill_step=args.distill_step,
        distill_rank=args.distill_rank,
        causal=True,
        block_frames=args.block_frames,
    )
    import ltx_core.model.video_vae.transformer.attention as na_mod

    if not hasattr(na_mod.natten, "na3d"):
        na_mod.natten = None
        na_mod._NATTEN_AVAILABLE = False
        print("[bench-causal] DiffVAE: Triton/eager NA fallback", flush=True)
    decoder = OfficialDiffVAEDecoder(args.vae, device=device)
    source = OneShotSource({})
    sampler = None
    wrote = []
    try:
        for name, row, cache_path, dest in pending:
            if dest.is_file():
                continue
            claim = dest.with_suffix(".claim") if args.claim else None
            if claim is not None:
                try:
                    fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    print(f"[bench-causal] skip claimed {name}", flush=True)
                    continue
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
            try:
                batch, control, motion = _batch_from_cache(cache_path, zero_hand=False)
                source.batches.clear()
                source.batches[name] = batch
                if sampler is None:
                    negative = torch.zeros_like(batch.video_prompt_embeds).to(device)
                    negative_mask = torch.zeros_like(batch.prompt_attention_mask).to(device)
                    plan = build_inference_plan(
                        InferenceSpec(seed=args.seed),
                        GuidanceSpec(
                            cfg_scale=float(args.cfg),
                            stg_scale=float(args.stg),
                            rescale_scale=0.7 if args.stg else 0.0,
                            stg_blocks=(28,) if args.stg else (),
                        ),
                    )
                    sampler = BlockARSampler(
                        model,
                        source,
                        negative,
                        negative_mask,
                        plan,
                        device,
                        block_frames=args.block_frames,
                        window_frames=args.window_frames,
                        kv_cache=True,
                        denoise_steps=args.denoise_steps,
                    )
                model.control = control.to(device=device, dtype=torch.bfloat16)
                model.motion = motion.to(device=device, dtype=torch.bfloat16)
                print(f"[bench-causal] {name} motion_abs={float(motion.float().abs().mean()):.4f}", flush=True)
                prefix = (
                    int(args.prefix_latents)
                    if args.prefix_latents is not None
                    else int(row.get("prefix_latents") or 1)
                )
                case = InferenceCase(
                    slot=0,
                    sample_id=name,
                    prompt=row.get("prompt") or name,
                    start_frame=0,
                    noise_seed=args.seed,
                    camera_fingerprint=name,
                    metadata={"generation_mode": "block_ar", "prefix_latents": prefix},
                )
                print(f"[bench-causal] {name} prefix_latents={prefix}", flush=True)
                with torch.inference_mode():
                    latent = sampler.sample(case)
                    frames = decoder.decode(latent.to(device=device), seed=args.seed)
                rgb = (
                    frames[0].permute(1, 2, 3, 0).clamp(0, 1).float().cpu().numpy() * 255.0
                ).round().astype(np.uint8)
                bgr = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb], axis=0)
                write_h264(dest, bgr)
                wrote.append(dest.name)
                print(f"[bench-causal] wrote {dest}", flush=True)
            finally:
                if claim is not None and claim.is_file():
                    claim.unlink()
    finally:
        decoder.close()
    receipt_name = "receipt.json"
    if args.claim:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "x").replace(",", "-")
        receipt_name = f"receipt_gpu{visible}_{os.getpid()}.json"
    (out / receipt_name).write_text(
        json.dumps(
            {
                "hand": args.hand_run,
                "hand_step": args.hand_step,
                "distill": args.distill_run,
                "distill_step": args.distill_step,
                "wrote": wrote,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[bench-causal] done {len(wrote)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
