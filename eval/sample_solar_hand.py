"""Teacher image-to-video inference with hand LoRA and sidecar conditioning."""

from __future__ import annotations

from pipelines.common.paths import path_default

import argparse
import gc
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from eval.io_utils import _remap_lora_keys, _write_mp4
from pipelines.ltx25_solar.hand_conditioning import HandConditioning
from pipelines.ltx25_solar.causal_attn import enable_causal_hand_motion, install_frame_causal_attention
from pipelines.ltx25_solar.hand_inject import (
    HandAwareBackbone,
    HandPatchifyProj,
    inject_distill_adapter,
    inject_hand_adapter,
    install_hand_patchify,
    peft_core,
)
from solarwm.backends.ltx25.checkpoint import inspect_base_checkpoint
from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY
from solarwm.backends.ltx25.inference import GuidanceSpec, InferenceSpec, build_inference_plan
from solarwm.backends.ltx25.official_codec import OfficialDiffVAEDecoder
from solarwm.backends.ltx25.torch_data import TorchBatch
from solarwm.backends.ltx25.torch_distributed import initialize
from solarwm.backends.ltx25.torch_inference import LTX25Sampler
from solarwm.backends.ltx25.torch_model import (
    LTX25SequenceParallelModel,
    inject_lora,
    load_strict_model,
)
from solarwm.inference.engine import InferenceCase

DEFAULT_CAMERA = (
    path_default("CAMERA")
)
DEFAULT_TRANSFORMER = (
    path_default("TRANSFORMER")
)
DEFAULT_VAE = path_default("VAE")
DEFAULT_RUN = path_default("RUN_DIR")
DEFAULT_CACHE = path_default("CACHE_DIR")


class OneShotSource:
    def __init__(self, batches: dict[str, TorchBatch]) -> None:
        self.batches = batches

    def get(self, sample_id: str) -> TorchBatch:
        return self.batches[sample_id]

    def case_fingerprint(self, sample_id: str, batch: TorchBatch | None = None) -> str:
        return sample_id

    def close(self) -> None:
        self.batches.clear()


class HandBoundModel(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.control: torch.Tensor | None = None
        self.motion: torch.Tensor | None = None

    def forward(self, *args, **kwargs):
        return self.inner(
            *args,
            **kwargs,
            hand_control_latent=self.control,
            hand_motion_features=self.motion,
        )


def _batch_from_cache(path: Path, *, zero_hand: bool) -> TorchBatch:
    tensors = load_file(str(path), device="cpu")
    first = tensors["first_frame_latent"].unsqueeze(0)
    video = tensors["video_latent"].unsqueeze(0)
    dummy = video.clone()
    dummy[:, :, :1] = first
    control = tensors["hand_control_latent"].unsqueeze(0)
    motion = tensors["hand_motion_features"].unsqueeze(0)
    if zero_hand:
        control = torch.zeros_like(control)
        motion = torch.zeros_like(motion)
    return TorchBatch(
        sample_id=path.stem,
        start_frame=0,
        plan_fingerprint=path.stem,
        video_latent=dummy,
        first_frame_latent=first,
        video_prompt_embeds=tensors["video_prompt_embeds"].unsqueeze(0),
        prompt_attention_mask=tensors["prompt_attention_mask"].unsqueeze(0),
        relative_w2c=tensors["relative_w2c"].unsqueeze(0),
        camera_k=tensors["camera_K"].unsqueeze(0),
        source_indices=torch.arange(153, dtype=torch.int64).unsqueeze(0),
        camera_source_indices=torch.tensor(
            STABLE_GEOMETRY.camera_pixel_indices, dtype=torch.int64
        ).unsqueeze(0),
    ), control, motion


def load_eval_model(
    *,
    run: Path,
    step: int,
    camera_adapter: str,
    transformer: str,
    hand_rank: int,
    device: torch.device,
    distill_run: Path | None = None,
    distill_step: int | None = None,
    distill_rank: int = 64,
    causal: bool = False,
    block_frames: int = 4,
) -> HandBoundModel:
    hand_lora_path = run / f"hand-lora-ema-step{step:05d}.safetensors"
    sidecar_path = run / f"hand-sidecar-step{step:05d}.safetensors"
    print(f"[hand-eval] ema={hand_lora_path}", flush=True)
    print(f"[hand-eval] sidecar={sidecar_path}", flush=True)
    inspection = inspect_base_checkpoint(transformer)
    loaded = load_strict_model(
        inspection,
        device=device,
        camera_translation_transform="linear",
        attention_backend="cudnn",
    )
    loaded.backbone.requires_grad_(False)
    core, camera_lora = inject_lora(loaded.core)
    raw = load_file(camera_adapter, device="cpu")
    camera_lora.load_state_dict(_remap_lora_keys(raw, set(camera_lora.keys)), broadcast=False)
    del raw
    hand_lora = inject_hand_adapter(core, rank=hand_rank)
    raw_hand = load_file(str(hand_lora_path), device="cpu")
    hand_lora.load_state_dict(_remap_lora_keys(raw_hand, set(hand_lora.keys)), broadcast=False)
    del raw_hand
    inner = peft_core(core)
    hand = HandConditioning(
        STABLE_GEOMETRY.latent_channels,
        int(inner.inner_dim),
        182,
        512,
    ).to(device=device, dtype=torch.bfloat16)
    side = load_file(str(sidecar_path), device="cpu")
    mapped = {key.removeprefix("hand_conditioning."): value for key, value in side.items()}
    hand.load_state_dict(mapped)
    hook = HandPatchifyProj(inner.patchify_proj, hand)
    install_hand_patchify(inner, hook)
    if distill_run is not None:
        dist_step = int(step if distill_step is None else distill_step)
        distill_path = Path(distill_run) / f"distill-lora-ema-step{dist_step:05d}.safetensors"
        if not distill_path.exists():
            dmd_path = Path(distill_run) / f"generator-ema-step{dist_step:05d}.safetensors"
            if dmd_path.exists():
                distill_path = dmd_path
        print(f"[hand-eval] distill={distill_path}", flush=True)
        distill_lora = inject_distill_adapter(core, rank=int(distill_rank))
        raw_d = load_file(str(distill_path), device="cpu")
        distill_lora.load_state_dict(_remap_lora_keys(raw_d, set(distill_lora.keys)), broadcast=False)
        del raw_d
        causal = True
    if causal:
        enable_causal_hand_motion(hook.hand)
        n = install_frame_causal_attention(inner, block_frames=int(block_frames))
        print(
            f"[hand-eval] causal=block-ar block_frames={block_frames} "
            f"hand_conv=causal wrappers={n}",
            flush=True,
        )
    loaded.backbone.transformer = LTX25SequenceParallelModel(core)
    model = HandBoundModel(HandAwareBackbone(loaded.backbone, hook).eval().requires_grad_(False))
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=DEFAULT_RUN)
    p.add_argument("--step", type=int, default=27307)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--out-dir", default="")
    p.add_argument("--camera-adapter", default=DEFAULT_CAMERA)
    p.add_argument("--transformer", default=DEFAULT_TRANSFORMER)
    p.add_argument("--vae", default=DEFAULT_VAE)
    p.add_argument("--hand-rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--stg", type=float, default=1.0)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--distill-run-dir", default="")
    p.add_argument("--distill-step", type=int, default=0)
    p.add_argument("--distill-rank", type=int, default=64)
    p.add_argument("--block-frames", type=int, default=4)
    p.add_argument(
        "--case",
        action="append",
        required=True,
        help="name:cache_filename:zero_hand (0|1). Repeat to sample multiple cases.",
    )
    return p.parse_args()


def _parse_cases(raw: list[str] | None) -> tuple[tuple[str, str, bool], ...]:
    if not raw:
        raise ValueError("Provide at least one --case name:cache_filename:zero_hand")
    parsed = []
    for spec in raw:
        name, filename, zero = spec.split(":", 2)
        parsed.append((name, filename, zero in ("1", "true", "True", "yes")))
    return tuple(parsed)


def main() -> None:
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / f"eval_step{args.step:05d}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    initialize(
        {
            "distributed": {
                "world_size": 1,
                "local_world_size": 1,
                "sequence_parallel_size": 1,
            }
        }
    )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    cases = _parse_cases(args.case)
    model = load_eval_model(
        run=run,
        step=args.step,
        camera_adapter=args.camera_adapter,
        transformer=args.transformer,
        hand_rank=args.hand_rank,
        device=device,
        distill_run=Path(args.distill_run_dir) if args.distill_run_dir else None,
        distill_step=args.distill_step or None,
        distill_rank=args.distill_rank,
        causal=bool(args.causal),
        block_frames=args.block_frames,
    )

    cache = Path(args.cache_dir)
    batches: dict[str, TorchBatch] = {}
    hands: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, filename, zero_hand in cases:
        batch, control, motion = _batch_from_cache(cache / filename, zero_hand=zero_hand)
        batches[name] = batch
        hands[name] = (
            control.to(device=device, dtype=torch.bfloat16),
            motion.to(device=device, dtype=torch.bfloat16),
        )
        print(
            f"[hand-eval] case={name} zero={zero_hand} motion_abs={float(motion.float().abs().mean()):.4f}",
            flush=True,
        )

    # Dummy negative: the cached prompt reused at CFG=1 would skip uncond; keep a zeros-like
    # caption from the first batch so CFG still runs.
    first = cases[0][0]
    negative = torch.zeros_like(batches[first].video_prompt_embeds).to(device)
    negative_mask = torch.zeros_like(batches[first].prompt_attention_mask).to(device)
    plan = build_inference_plan(
        InferenceSpec(seed=args.seed),
        GuidanceSpec(
            cfg_scale=float(args.cfg),
            stg_scale=float(args.stg),
            rescale_scale=0.7 if args.stg else 0.0,
            stg_blocks=(28,) if args.stg else (),
        ),
    )
    sampler = LTX25Sampler(model, OneShotSource(batches), negative, negative_mask, plan, device)

    import ltx_core.model.video_vae.transformer.attention as na_mod

    if not hasattr(na_mod.natten, "na3d"):
        na_mod.natten = None
        na_mod._NATTEN_AVAILABLE = False
        print("[hand-eval] DiffVAE: Triton/eager NA fallback", flush=True)

    decoder = OfficialDiffVAEDecoder(args.vae, device=device)
    wrote = []
    try:
        for name, _, _ in cases:
            dest = out / f"{name}.mp4"
            print(f"[hand-eval] sampling {name}", flush=True)
            model.control, model.motion = hands[name]
            case = InferenceCase(
                slot=0,
                sample_id=name,
                prompt=name,
                start_frame=0,
                noise_seed=args.seed,
                camera_fingerprint=name,
                metadata={},
            )
            latent = sampler.sample(case)
            frames = decoder.decode(latent.to(device=device), seed=case.noise_seed)
            _write_mp4(dest, frames, fps=24)
            wrote.append(dest.name)
            print(f"[hand-eval] wrote {dest}", flush=True)
    finally:
        decoder.close()

    receipt = {
        "step": args.step,
        "run": str(run),
        "cases": [name for name, _, _ in cases],
        "wrote": wrote,
        "cfg": args.cfg,
        "stg": args.stg,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    videos = "".join(
        f'<h3>{name}</h3><video controls muted loop src="{name}.mp4" width="768"></video>\n'
        for name, _, _ in cases
    )
    (out / "index.html").write_text(
        "<html><body style='background:#111;color:#eee;font-family:sans-serif'>\n"
        f"<h2>hand ema step {args.step}</h2>\n{videos}</body></html>\n"
    )
    print(f"[hand-eval] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
