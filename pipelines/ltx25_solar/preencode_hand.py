"""Preencode EgoStandard H3-hand mix clips into Solar LTX-2.5 latent cache."""

from __future__ import annotations

from pipelines.common.paths import path_default

import argparse
import fcntl
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from pipelines.ltx25_solar.hand_geometry import (
    clip_cache_name,
    is_still_role,
    iter_jsonl_rows,
    load_c2w_and_k,
    load_cover_rgb,
    load_cover_still,
    resample_motion_to_ltx,
    zero_motion_ltx,
)
from solarwm.backends.ltx25.camera import prepare_latent_camera
from solarwm.backends.ltx25.geometry import cover_resize
from solarwm.backends.ltx25.official_codec import OfficialOnlineCodec
from solarwm.backends.ltx25.torch_distributed import initialize

EMPTY_CONTROL_NAME = "_shared_empty_hand_control.safetensors"


def _bf16(art) -> torch.Tensor:
    raw = np.frombuffer(art.data, dtype="<u2").copy()
    return torch.from_numpy(raw).view(torch.bfloat16).reshape(art.shape).clone()


def _i64(art) -> torch.Tensor:
    raw = np.frombuffer(art.data, dtype="<i8").copy()
    return torch.from_numpy(raw).to(torch.int64).reshape(art.shape).clone()


def shared_empty_control_latent(
    *,
    codec,
    cache_dir: Path,
    control_path: Path,
    resize,
) -> torch.Tensor:
    dest = cache_dir / EMPTY_CONTROL_NAME
    lock_path = dest.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.is_file():
            return load_file(str(dest), device="cpu")["hand_control_latent"].clone()
        pixels = load_cover_still(control_path, resize)
        control_art, _ = codec.encode_video(pixels)
        tensor = _bf16(control_art)
        tmp = dest.with_suffix(".tmp")
        save_file({"hand_control_latent": tensor}, str(tmp))
        tmp.replace(dest)
        print(f"[preencode-hand] wrote shared empty control {dest.name}", flush=True)
        return tensor.clone()


def encode_row(
    row: dict,
    *,
    codec,
    cache_dir: Path,
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> Path | None:
    record = clip_cache_name(str(row["record_id"]))
    dest = cache_dir / f"{record}.safetensors"
    if dest.is_file():
        return dest
    still = is_still_role(str(row.get("role") or ""))
    video_path = Path(row["video"])
    control_path = Path(row["hand_control"])
    motion_path = Path(row["hand_motion"])
    camera_path = Path(row["camera"])
    needed = (video_path, control_path, camera_path) if still else (
        video_path,
        control_path,
        motion_path,
        camera_path,
    )
    for path in needed:
        if not path.is_file():
            print(f"[preencode-hand] missing {path}", flush=True)
            return None
    try:
        probe = cv2.VideoCapture(str(video_path))
        ok, frame = probe.read()
        probe.release()
        if not ok:
            raise FileNotFoundError(video_path)
        height, width = frame.shape[:2]
        resize = cover_resize(height, width)
        rgb = load_cover_still(video_path, resize) if still else load_cover_rgb(video_path, resize)
        c2w, k_canvas = load_c2w_and_k(camera_path)
        relative, k_lat = prepare_latent_camera(
            c2w,
            k_canvas,
            convention="absolute_c2w",
            resize=resize,
        )
        magnitudes = np.linalg.norm(relative[:, :3, 3], axis=-1)
        if float(magnitudes.max()) > 20.0 or float(np.abs(relative).max()) > 20.0:
            print(f"[preencode-hand] skip {record}: camera travel out of range", flush=True)
            return None
        motion = zero_motion_ltx() if still else resample_motion_to_ltx(motion_path)
        caption = str(row.get("prompt") or "").strip() if still else str(
            row.get("prompt_action") or row.get("prompt") or ""
        ).strip()
        if not caption:
            caption = "First-person video of two hands working at a task."
        video_art, first_art = codec.encode_video(rgb)
        if still:
            control_latent = shared_empty_control_latent(
                codec=codec,
                cache_dir=cache_dir,
                control_path=control_path,
                resize=resize,
            )
        else:
            control = load_cover_rgb(control_path, resize)
            control_art, _ = codec.encode_video(control)
            control_latent = _bf16(control_art)
        cached = prompt_cache.get(caption)
        if cached is None:
            prompt_art, mask_art = codec.encode_prompt(caption)
            cached = (_bf16(prompt_art), _i64(mask_art))
            prompt_cache[caption] = cached
        prompt, mask = cached
    except Exception as exc:
        print(f"[preencode-hand] skip {record}: {type(exc).__name__}: {exc}", flush=True)
        return None

    tensors = {
        "video_latent": _bf16(video_art),
        "first_frame_latent": _bf16(first_art),
        "hand_control_latent": control_latent,
        "hand_motion_features": torch.from_numpy(np.ascontiguousarray(motion)).to(torch.float32),
        "video_prompt_embeds": prompt.clone(),
        "prompt_attention_mask": mask.clone(),
        "relative_w2c": torch.from_numpy(np.ascontiguousarray(relative)).to(torch.float32),
        "camera_K": torch.from_numpy(np.ascontiguousarray(k_lat)).to(torch.float32),
    }
    tmp = dest.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(dest)
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jsonl",
        default=path_default("JSONL"),
    )
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"),
                   help="Root for relative video, hand, and camera paths in the manifest.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--transformer",
        default=(
            path_default("TRANSFORMER")
        ),
    )
    p.add_argument(
        "--vae",
        default=path_default("VAE"),
    )
    p.add_argument(
        "--gemma",
        default=(
            path_default("GEMMA")
        ),
    )
    args = p.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOCAL_RANK", "0")
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
    codec = OfficialOnlineCodec(
        transformer_path=args.transformer,
        video_vae_path=args.vae,
        gemma4_path=args.gemma,
        device=device,
        identity=f"h3hand-solar-preencode-s{args.shard_id}",
    )
    rows = iter_jsonl_rows(
        Path(args.jsonl),
        limit=args.limit,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    for row in rows:
        for field in ("video", "hand_control", "hand_motion", "camera"):
            value = Path(row[field])
            if not value.is_absolute():
                row[field] = str(data_root / value)

    pending = [
        row
        for row in rows
        if not (cache / f"{clip_cache_name(str(row['record_id']))}.safetensors").is_file()
    ]
    print(
        f"[preencode-hand] shard={args.shard_id}/{args.num_shards} "
        f"rows={len(rows)} pending={len(pending)}",
        flush=True,
    )
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    ok = 0
    for i, row in enumerate(pending):
        dest = encode_row(row, codec=codec, cache_dir=cache, prompt_cache=prompt_cache)
        if dest is not None:
            ok += 1
        if (i + 1) % 10 == 0 or dest is None:
            print(
                f"[preencode-hand] {i+1}/{len(pending)} ok={ok} last={dest.name if dest else 'skip'}",
                flush=True,
            )
    print(
        f"[preencode-hand] done shard={args.shard_id} wrote={ok}/{len(pending)} "
        f"prompt_cache={len(prompt_cache)}",
        flush=True,
    )
    codec.close()


if __name__ == "__main__":
    main()
