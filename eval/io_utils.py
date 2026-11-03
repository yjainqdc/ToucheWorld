"""Tiny I/O helpers so inference does not pull in the 8-arm battery script."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


def _write_mp4(path: Path, frames_bcthw: torch.Tensor, fps: int = 24) -> None:
    video = frames_bcthw[0].permute(1, 2, 3, 0).clamp(0, 1).cpu().numpy()
    height, width = video.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    for frame in video:
        bgr = cv2.cvtColor((frame * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def _remap_lora_keys(values: dict[str, torch.Tensor], expected: set[str]) -> dict[str, torch.Tensor]:
    if set(values) == expected:
        return values
    remapped = {}
    for key, tensor in values.items():
        name = key
        for prefix in ("base_model.model.", "base_model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        remapped[name] = tensor
    if set(remapped) != expected:
        missing = sorted(expected - set(remapped))[:8]
        extra = sorted(set(remapped) - expected)[:8]
        raise KeyError(f"LoRA keys differ; missing={missing} extra={extra}")
    return remapped


def write_h264(path: Path, bgr: np.ndarray, fps: float = 24.0) -> None:
    """Write T×H×W×3 BGR. Prefer ffmpeg H.264; fall back to OpenCV mp4v."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frames, height, width, channels = bgr.shape
    if channels != 3:
        raise ValueError(f"expected BGR HxWx3, got {bgr.shape}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        tmp = path.with_suffix(".tmp.mp4")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-crf", "18", "-preset", "veryfast",
            "-movflags", "+faststart", str(tmp),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        proc.stdin.write(np.ascontiguousarray(bgr).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed {path}")
        tmp.replace(path)
        return
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    for frame in bgr:
        writer.write(frame)
    writer.release()
