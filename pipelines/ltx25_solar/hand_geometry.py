"""H3 158f/1344×768 pack → Solar LTX-2.5 153f/512×768 + 20-latent motion."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY, causal_camera_pixel_indices

TARGET = STABLE_GEOMETRY.pixel_frames
MOTION_FEATURE_DIM = 182


def clip_cache_name(record_id: str) -> str:
    return str(record_id).replace(":", "_")


def iter_jsonl_rows(
    path: Path,
    *,
    limit: int = 0,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit and limit > 0:
        rows = rows[:limit]
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise ValueError("shard_id out of range")
    return [row for i, row in enumerate(rows) if i % num_shards == shard_id]


def _pad_hold(frames: np.ndarray, target: int) -> np.ndarray:
    if len(frames) >= target:
        return frames[:target]
    return np.concatenate(
        [frames, np.repeat(frames[-1:], target - len(frames), axis=0)],
        axis=0,
    )


def _cover_frame(frame: np.ndarray, resize) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scaled = cv2.resize(
        rgb,
        (resize.resized_width, resize.resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    crop = scaled[
        resize.crop_top : resize.crop_top + STABLE_GEOMETRY.height,
        resize.crop_left : resize.crop_left + STABLE_GEOMETRY.width,
    ]
    return crop.astype(np.float32) / 255.0


def load_cover_rgb(path: Path, resize) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_cover_frame(frame, resize))
    cap.release()
    if not frames:
        raise FileNotFoundError(path)
    stacked = np.stack(frames, axis=0)
    return _pad_hold(stacked, TARGET)


def load_cover_still(path: Path, resize) -> np.ndarray:
    """Decode one pixel frame and tile to 153. H.264 stills are not bit-identical."""

    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(path)
    crop = _cover_frame(frame, resize)
    return np.repeat(crop[None], TARGET, axis=0)


def zero_motion_ltx() -> np.ndarray:
    return np.zeros((STABLE_GEOMETRY.latent_frames, MOTION_FEATURE_DIM), dtype=np.float32)


def is_still_role(role: str) -> bool:
    return str(role) in ("still_empty", "still_hands")


def quat_wxyz_to_R(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_c2w_and_k(camera_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(camera_path) as source:
        t = np.asarray(source["c2w_t"], dtype=np.float64)
        quat = np.asarray(source["c2w_quat_wxyz"], dtype=np.float64)
        k = np.asarray(source["K_canvas"], dtype=np.float64)
    if t.shape[0] < 1 or quat.shape[0] != t.shape[0] or k.shape != (3, 3):
        raise ValueError(f"unexpected camera.npz layout in {camera_path}")
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], t.shape[0], axis=0)
    for i in range(t.shape[0]):
        c2w[i, :3, :3] = quat_wxyz_to_R(quat[i])
        c2w[i, :3, 3] = t[i]
    return _pad_hold(c2w, TARGET), k


def window_local_frame_ids(
    anchors: np.ndarray,
    source_frame_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Map MCAP-absolute hand stamps onto the 0..157 clip window.

    EgoStandard ``anchor_source_frame_ids`` are absolute episode indices
    (360, 540, …). Solar samples LTX latents at local pixel frames
    ``0,1,9,…,145``. Subtract the window origin so nearest-neighbour
    lookup does not collapse onto row 0.
    """

    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.size == 0:
        raise ValueError("anchor_source_frame_ids is empty")
    local = anchors
    if source_frame_ids is not None and len(source_frame_ids) > 0:
        origin = int(np.asarray(source_frame_ids, dtype=np.int64)[0])
        shifted = anchors - origin
        if int(shifted.min()) >= -2 and int(shifted.max()) <= TARGET + 24:
            local = shifted
    if int(local.min()) >= 0 and int(local.max()) <= TARGET + 8:
        return np.clip(local, 0, None)
    return anchors - int(anchors.min())


def resample_motion_to_ltx(motion_path: Path) -> np.ndarray:
    with np.load(motion_path) as source:
        features = np.asarray(source["motion_features"], dtype=np.float32)
        anchors = np.asarray(source["anchor_source_frame_ids"], dtype=np.int64)
        source_ids = (
            np.asarray(source["source_frame_ids"], dtype=np.int64)
            if "source_frame_ids" in source.files
            else None
        )
    if features.ndim != 2 or features.shape[1] != MOTION_FEATURE_DIM:
        raise ValueError(f"motion_features must be [T,182], got {features.shape}")
    if anchors.shape[0] != features.shape[0]:
        raise ValueError("anchor_source_frame_ids must match motion_features rows")
    local = window_local_frame_ids(anchors, source_ids)
    ltx_px = np.asarray(causal_camera_pixel_indices(TARGET), dtype=np.int64)
    index = np.abs(local[:, None] - ltx_px[None, :]).argmin(axis=0)
    sampled = features[index]
    if sampled.shape != (STABLE_GEOMETRY.latent_frames, MOTION_FEATURE_DIM):
        raise ValueError(f"resampled motion shape {sampled.shape}")
    if not np.isfinite(sampled).all():
        raise ValueError("non-finite motion features")
    return sampled
