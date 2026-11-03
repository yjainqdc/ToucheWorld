"""Block-causal attention for LTX-2.5 video self-attention.

Tokens pack as ``(f h w)``. Latent frame 0 is its own context block. Remaining
frames are grouped into chunks of ``block_frames``. A query in block ``i`` may
attend to every token in blocks ``0..i`` (bidirectional inside the chunk) and
to none in later chunks. Text stays on cross-attention.

The mask is always the full ``video_tokens × video_tokens`` document so sequence
parallel all-to-all still matches.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY

# 0 = attend the full causal past. >0 = sliding window in latent frames.
_WINDOW_FRAMES = 0
# A shared Solar-Hand DMD backbone switches between a causal generator and
# bidirectional teacher/critic adapters.  The wrapper stays installed while
# this flag selects whether it contributes a mask.
_CAUSAL_ENABLED = True


def causal_window_frames() -> int:
    return int(_WINDOW_FRAMES)


def set_causal_window_frames(window_frames: int) -> None:
    global _WINDOW_FRAMES
    if int(window_frames) < 0:
        raise ValueError("window_frames must be >= 0")
    _WINDOW_FRAMES = int(window_frames)


def causal_attention_enabled() -> bool:
    return bool(_CAUSAL_ENABLED)


def set_causal_attention_enabled(enabled: bool) -> None:
    global _CAUSAL_ENABLED
    _CAUSAL_ENABLED = bool(enabled)


def latent_block_ids(
    latent_frames: int,
    block_frames: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Block index per latent frame. Frame 0 is block 0; then groups of ``block_frames``."""

    if latent_frames <= 0 or block_frames <= 0:
        raise ValueError("latent_frames and block_frames must be positive")
    ids = torch.zeros(latent_frames, device=device, dtype=torch.int64)
    if latent_frames > 1:
        ids[1:] = 1 + torch.arange(latent_frames - 1, device=device, dtype=torch.int64) // int(
            block_frames
        )
    return ids


def denoise_block_ranges(latent_frames: int, block_frames: int) -> tuple[tuple[int, int], ...]:
    """Half-open ``[start, end)`` ranges that teacher-forcing may denoise (never frame 0)."""

    if latent_frames <= 1:
        raise ValueError("need at least two latent frames for block AR")
    ranges = []
    start = 1
    while start < latent_frames:
        end = min(start + int(block_frames), latent_frames)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def window_history_frames(start: int, window_frames: int, latent_frames: int) -> tuple[int, ...]:
    """Committed frames a current block starting at ``start`` may attend."""

    if start <= 0:
        return ()
    if int(window_frames) <= 0:
        return tuple(range(0, start))
    seen = {0}
    seen.update(range(max(1, start - int(window_frames)), start))
    return tuple(f for f in sorted(seen) if 0 <= f < min(start, latent_frames))


def frame_token_index(frames: tuple[int, ...] | list[int], *, tokens_per_latent: int | None = None) -> torch.Tensor:
    per = int(STABLE_GEOMETRY.tokens_per_latent if tokens_per_latent is None else tokens_per_latent)
    if not frames:
        return torch.zeros(0, dtype=torch.int64)
    return torch.cat(
        [torch.arange(int(f) * per, (int(f) + 1) * per, dtype=torch.int64) for f in frames]
    )


def frame_block_keep_mask(
    *,
    latent_frames: int | None = None,
    tokens_per_latent: int | None = None,
    block_frames: int = 1,
    window_frames: int | None = None,
    keep_first_frame: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean keep mask ``[S, S]``: True = attend.

    ``window_frames`` limits how far back a query may look, in latent frames.
    Frame 0 stays visible when ``keep_first_frame`` is set (i2v). Same-block
    tokens stay bidirectional. ``window_frames=0`` keeps the full causal past.
    """

    frames = int(STABLE_GEOMETRY.latent_frames if latent_frames is None else latent_frames)
    per = int(STABLE_GEOMETRY.tokens_per_latent if tokens_per_latent is None else tokens_per_latent)
    ids = latent_block_ids(frames, int(block_frames), device=device)
    token_ids = ids.repeat_interleave(per)
    keep = token_ids[:, None] >= token_ids[None, :]
    limit = int(causal_window_frames() if window_frames is None else window_frames)
    if limit <= 0:
        return keep
    frame_ix = torch.arange(frames, device=device, dtype=torch.int64).repeat_interleave(per)
    nearby = (frame_ix[:, None] - frame_ix[None, :]) <= limit
    same = token_ids[:, None] == token_ids[None, :]
    first = frame_ix[None, :] == 0 if keep_first_frame else torch.zeros_like(keep)
    return keep & (nearby | same | first)


def frame_causal_additive_mask(
    *,
    dtype: torch.dtype,
    device: torch.device,
    latent_frames: int | None = None,
    tokens_per_latent: int | None = None,
    block_frames: int = 1,
    window_frames: int | None = None,
) -> torch.Tensor:
    """Additive SDPA bias ``[1, 1, S, S]``: 0 keep, ``finfo.min`` block."""

    keep = frame_block_keep_mask(
        latent_frames=latent_frames,
        tokens_per_latent=tokens_per_latent,
        block_frames=block_frames,
        window_frames=window_frames,
        device=device,
    )
    bias = torch.zeros(keep.shape, device=device, dtype=dtype)
    bias.masked_fill_(~keep, torch.finfo(dtype).min)
    return bias.unsqueeze(0).unsqueeze(0)


class _CausalPrepare:
    def __init__(self, original: Any, *, block_frames: int) -> None:
        self.original = original
        self.block_frames = int(block_frames)
        self._cache: dict[tuple[str, str, int, int, int], torch.Tensor] = {}

    def __call__(self, modality: Any, cross_modality: Any = None) -> Any:
        arguments = self.original(modality, cross_modality)
        if not causal_attention_enabled():
            return arguments
        device = arguments.x.device
        dtype = arguments.x.dtype
        frames = STABLE_GEOMETRY.latent_frames
        per = STABLE_GEOMETRY.tokens_per_latent
        window = causal_window_frames()
        key = (str(device), str(dtype), frames, per, self.block_frames, window)
        mask = self._cache.get(key)
        if mask is None or mask.device != device or mask.dtype != dtype:
            mask = frame_causal_additive_mask(
                dtype=dtype,
                device=device,
                latent_frames=frames,
                tokens_per_latent=per,
                block_frames=self.block_frames,
                window_frames=window,
            )
            self._cache[key] = mask
        return replace(arguments, self_attention_mask=mask)


def install_frame_causal_attention(core: torch.nn.Module, *, block_frames: int = 4) -> int:
    """Wrap every video-args preprocessor so attn1 always sees the block mask."""

    if int(block_frames) <= 0:
        raise ValueError("block_frames must be positive")
    seen: set[int] = set()
    wrapped = 0
    stack: list[Any] = [core]
    visited: set[int] = set()
    while stack:
        module = stack.pop()
        ident = id(module)
        if ident in visited:
            continue
        visited.add(ident)
        preprocessor = getattr(module, "video_args_preprocessor", None)
        candidates = [preprocessor, getattr(preprocessor, "simple_preprocessor", None)]
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            prepare = getattr(candidate, "prepare", None)
            if prepare is None:
                continue
            if isinstance(prepare, _CausalPrepare):
                prepare.block_frames = int(block_frames)
                prepare._cache.clear()
            else:
                candidate.prepare = _CausalPrepare(prepare, block_frames=int(block_frames))
            seen.add(id(candidate))
            wrapped += 1
        for child in getattr(module, "children", lambda: ())():
            stack.append(child)
    if wrapped <= 0:
        raise RuntimeError("no video_args_preprocessor found to install causal attention")
    return wrapped


def enable_causal_hand_motion(hand: Any) -> None:
    encoder = getattr(hand, "motion_encoder", None)
    if encoder is None or not hasattr(encoder, "causal"):
        raise RuntimeError("hand sidecar has no motion_encoder.causal flag")
    encoder.causal = True
