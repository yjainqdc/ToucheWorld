"""Layer-wise K/V cache for block-AR video self-attention.

History tokens sit at timestep 0 and are causal, so their post-PRoPE K/V (and
attention output) are constant across Euler steps of the current chunk. The
first step fills the cache; later steps only project the current chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from pipelines.ltx25_solar.causal_attn import frame_token_index, window_history_frames
from solarwm.backends.ltx25.geometry import STABLE_GEOMETRY
from solarwm.backends.ltx25.torch_model import _camera_pair, _token_camera_rows
from solarwm.backends.ltx25.torch_prope import prope_qkv


def _gather(tensor: torch.Tensor | tuple | None, index: torch.Tensor, seq_len: int):
    if tensor is None:
        return None
    if isinstance(tensor, tuple):
        return tuple(_gather(item, index, seq_len) for item in tensor)
    dim = None
    for axis, size in enumerate(tensor.shape):
        if size == seq_len:
            dim = axis
            break
    if dim is None:
        return tensor
    return tensor.index_select(dim, index.to(device=tensor.device))


@dataclass
class _LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    hist_out: torch.Tensor


class AttnKVCache:
    def __init__(self) -> None:
        self.mode = "off"
        self.branch = "cond"
        self.hist_index = torch.zeros(0, dtype=torch.int64)
        self.curr_index = torch.zeros(0, dtype=torch.int64)
        self._layers: dict[tuple[str, int], _LayerCache] = {}

    def begin_block(self, start: int, end: int, window_frames: int) -> None:
        hist = window_history_frames(start, window_frames, STABLE_GEOMETRY.latent_frames)
        curr = tuple(range(start, end))
        self.hist_index = frame_token_index(hist)
        self.curr_index = frame_token_index(curr)
        self._layers.clear()
        self.mode = "fill"

    def hit_mask(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        h, c = int(self.hist_index.numel()), int(self.curr_index.numel())
        if h <= 0 or c <= 0:
            return None
        keep = torch.ones(c, h + c, device=device, dtype=torch.bool)
        bias = torch.zeros(c, h + c, device=device, dtype=dtype)
        bias.masked_fill_(~keep, torch.finfo(dtype).min)
        return bias.unsqueeze(0).unsqueeze(0)

    def dispatch(self, module: torch.nn.Module, original: Callable, *args: Any, **kwargs: Any):
        if self.mode == "off" or bool(kwargs.get("all_perturbed", False)):
            return original(*args, **kwargs)
        if self.mode == "fill":
            output = original(*args, **kwargs)
            self._capture(module, output, *args, **kwargs)
            return output
        if self.mode == "hit" and self.hist_index.numel() > 0 and self.curr_index.numel() > 0:
            return self._replay(module, original, *args, **kwargs)
        return original(*args, **kwargs)

    def _capture(self, module: torch.nn.Module, output: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        x = args[0] if args else kwargs["x"]
        pe = kwargs.get("pe", args[3] if len(args) > 3 else None)
        viewmats = kwargs.get("camera_viewmats")
        camera_k = kwargs.get("camera_k")
        hist = self.hist_index.to(device=x.device)
        _query, key_heads, value_heads, _transform = _project_prope(module, x, pe, viewmats, camera_k)
        self._layers[(self.branch, id(module))] = _LayerCache(
            key=key_heads.index_select(1, hist).detach(),
            value=value_heads.index_select(1, hist).detach(),
            hist_out=output.index_select(1, hist).detach(),
        )

    def _replay(self, module: torch.nn.Module, original: Callable, *args: Any, **kwargs: Any):
        cached = self._layers.get((self.branch, id(module)))
        if cached is None:
            return original(*args, **kwargs)
        x = args[0]
        mask = kwargs.get("mask", args[2] if len(args) > 2 else None)
        pe = kwargs.get("pe", args[3] if len(args) > 3 else None)
        perturbation_mask = kwargs.get("perturbation_mask", args[5] if len(args) > 5 else None)
        viewmats = kwargs.get("camera_viewmats")
        camera_k = kwargs.get("camera_k")
        seq = x.shape[1]
        curr = self.curr_index.to(device=x.device)
        hist = self.hist_index.to(device=x.device)
        x_c = x.index_select(1, curr)
        pe_c = _gather(pe, curr, seq)
        vm_c = _gather(viewmats, curr, seq)
        k_c = _gather(camera_k, curr, seq)
        query_heads, key_heads, value_heads, output_transform = _project_prope(
            module, x_c, pe_c, vm_c, k_c
        )
        n_hist = cached.key.shape[1]
        key_heads = torch.cat([cached.key.to(device=x.device, dtype=key_heads.dtype), key_heads], dim=1)
        value_heads = torch.cat(
            [cached.value.to(device=x.device, dtype=value_heads.dtype), value_heads], dim=1
        )
        hit_mask = self.hit_mask(dtype=query_heads.dtype, device=query_heads.device)
        current = module.masked_attention_function(
            query_heads.flatten(-2),
            key_heads.flatten(-2),
            value_heads.flatten(-2),
            module.heads,
            hit_mask,
        )
        if output_transform is not None:
            current_heads = current.view(x_c.shape[0], x_c.shape[1], module.heads, module.dim_head).transpose(1, 2)
            current = output_transform(current_heads).transpose(1, 2).contiguous().flatten(-2)
        if perturbation_mask is not None:
            pert = perturbation_mask.index_select(1, curr)
            bypass = value_heads[:, n_hist:].flatten(-2)
            current = current * pert + bypass * (1 - pert)
        if module.to_gate_logits is not None:
            current = module.gated_attention_function(x_c, current, module)
        current = module.to_out(current)
        merged = x.new_zeros(x.shape[0], seq, current.shape[-1], dtype=current.dtype)
        merged.index_copy_(1, hist, cached.hist_out.to(dtype=current.dtype, device=x.device))
        merged.index_copy_(1, curr, current)
        return merged


def _project_prope(module, x, pe, viewmats, camera_k):
    query = module.to_q(x)
    key = module.to_k(x)
    value = module.to_v(x)
    query, key = module.preattention_function(query, key, module, None, pe, None)
    batch, seq, hidden = query.shape
    query_heads = query.view(batch, seq, module.heads, module.dim_head)
    key_heads = key.view(batch, seq, module.heads, module.dim_head)
    value_heads = value.view(batch, seq, module.heads, module.dim_head)
    output_transform = None
    if _camera_pair(viewmats, camera_k):
        token_viewmats = _token_camera_rows(
            viewmats, batch=batch, sequence=seq, matrix_size=4, name="camera_viewmats"
        )
        token_intrinsics = _token_camera_rows(
            camera_k, batch=batch, sequence=seq, matrix_size=3, name="camera_K"
        )
        transformed = prope_qkv(
            query_heads.transpose(1, 2),
            key_heads.transpose(1, 2),
            value_heads.transpose(1, 2),
            viewmats=token_viewmats.to(device=query.device, dtype=query.dtype),
            intrinsics=token_intrinsics.to(device=query.device, dtype=query.dtype),
            camera_translation_transform=getattr(module, "camera_translation_transform", "linear"),
        )
        query_heads = transformed[0].transpose(1, 2).contiguous()
        key_heads = transformed[1].transpose(1, 2).contiguous()
        value_heads = transformed[2].transpose(1, 2).contiguous()
        output_transform = transformed[3]
    return query_heads, key_heads, value_heads, output_transform


def install_attn_kv_cache(root: torch.nn.Module, store: AttnKVCache) -> int:
    wrapped = 0
    for module in root.modules():
        if module.__class__.__name__ != "SolarLTX25VideoSelfAttention":
            continue
        if getattr(module, "_block_kv_wrapped", False):
            continue
        original = module.forward

        def bound(*args, _orig=original, _mod=module, **kwargs):
            return store.dispatch(_mod, _orig, *args, **kwargs)

        module.forward = bound
        module._block_kv_wrapped = True
        wrapped += 1
    if wrapped <= 0:
        raise RuntimeError("no SolarLTX25VideoSelfAttention modules to wrap for KV cache")
    return wrapped
