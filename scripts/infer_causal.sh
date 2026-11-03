#!/usr/bin/env bash
# Causal Block-AR. Hand LoRA + distill LoRA from the same causal27 folder.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

"${PY}" -u -m eval.sample_bench_causal \
  --jsonl "${JSONL:?set JSONL}" \
  --cache-dir "${CACHE_DIR:?set CACHE_DIR}" \
  --out-dir "${OUT_DIR:-${ROOT}/outputs/causal}" \
  --hand-run "${HAND_RUN:-${MODEL_DIR}/ToucheWorld/causal27}" \
  --hand-step "${HAND_STEP:-27307}" \
  --distill-run "${DISTILL_RUN:-${HAND_RUN:-${MODEL_DIR}/ToucheWorld/causal27}}" \
  --distill-step "${DISTILL_STEP:-10240}" \
  --prefix-latents "${PREFIX_LATENTS:-1}" \
  --block-frames "${BLOCK_FRAMES:-4}" \
  --window-frames "${WINDOW_FRAMES:-8}" \
  --denoise-steps "${DENOISE_STEPS:-30}" \
  --transformer "${TRANSFORMER}" \
  --vae "${VAE}" \
  --camera-adapter "${CAMERA}"
