#!/usr/bin/env bash
# Teacher first-frame image-to-video inference.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

"${PY}" -u -m eval.sample_solar_hand \
  --run-dir "${RUN_DIR:-${MODEL_DIR}/ToucheWorld/teacher27}" \
  --step "${STEP:-27307}" \
  --cache-dir "${CACHE_DIR}" \
  --out-dir "${OUT_DIR:-${ROOT}/outputs/teacher}" \
  --transformer "${TRANSFORMER}" \
  --vae "${VAE}" \
  --camera-adapter "${CAMERA}" \
  --cfg "${CFG:-3.0}" \
  --stg "${STG:-1.0}" \
  --case "${CASE:?set CASE to name:cache_filename:zero_hand}" \
  "$@"
