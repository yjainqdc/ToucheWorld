#!/usr/bin/env bash
# Block-AR teacher forcing: history clean, denoise one chunk, causal hand conv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export PYTHONUNBUFFERED=1

JSONL=${JSONL:-${DATA_ROOT}/train/train.jsonl}
CACHE=${CACHE:-${CACHE_DIR}}
HAND_RUN=${HAND_RUN:-${MODEL_DIR}/ToucheWorld/teacher27}
HAND_STEP=${HAND_STEP:-27307}
OUT=${OUTPUT_DIR:-${ROOT}/runs/causal_$(date -u +%Y%m%dT%H%M%SZ)}
TRAIN_GPUS=${TRAIN_GPUS:-0}
mkdir -p "${OUT}"

{
  echo "jsonl=${JSONL}"
  echo "cache=${CACHE}"
  echo "hand_run=${HAND_RUN} hand_step=${HAND_STEP}"
  echo "out=${OUT}"
  echo "lr=${LR:-1e-4} steps=${MAX_STEPS:-2000} distill_rank=${DISTILL_RANK:-64} block_frames=${BLOCK_FRAMES:-4}"
  echo "train_gpus=${TRAIN_GPUS} sp=${SP_SIZE:-1}"
} | tee "${OUT}/launch.txt"

NPROC=$(tr ',' '\n' <<<"${TRAIN_GPUS}" | wc -l)
echo "== causal distill nGPU=${NPROC} =="
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT:-29751}" \
  -m pipelines.ltx25_solar.train_causal \
  --cache-dir "${CACHE}" \
  --jsonl "${JSONL}" \
  --still-empty-limit "${STILL_EMPTY_LIMIT:-20}" \
  --still-hands-limit "${STILL_HANDS_LIMIT:-20}" \
  --output-dir "${OUT}" \
  --hand-run-dir "${HAND_RUN}" \
  --hand-step "${HAND_STEP}" \
  --lr "${LR:-1e-4}" \
  --max-steps "${MAX_STEPS:-2000}" \
  --save-every "${SAVE_EVERY:-250}" \
  --hand-rank "${HAND_RANK:-32}" \
  --distill-rank "${DISTILL_RANK:-64}" \
  --block-frames "${BLOCK_FRAMES:-4}" \
  --sp-size "${SP_SIZE:-1}" \
  --roles "${ROLES:-follow,grow,still_empty,still_hands}" \
  --hand-region-weight "${HAND_REGION_WEIGHT:-0}" \
  --start-step "${START_STEP:-0}" \
  --activation-checkpoint \
  2>&1 | tee "${OUT}/train.log"

echo "done -> ${OUT}"
