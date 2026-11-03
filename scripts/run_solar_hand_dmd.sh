#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
HAND_RUN=${HAND_RUN:-${MODEL_DIR}/ToucheWorld/teacher27}
CAUSAL_RUN=${CAUSAL_RUN:-${MODEL_DIR}/ToucheWorld/causal27}
CACHE_DIR=${CACHE_DIR:-${ROOT}/cache}
JSONL=${JSONL:-${DATA_ROOT}/train/train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new DMD run directory}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
MASTER_PORT=${MASTER_PORT:-29871}
MAX_STEPS=${MAX_STEPS:-2000}
START_STEP=${START_STEP:-0}
SAVE_EVERY=${SAVE_EVERY:-250}
SMOKE=${SMOKE:-0}

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

IFS=',' read -ra GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
ARGS=(
  --cache-dir "${CACHE_DIR}"
  --jsonl "${JSONL}"
  --hand-run-dir "${HAND_RUN}"
  --hand-step "${HAND_STEP:-27307}"
  --causal-run-dir "${CAUSAL_RUN}"
  --causal-step "${CAUSAL_STEP:-10240}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
  --start-step "${START_STEP}"
  --save-every "${SAVE_EVERY}"
  --student-steps 4
  --block-frames 4
  --window-frames 8
  --generator-lr 2e-5
  --critic-lr 2e-5
  --critic-warmup 20
  --gen-update-ratio 5
  --sft-weight 4
  --fkl-weight 1
  --real-fkl-weight 0.5
  --activation-checkpoint
)
if [[ "${SMOKE}" == 1 ]]; then
  ARGS+=(--critic-warmup 0 --gen-update-ratio 1 --smoke-contract)
fi

mkdir -p "${OUTPUT_DIR}"
printf '%q ' "$0" "${ARGS[@]}" >"${OUTPUT_DIR}/launch.txt"
printf '\n' >>"${OUTPUT_DIR}/launch.txt"
"${PY}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${#GPU_IDS[@]}" \
  --master_port="${MASTER_PORT}" \
  -m pipelines.ltx25_solar.train_dmd \
  "${ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
