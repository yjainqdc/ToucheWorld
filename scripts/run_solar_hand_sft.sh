#!/usr/bin/env bash
# Preencode H3-hand mix → Solar LTX-2.5 latents, then train a new hand LoRA
# while the Solar Stage0.5 camera LoRA stays frozen.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export PYTHONUNBUFFERED=1

JSONL=${JSONL:-${DATA_ROOT}/train/train.jsonl}
CACHE=${CACHE:-${CACHE_DIR}}
OUT=${OUTPUT_DIR:-${ROOT}/runs/teacher_$(date -u +%Y%m%dT%H%M%SZ)}
LIMIT=${LIMIT:-0}
LR=${LR:-1e-4}
MAX_STEPS=${MAX_STEPS:-2000}
PREENCODE_GPUS=${PREENCODE_GPUS:-0}
TRAIN_GPUS=${TRAIN_GPUS:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_PREENCODE=${SKIP_PREENCODE:-0}
STILL_EMPTY_LIMIT=${STILL_EMPTY_LIMIT:-20}
STILL_HANDS_LIMIT=${STILL_HANDS_LIMIT:-20}
mkdir -p "${CACHE}" "${OUT}"

{
  echo "jsonl=${JSONL}"
  echo "cache=${CACHE}"
  echo "out=${OUT}"
  echo "limit=${LIMIT} (0=all)"
  echo "lr=${LR} steps=${MAX_STEPS} hand_rank=${HAND_RANK:-32}"
  echo "preencode_gpus=${PREENCODE_GPUS}"
  echo "train_gpus=${TRAIN_GPUS} sp=${SP_SIZE:-1}"
  echo "still_empty_limit=${STILL_EMPTY_LIMIT} still_hands_limit=${STILL_HANDS_LIMIT}"
  echo "roles=${ROLES:-follow,grow,still_empty,still_hands}"
  echo "hand_region_weight=${HAND_REGION_WEIGHT:-0} start_step=${START_STEP:-0}"
  echo "resume_dir=${RESUME_DIR:-}"
} | tee "${OUT}/launch.txt"

if [[ "${SKIP_PREENCODE}" != "0" ]]; then
  echo "SKIP_PREENCODE=1; using cache=${CACHE}"
else
IFS=',' read -ra GPU_ARR <<< "${PREENCODE_GPUS}"
N_SHARD=${#GPU_ARR[@]}
echo "== preencode ${N_SHARD} shards =="
PIDS=()
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  echo "shard ${i}/${N_SHARD} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 \
    MASTER_PORT="$((29840 + i))" \
    "${PY}" -u -m pipelines.ltx25_solar.preencode_hand \
    --jsonl "${JSONL}" \
    --cache-dir "${CACHE}" \
    --limit "${LIMIT}" \
    --shard-id "${i}" \
    --num-shards "${N_SHARD}" \
    > "${OUT}/preencode.shard${i}.log" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "preencode shard ${i} failed (pid=${PIDS[$i]})"
    FAIL=1
  fi
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "preencode failed; not starting train"
  exit 1
fi
echo "preencode done cache=$(ls -1 "${CACHE}"/*.safetensors 2>/dev/null | wc -l)"
fi

if [[ "${SKIP_TRAIN}" != "0" ]]; then
  echo "SKIP_TRAIN=1; cache is ready at ${CACHE}"
  exit 0
fi

NPROC=$(tr ',' '\n' <<<"${TRAIN_GPUS}" | wc -l)
echo "== train nGPU=${NPROC} =="
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT:-29741}" \
  -m pipelines.ltx25_solar.train_hand \
  --cache-dir "${CACHE}" \
  --jsonl "${JSONL}" \
  --still-empty-limit "${STILL_EMPTY_LIMIT}" \
  --still-hands-limit "${STILL_HANDS_LIMIT}" \
  --output-dir "${OUT}" \
  --lr "${LR}" \
  --max-steps "${MAX_STEPS}" \
  --save-every "${SAVE_EVERY:-250}" \
  --hand-rank "${HAND_RANK:-32}" \
  --sp-size "${SP_SIZE:-1}" \
  --roles "${ROLES:-follow,grow,still_empty,still_hands}" \
  --hand-region-weight "${HAND_REGION_WEIGHT:-0}" \
  --start-step "${START_STEP:-0}" \
  --resume-dir "${RESUME_DIR:-}" \
  --activation-checkpoint \
  2>&1 | tee "${OUT}/train.log"

echo "done -> ${OUT}"
