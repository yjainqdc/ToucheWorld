#!/usr/bin/env bash
# Shared environment for the training and inference launchers.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${TOUCHEWORLD_CONFIG:-${ROOT}/config.env}
if [[ -f "${CONFIG}" ]]; then
  # Keep explicitly supplied settings ahead of values from config.env.
  _saved_names=()
  _saved_values=()
  _saved_count=0
  for _name in LTX25_ROOT SOLAR_SRC LTX25_PYTHON DATA_ROOT JSONL CACHE_DIR MODEL_DIR \
      TRANSFORMER VAE GEMMA CAMERA RUN_DIR HAND_RUN DISTILL_RUN TRAIN_GPUS PREENCODE_GPUS; do
    if declare -p "${_name}" &>/dev/null; then
      _saved_names[_saved_count]=${_name}
      _saved_values[_saved_count]=${!_name}
      _saved_count=$((_saved_count + 1))
    fi
  done
  set -a
  source "${CONFIG}"
  set +a
  for ((_i=0; _i<_saved_count; _i++)); do
    printf -v "${_saved_names[_i]}" '%s' "${_saved_values[_i]}"
    export "${_saved_names[_i]}"
  done
  unset _saved_names _saved_values _saved_count _name _i
fi
cd "${ROOT}"
LTX25=${LTX25_ROOT:-${ROOT}/third_party/LTX-2.5}
SOLAR_SRC=${SOLAR_SRC:-${ROOT}/third_party/SolarWM/src}
PY=${LTX25_PYTHON:-python}
if [[ ! -d "${LTX25}/packages/ltx-core/src" ]]; then
  echo "Set LTX25_ROOT to an LTX-2.5 source checkout (see config.env.example)." >&2
  exit 1
fi
if [[ ! -d "${SOLAR_SRC}/solarwm" ]]; then
  echo "Set SOLAR_SRC to the directory containing the compatible solarwm package." >&2
  exit 1
fi
export PYTHONPATH="${ROOT}:${LTX25}/packages/ltx-trainer/src:${LTX25}/packages/ltx-core/src:${LTX25}/packages/ltx-pipelines/src:${SOLAR_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export MODEL_DIR=${MODEL_DIR:-${ROOT}/models}
export DATA_ROOT=${DATA_ROOT:-${ROOT}/data}
export JSONL=${JSONL:-${DATA_ROOT}/train/train.jsonl}
export CACHE_DIR=${CACHE_DIR:-${ROOT}/cache}
export TRANSFORMER=${TRANSFORMER:-${MODEL_DIR}/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors}
export VAE=${VAE:-${MODEL_DIR}/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors}
export GEMMA=${GEMMA:-${MODEL_DIR}/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors}
export CAMERA=${CAMERA:-${MODEL_DIR}/solarwm/camera_adapter.safetensors}
