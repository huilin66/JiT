#!/usr/bin/env bash
set -euo pipefail

# One-step submission: predict scene labels with ConvNeXt, then run JiT inference.

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/Drop}
JIT_CKPT=${JIT_CKPT:-run/train/ablation_b16_3x3090/b16_scene_no_head/16}
JIT_CKPT_TYPE=${JIT_CKPT_TYPE:-best}
SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext/checkpoint-best.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions}

MODEL_NAME=${MODEL_NAME:-}
STATE_KEY=${STATE_KEY:-auto}
STEPS=${STEPS:-1}
STRIDE=${STRIDE:-128}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-128}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}
NOTES=${NOTES:-auto_scene_convnext}

if [[ ! -f "${SCENE_CKPT}" ]]; then
  echo "Missing scene checkpoint: ${SCENE_CKPT}" >&2
  exit 2
fi

model_name_args=()
if [[ -n "${MODEL_NAME}" ]]; then
  model_name_args+=(--model-name "${MODEL_NAME}")
fi

python submit_jit.py \
  --input-dir "${INPUT_DIR}" \
  --checkpoint "${JIT_CKPT}" \
  --ckpt_type "${JIT_CKPT_TYPE}" \
  --output-root "${OUTPUT_ROOT}" \
  --state-key "${STATE_KEY}" \
  --scene-checkpoint "${SCENE_CKPT}" \
  --scene-batch-size "${SCENE_BATCH_SIZE}" \
  --scene-num-workers "${SCENE_NUM_WORKERS}" \
  --steps "${STEPS}" \
  --stride "${STRIDE}" \
  --tile-batch-size "${TILE_BATCH_SIZE}" \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --notes "${NOTES}" \
  "${model_name_args[@]}"
