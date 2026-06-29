#!/usr/bin/env bash
set -euo pipefail

# Predict scene labels for Drop images. The JSON can be passed to submit_jit.py
# with --use-scene --scene-json.

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/Drop}
SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext/checkpoint-best.pth}
OUTPUT_JSON=${OUTPUT_JSON:-run/scene_convnext/scene_pred.json}
OUTPUT_CSV=${OUTPUT_CSV:-}

BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
RECURSIVE=${RECURSIVE:-1}

if [[ ! -f "${SCENE_CKPT}" ]]; then
  echo "Missing scene checkpoint: ${SCENE_CKPT}" >&2
  exit 2
fi

recursive_args=()
if [[ "${RECURSIVE}" != "1" ]]; then
  recursive_args+=(--no-recursive)
fi

csv_args=()
if [[ -n "${OUTPUT_CSV}" ]]; then
  csv_args+=(--output-csv "${OUTPUT_CSV}")
fi

python scene_tools/infer_scene_convnext.py \
  --input-dir "${INPUT_DIR}" \
  --checkpoint "${SCENE_CKPT}" \
  --output-json "${OUTPUT_JSON}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  "${csv_args[@]}" \
  "${recursive_args[@]}"
