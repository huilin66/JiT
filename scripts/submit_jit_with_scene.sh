#!/usr/bin/env bash
set -euo pipefail

# One-step submission: predict scene labels with ConvNeXt, then run JiT inference.

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/Drop}
JIT_CKPT=${JIT_CKPT:-run/ablation_b16_3x3090/b16_blur_2scene_no_head}
JIT_CKPT_TYPE=${JIT_CKPT_TYPE:-best}
SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_blur_2scene/checkpoint-best.pth}
SCENE_JSON=${SCENE_JSON:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions}

MODEL_NAME=${MODEL_NAME:-}
STATE_KEY=${STATE_KEY:-auto}
STEPS=${STEPS:-1}
STRIDE=${STRIDE:-128}
JIT_CKPT_TYPES=${JIT_CKPT_TYPES:-${JIT_CKPT_TYPE}}
STATE_KEYS=${STATE_KEYS:-${STATE_KEY}}
STEPS_LIST=${STEPS_LIST:-${STEPS}}
STRIDES=${STRIDES:-${STRIDE}}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-8}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-8}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}
NOTES=${NOTES:-auto_scene_convnext}

split_list() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  SPLIT_RESULT=(${raw})
}

model_name_args=()
if [[ -n "${MODEL_NAME}" ]]; then
  model_name_args+=(--model-name "${MODEL_NAME}")
fi

if [[ -z "${SCENE_JSON}" ]]; then
  if [[ ! -f "${SCENE_CKPT}" ]]; then
    echo "Missing scene checkpoint: ${SCENE_CKPT}" >&2
    exit 2
  fi
  scene_stamp=$(date +%Y%m%d_%H%M%S)
  scene_dir="${OUTPUT_ROOT}/scene_predictions"
  mkdir -p "${scene_dir}"
  SCENE_JSON="${scene_dir}/scene_${scene_stamp}.json"
  SCENE_CSV="${scene_dir}/scene_${scene_stamp}.csv"

  echo "============================================================"
  echo "[Scene] Predict labels once for all JiT submission runs"
  echo "Input: ${INPUT_DIR}"
  echo "Checkpoint: ${SCENE_CKPT}"
  echo "JSON: ${SCENE_JSON}"
  echo "============================================================"

  python scene_tools/infer_scene_convnext.py \
    --input-dir "${INPUT_DIR}" \
    --checkpoint "${SCENE_CKPT}" \
    --output-json "${SCENE_JSON}" \
    --output-csv "${SCENE_CSV}" \
    --batch-size "${SCENE_BATCH_SIZE}" \
    --num-workers "${SCENE_NUM_WORKERS}" \
    --device "${DEVICE}" \
    --amp-dtype "${AMP_DTYPE}"
else
  if [[ ! -f "${SCENE_JSON}" ]]; then
    echo "Missing scene json: ${SCENE_JSON}" >&2
    exit 2
  fi
fi

split_list "${JIT_CKPT_TYPES}"
ckpt_type_list=("${SPLIT_RESULT[@]}")
split_list "${STATE_KEYS}"
state_key_list=("${SPLIT_RESULT[@]}")
split_list "${STEPS_LIST}"
steps_list=("${SPLIT_RESULT[@]}")
split_list "${STRIDES}"
stride_list=("${SPLIT_RESULT[@]}")

for ckpt_type in "${ckpt_type_list[@]}"; do
  for state_key in "${state_key_list[@]}"; do
    for steps in "${steps_list[@]}"; do
      for stride in "${stride_list[@]}"; do
        run_notes="${NOTES}; ckpt_type=${ckpt_type}; state_key=${state_key}; steps=${steps}; stride=${stride}"
        echo "============================================================"
        echo "[JiT] ckpt_type=${ckpt_type}, state_key=${state_key}, steps=${steps}, stride=${stride}"
        echo "Checkpoint: ${JIT_CKPT}"
        echo "Scene JSON: ${SCENE_JSON}"
        echo "============================================================"

        python submit_jit.py \
          --input-dir "${INPUT_DIR}" \
          --checkpoint "${JIT_CKPT}" \
          --ckpt_type "${ckpt_type}" \
          --output-root "${OUTPUT_ROOT}" \
          --state-key "${state_key}" \
          --use-scene \
          --scene-json "${SCENE_JSON}" \
          --steps "${steps}" \
          --stride "${stride}" \
          --tile-batch-size "${TILE_BATCH_SIZE}" \
          --device "${DEVICE}" \
          --amp-dtype "${AMP_DTYPE}" \
          --notes "${run_notes}" \
          "${model_name_args[@]}"
      done
    done
  done
done
