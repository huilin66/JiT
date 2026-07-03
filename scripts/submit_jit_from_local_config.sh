#!/usr/bin/env bash
set -euo pipefail

# Generate official-test JiT submission from a selected local sweep config.
# No Clear folder and no local evaluation are used here.

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/Drop}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/submission_history.csv}
SWEEP_DIR=${SWEEP_DIR:-submissions/local_val_sweep}

# CONFIG_CSV=${CONFIG_CSV:-${SWEEP_DIR}/local_val_sweep.csv}
CONFIG_CSV=${CONFIG_CSV:-}
CONFIG_ROW=${CONFIG_ROW:-}
CONFIG_MODEL_NAME=${CONFIG_MODEL_NAME:-}


JIT_CKPT=${JIT_CKPT:-run/train/b16_focus_2scene_msdt_refiner_plan_b_conservative_1x5090/16}
JIT_CKPT_TYPE=${JIT_CKPT_TYPE:-last}
STATE_KEY=${STATE_KEY:-model_ema2}
STEPS=${STEPS:-1}
STRIDE=${STRIDE:-64}

SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SCENE_JSON=${SCENE_JSON:-}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-8}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}

MODEL_NAME=${MODEL_NAME:-}
MODEL_NAME_PREFIX=${MODEL_NAME_PREFIX:-jit_submit}
EXPLICIT_MODEL_NAME=0
if [[ -n "${MODEL_NAME}" ]]; then
  EXPLICIT_MODEL_NAME=1
fi
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-8}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
TTA_HFLIP=${TTA_HFLIP:-1}
TTA_VFLIP=${TTA_VFLIP:-0}
SCALES=${SCALES:-1.0}
NOTES=${NOTES:-submit_from_local_config}
REMOVE_IMAGES_AFTER_ZIP=${REMOVE_IMAGES_AFTER_ZIP:-0}

if [[ -n "${CONFIG_CSV}" ]]; then
  submit_scene_json="${SCENE_JSON}"
  row_args=(--csv "${CONFIG_CSV}")
  if [[ -n "${CONFIG_MODEL_NAME}" ]]; then
    row_args+=(--model-name "${CONFIG_MODEL_NAME}")
  else
    CONFIG_ROW="${CONFIG_ROW:-1}"
    row_args+=(--row "${CONFIG_ROW}")
  fi
  eval "$(python tools/local_sweep_row_to_env.py "${row_args[@]}")"
  JIT_CKPT_TYPE="${JIT_CKPT_TYPES}"
  STATE_KEY="${STATE_KEYS}"
  STEPS="${STEPS_LIST}"
  STRIDE="${STRIDES}"
  if [[ -n "${submit_scene_json}" ]]; then
    SCENE_JSON="${submit_scene_json}"
  else
    SCENE_JSON=""
  fi
fi

if [[ -z "${MODEL_NAME}" ]]; then
  if [[ -n "${CONFIG_ROW}" || -n "${CONFIG_MODEL_NAME}" ]]; then
    MODEL_NAME="submit_${JIT_CKPT_TYPE}_${STATE_KEY}_s${STEPS}_r${STRIDE}"
  else
    MODEL_NAME="${MODEL_NAME_PREFIX}_${JIT_CKPT_TYPE}_${STATE_KEY}_s${STEPS}_r${STRIDE}"
  fi
fi

tta_args=()
tta_suffix=""
if [[ "${TTA_HFLIP}" == "1" ]]; then
  tta_args+=(--tta-hflip)
  tta_suffix="${tta_suffix}_hflip"
fi
if [[ "${TTA_VFLIP}" == "1" ]]; then
  tta_args+=(--tta-vflip)
  tta_suffix="${tta_suffix}_vflip"
fi
if [[ -n "${tta_suffix}" && "${EXPLICIT_MODEL_NAME}" == "0" ]]; then
  MODEL_NAME="${MODEL_NAME}${tta_suffix}"
fi
scale_suffix=""
if [[ "${SCALES}" != "1.0" && "${SCALES}" != "1" ]]; then
  scale_suffix="_ms${SCALES//,/__}"
  scale_suffix="${scale_suffix//./p}"
fi
if [[ -n "${scale_suffix}" && "${EXPLICIT_MODEL_NAME}" == "0" ]]; then
  MODEL_NAME="${MODEL_NAME}${scale_suffix}"
fi

scene_args=()
if [[ -n "${SCENE_JSON}" ]]; then
  scene_args+=(--scene-json "${SCENE_JSON}")
else
  scene_stamp=$(date +%Y%m%d_%H%M%S)
  scene_dir="${OUTPUT_ROOT}/scene_predictions"
  mkdir -p "${scene_dir}"
  scene_args+=(
    --scene-checkpoint "${SCENE_CKPT}"
    --scene-output-json "${scene_dir}/${MODEL_NAME}_${scene_stamp}.json"
    --scene-output-csv "${scene_dir}/${MODEL_NAME}_${scene_stamp}.csv"
    --scene-batch-size "${SCENE_BATCH_SIZE}"
    --scene-num-workers "${SCENE_NUM_WORKERS}"
  )
fi

remove_args=()
if [[ "${REMOVE_IMAGES_AFTER_ZIP}" == "1" ]]; then
  remove_args+=(--remove-images-after-zip)
fi

echo "============================================================"
echo "[Submit JiT] ${MODEL_NAME}"
echo "input_dir=${INPUT_DIR}"
echo "checkpoint=${JIT_CKPT}"
echo "ckpt_type=${JIT_CKPT_TYPE}"
echo "state_key=${STATE_KEY}"
echo "steps=${STEPS}"
echo "stride=${STRIDE}"
echo "tta_hflip=${TTA_HFLIP}"
echo "tta_vflip=${TTA_VFLIP}"
echo "scales=${SCALES}"
echo "scene_json=${SCENE_JSON:-<predict with SCENE_CKPT>}"
echo "output_root=${OUTPUT_ROOT}"
echo "============================================================"

python submit_jit.py \
  --input-dir "${INPUT_DIR}" \
  --checkpoint "${JIT_CKPT}" \
  --ckpt_type "${JIT_CKPT_TYPE}" \
  --output-root "${OUTPUT_ROOT}" \
  --history-csv "${HISTORY_CSV}" \
  --model-name "${MODEL_NAME}" \
  --state-key "${STATE_KEY}" \
  --use-scene \
  "${scene_args[@]}" \
  --steps "${STEPS}" \
  --stride "${STRIDE}" \
  --tile-batch-size "${TILE_BATCH_SIZE}" \
  "${tta_args[@]}" \
  --scales "${SCALES}" \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --notes "${NOTES}; ckpt_type=${JIT_CKPT_TYPE}; state_key=${STATE_KEY}; steps=${STEPS}; stride=${STRIDE}; tta_hflip=${TTA_HFLIP}; tta_vflip=${TTA_VFLIP}; scales=${SCALES}" \
  "${remove_args[@]}"

echo "Submission prediction finished: ${OUTPUT_ROOT}"
