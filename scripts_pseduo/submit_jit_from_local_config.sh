#!/usr/bin/env bash
set -euo pipefail

# Generate official-test JiT submission from a selected local sweep config.
# No Clear folder and no local evaluation are used here.
# DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_ROOT=${DATA_ROOT:-E:/cp_dir/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/test-input}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions_test}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/submission_history.csv}
SWEEP_DIR=${SWEEP_DIR:-submissions/local_val_sweep}

# CONFIG_CSV=${CONFIG_CSV:-${SWEEP_DIR}/local_val_sweep.csv}
CONFIG_CSV=${CONFIG_CSV:-}
CONFIG_ROW=${CONFIG_ROW:-}
CONFIG_MODEL_NAME=${CONFIG_MODEL_NAME:-}

# JIT_CKPT=${JIT_CKPT:-/data/huilin/projects/JiT/run/train/focus_2scene_msdt_refiner_h_1xA100_48g/h16_refiner_c1/16}
JIT_CKPT=${JIT_CKPT:-run/p2_b16_dn_blur_4scene_jit_best_base_hparams_3x3090}
JIT_CKPT_TYPE=${JIT_CKPT_TYPE:-last}
STATE_KEY=${STATE_KEY:-model_ema1}
STEPS=${STEPS:-1}
STRIDE=${STRIDE:-32}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-128}


SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SCENE_JSON=${SCENE_JSON:-}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-512}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}

MODEL_NAME=${MODEL_NAME:-}
MODEL_NAME_PREFIX=${MODEL_NAME_PREFIX:-jit}
EXPLICIT_MODEL_NAME=0
if [[ -n "${MODEL_NAME}" ]]; then
  EXPLICIT_MODEL_NAME=1
fi

DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
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
  case "${JIT_CKPT_TYPE}" in
    best) ckpt_tag="b" ;;
    last) ckpt_tag="l" ;;
    *) ckpt_tag="${JIT_CKPT_TYPE}" ;;
  esac
  case "${STATE_KEY}" in
    model_ema1) state_tag="e1" ;;
    model_ema2) state_tag="e2" ;;
    model) state_tag="m" ;;
    auto) state_tag="a" ;;
    *) state_tag="${STATE_KEY}" ;;
  esac
  MODEL_NAME="${MODEL_NAME_PREFIX}_${ckpt_tag}_${state_tag}_s${STEPS}_r${STRIDE}"
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
echo "aug_infer=0"
echo "scene_json=${SCENE_JSON:-<predict with SCENE_CKPT>}"
echo "scene_ckpt=${SCENE_CKPT}"
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
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --notes "${NOTES}; ckpt_type=${JIT_CKPT_TYPE}; state_key=${STATE_KEY}; steps=${STEPS}; stride=${STRIDE}; aug_infer=0" \
  "${remove_args[@]}"

echo "Submission prediction finished: ${OUTPUT_ROOT}"
