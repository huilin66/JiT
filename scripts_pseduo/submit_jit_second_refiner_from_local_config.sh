#!/usr/bin/env bash
set -euo pipefail

# Generate official-test JiT submission from a selected local sweep config,
# then apply the checkpoint detail_refiner a second time:
#   y1 = model(x)              # JiT + trained refiner
#   y2 = refiner(y1)
#   y  = w1 * y1 + w2 * y2
# No Clear folder and no local evaluation are used here.
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
# DATA_ROOT=${DATA_ROOT:-E:/cp_dir/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/test-input}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions_test}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/submission_history.csv}
SWEEP_DIR=${SWEEP_DIR:-submissions/local_val_sweep}

# CONFIG_CSV=${CONFIG_CSV:-${SWEEP_DIR}/local_val_sweep.csv}
CONFIG_CSV=${CONFIG_CSV:-}
CONFIG_ROW=${CONFIG_ROW:-}
CONFIG_MODEL_NAME=${CONFIG_MODEL_NAME:-}

# JIT_CKPT=${JIT_CKPT:-/data/huilin/projects/JiT/run/train/focus_2scene_msdt_refiner_h_1xA100_48g/h16_refiner_c1/16}
JIT_CKPT=${JIT_CKPT:-run/train_pseudo/p4_b16_focus_2scene_from_best_refiner_refiner_300ep_3x3090}
JIT_CKPT_TYPE=${JIT_CKPT_TYPE:-last}
STATE_KEY=${STATE_KEY:-model_ema1}
STEPS=${STEPS:-1}
STRIDE=${STRIDE:-16}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-8}


SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SCENE_JSON=${SCENE_JSON:-}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-8}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}
UPDATE_SCENE_FROM_FOCUS_PSEUDO=${UPDATE_SCENE_FROM_FOCUS_PSEUDO:-1}
FOCUS2SCENE_PSEUDO_JSON=${FOCUS2SCENE_PSEUDO_JSON:-${DATA_ROOT}/RainDrop_Train/Drop_focus_2scene_test_pseudo.json}
FOCUS2SCENE_PSEUDO_PREFIX=${FOCUS2SCENE_PSEUDO_PREFIX:-test_pseudo_}
REQUIRE_FOCUS2SCENE_PSEUDO_MATCH=${REQUIRE_FOCUS2SCENE_PSEUDO_MATCH:-1}

MODEL_NAME=${MODEL_NAME:-}
MODEL_NAME_PREFIX=${MODEL_NAME_PREFIX:-jit2r}
EXPLICIT_MODEL_NAME=0
if [[ -n "${MODEL_NAME}" ]]; then
  EXPLICIT_MODEL_NAME=1
fi

DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
TTA_HFLIP=${TTA_HFLIP:-1}
TTA_VFLIP=${TTA_VFLIP:-0}
TTA_ROT90=${TTA_ROT90:-0}
TTA_ROT180=${TTA_ROT180:-0}
TTA_ROT270=${TTA_ROT270:-0}
SCALES=${SCALES:-1.0}
SECOND_REFINER=${SECOND_REFINER:-1}
SECOND_REFINER_W1=${SECOND_REFINER_W1:-0.0}
SECOND_REFINER_W2=${SECOND_REFINER_W2:-1.0}
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

tta_args=()
if [[ "${TTA_HFLIP}" == "1" ]]; then
  tta_args+=(--tta-hflip)
fi
if [[ "${TTA_VFLIP}" == "1" ]]; then
  tta_args+=(--tta-vflip)
fi
if [[ "${TTA_ROT90}" == "1" ]]; then
  tta_args+=(--tta-rot90)
fi
if [[ "${TTA_ROT180}" == "1" ]]; then
  tta_args+=(--tta-rot180)
fi
if [[ "${TTA_ROT270}" == "1" ]]; then
  tta_args+=(--tta-rot270)
fi
second_refiner_args=()
if [[ "${SECOND_REFINER}" == "1" ]]; then
  second_refiner_args+=(
    --second-refiner
    --second-refiner-w1 "${SECOND_REFINER_W1}"
    --second-refiner-w2 "${SECOND_REFINER_W2}"
  )
fi

scene_args=()
if [[ "${UPDATE_SCENE_FROM_FOCUS_PSEUDO}" == "1" ]]; then
  if [[ ! -f "${FOCUS2SCENE_PSEUDO_JSON}" ]]; then
    echo "Missing FOCUS2SCENE_PSEUDO_JSON: ${FOCUS2SCENE_PSEUDO_JSON}" >&2
    exit 2
  fi
  scene_stamp=$(date +%Y%m%d_%H%M%S)
  scene_dir="${OUTPUT_ROOT}/scene_predictions"
  mkdir -p "${scene_dir}"

  raw_scene_json="${SCENE_JSON}"
  if [[ -z "${raw_scene_json}" ]]; then
    raw_scene_json="${scene_dir}/${MODEL_NAME}_${scene_stamp}_raw.json"
    raw_scene_csv="${scene_dir}/${MODEL_NAME}_${scene_stamp}_raw.csv"
    echo "============================================================"
    echo "[Scene pre-infer] ${raw_scene_json}"
    echo "checkpoint=${SCENE_CKPT}"
    echo "batch=${SCENE_BATCH_SIZE}, workers=${SCENE_NUM_WORKERS}"
    echo "============================================================"
    python -m scene_tools.infer_scene_convnext \
      --input-dir "${INPUT_DIR}" \
      --checkpoint "${SCENE_CKPT}" \
      --output-json "${raw_scene_json}" \
      --output-csv "${raw_scene_csv}" \
      --batch-size "${SCENE_BATCH_SIZE}" \
      --num-workers "${SCENE_NUM_WORKERS}" \
      --device "${DEVICE}" \
      --amp-dtype "${AMP_DTYPE}"
  fi

  updated_scene_json="${scene_dir}/${MODEL_NAME}_${scene_stamp}_focus_pseudo.json"
  updated_scene_manifest="${scene_dir}/${MODEL_NAME}_${scene_stamp}_focus_pseudo_manifest.json"
  update_scene_args=()
  if [[ "${REQUIRE_FOCUS2SCENE_PSEUDO_MATCH}" == "1" ]]; then
    update_scene_args+=(--require-any)
  fi
  python tools/update_scene_json_from_focus_pseudo.py \
    --scene-json "${raw_scene_json}" \
    --focus2scene-pseudo-json "${FOCUS2SCENE_PSEUDO_JSON}" \
    --input-dir "${INPUT_DIR}" \
    --output-json "${updated_scene_json}" \
    --manifest-json "${updated_scene_manifest}" \
    --prefix "${FOCUS2SCENE_PSEUDO_PREFIX}" \
    "${update_scene_args[@]}"

  SCENE_JSON="${updated_scene_json}"
  scene_args+=(--scene-json "${SCENE_JSON}")
elif [[ -n "${SCENE_JSON}" ]]; then
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
echo "tta_rot90=${TTA_ROT90}"
echo "tta_rot180=${TTA_ROT180}"
echo "tta_rot270=${TTA_ROT270}"
echo "scales=${SCALES}"
echo "second_refiner=${SECOND_REFINER}"
echo "second_refiner_w1=${SECOND_REFINER_W1}"
echo "second_refiner_w2=${SECOND_REFINER_W2}"
echo "update_scene_from_focus_pseudo=${UPDATE_SCENE_FROM_FOCUS_PSEUDO}"
echo "focus2scene_pseudo_json=${FOCUS2SCENE_PSEUDO_JSON}"
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
  "${tta_args[@]}" \
  --scales "${SCALES}" \
  "${second_refiner_args[@]}" \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --notes "${NOTES}; ckpt_type=${JIT_CKPT_TYPE}; state_key=${STATE_KEY}; steps=${STEPS}; stride=${STRIDE}; tta_hflip=${TTA_HFLIP}; tta_vflip=${TTA_VFLIP}; tta_rot90=${TTA_ROT90}; tta_rot180=${TTA_ROT180}; tta_rot270=${TTA_ROT270}; scales=${SCALES}; second_refiner=${SECOND_REFINER}; second_refiner_w1=${SECOND_REFINER_W1}; second_refiner_w2=${SECOND_REFINER_W2}; update_scene_from_focus_pseudo=${UPDATE_SCENE_FROM_FOCUS_PSEUDO}" \
  "${remove_args[@]}"

echo "Submission prediction finished: ${OUTPUT_ROOT}"
