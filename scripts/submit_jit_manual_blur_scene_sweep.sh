#!/usr/bin/env bash
set -euo pipefail

# Submit/sweep JiT checkpoints trained with manual blur scene labels.
#
# This script supports cheap inference "enhancements":
#   - checkpoint type sweep: best,last
#   - state sweep: model_ema1,model_ema2,model
#   - ODE steps sweep: 1,2,4
#   - tile stride sweep: 128,64
# and optional output-level ensemble over all produced directories.
#
# IMPORTANT: SCENE_JSON must map the INPUT_DIR filenames. For official test data,
# either provide a test filename->label JSON or use SCENE_CKPT to predict labels.

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/Drop}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/manual_blur_scene_sweep}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/submission_history.csv}

SCENE_MODE=${SCENE_MODE:-blur2} # blur2 or dn_blur4
case "${SCENE_MODE}" in
  blur2)
    DEFAULT_CKPT="run/train/ablation_b16_manual_blur_scene_no_head_3x3090/b16_blur_2scene_no_head/16"
    DEFAULT_SCENE_JSON="${DATA_ROOT}/RainDrop_Train/Drop_blur_2scene.json"
    ;;
  dn_blur4)
    DEFAULT_CKPT="run/train/ablation_b16_manual_blur_scene_no_head_3x3090/b16_dn_blur_4scene_no_head/16"
    DEFAULT_SCENE_JSON="${DATA_ROOT}/RainDrop_Train/Drop_dn_blur_4scene.json"
    ;;
  *)
    echo "SCENE_MODE must be blur2 or dn_blur4; got: ${SCENE_MODE}" >&2
    exit 2
    ;;
esac

JIT_CKPT=${JIT_CKPT:-${DEFAULT_CKPT}}
SCENE_JSON=${SCENE_JSON:-${DEFAULT_SCENE_JSON}}
SCENE_CKPT=${SCENE_CKPT:-}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-128}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}

JIT_CKPT_TYPES=${JIT_CKPT_TYPES:-best,last}
STATE_KEYS=${STATE_KEYS:-model_ema1,model_ema2,model}
STEPS_LIST=${STEPS_LIST:-1,2,4}
STRIDES=${STRIDES:-128,64}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-32}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
MODEL_NAME_PREFIX=${MODEL_NAME_PREFIX:-jit_${SCENE_MODE}}
NOTES=${NOTES:-manual_blur_scene_sweep}

CREATE_ENSEMBLE=${CREATE_ENSEMBLE:-1}
ENSEMBLE_NAME=${ENSEMBLE_NAME:-${MODEL_NAME_PREFIX}_ensemble}
ENSEMBLE_WEIGHTS=${ENSEMBLE_WEIGHTS:-}
REMOVE_IMAGES_AFTER_ZIP=${REMOVE_IMAGES_AFTER_ZIP:-0}

split_list() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  SPLIT_RESULT=(${raw})
}

mkdir -p "${OUTPUT_ROOT}"

if [[ -n "${SCENE_CKPT}" ]]; then
  scene_stamp=$(date +%Y%m%d_%H%M%S)
  scene_dir="${OUTPUT_ROOT}/scene_predictions"
  mkdir -p "${scene_dir}"
  SCENE_JSON="${scene_dir}/${MODEL_NAME_PREFIX}_${scene_stamp}.json"
  SCENE_CSV="${scene_dir}/${MODEL_NAME_PREFIX}_${scene_stamp}.csv"

  echo "============================================================"
  echo "[Scene] Predict labels for ${INPUT_DIR}"
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
fi

if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "Missing SCENE_JSON: ${SCENE_JSON}" >&2
  echo "Provide SCENE_JSON matching INPUT_DIR filenames, or provide SCENE_CKPT." >&2
  exit 2
fi

split_list "${JIT_CKPT_TYPES}"
ckpt_type_list=("${SPLIT_RESULT[@]}")
split_list "${STATE_KEYS}"
state_key_list=("${SPLIT_RESULT[@]}")
split_list "${STEPS_LIST}"
steps_list=("${SPLIT_RESULT[@]}")
split_list "${STRIDES}"
stride_list=("${SPLIT_RESULT[@]}")

output_dirs=()
for ckpt_type in "${ckpt_type_list[@]}"; do
  for state_key in "${state_key_list[@]}"; do
    for steps in "${steps_list[@]}"; do
      for stride in "${stride_list[@]}"; do
        run_model_name="${MODEL_NAME_PREFIX}_${ckpt_type}_${state_key}_s${steps}_r${stride}"
        run_notes="${NOTES}; scene_mode=${SCENE_MODE}; ckpt_type=${ckpt_type}; state_key=${state_key}; steps=${steps}; stride=${stride}"

        echo "============================================================"
        echo "[JiT] ${run_model_name}"
        echo "Checkpoint: ${JIT_CKPT}"
        echo "Scene JSON: ${SCENE_JSON}"
        echo "============================================================"

        before_count=$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${run_model_name}_${ckpt_type}_*" 2>/dev/null | wc -l || true)
        python submit_jit.py \
          --input-dir "${INPUT_DIR}" \
          --checkpoint "${JIT_CKPT}" \
          --ckpt_type "${ckpt_type}" \
          --output-root "${OUTPUT_ROOT}" \
          --history-csv "${HISTORY_CSV}" \
          --model-name "${run_model_name}" \
          --state-key "${state_key}" \
          --use-scene \
          --scene-json "${SCENE_JSON}" \
          --steps "${steps}" \
          --stride "${stride}" \
          --tile-batch-size "${TILE_BATCH_SIZE}" \
          --device "${DEVICE}" \
          --amp-dtype "${AMP_DTYPE}" \
          --notes "${run_notes}"

        latest_dir=$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${run_model_name}_${ckpt_type}_*" 2>/dev/null | sort | tail -n 1 || true)
        after_count=$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${run_model_name}_${ckpt_type}_*" 2>/dev/null | wc -l || true)
        if [[ -z "${latest_dir}" || "${after_count}" == "${before_count}" ]]; then
          echo "Could not locate output dir for ${run_model_name}; ensemble will skip it." >&2
        else
          output_dirs+=("${latest_dir}")
        fi
      done
    done
  done
done

if [[ "${CREATE_ENSEMBLE}" == "1" ]]; then
  if [[ "${#output_dirs[@]}" -lt 2 ]]; then
    echo "Need at least two output dirs for ensemble; got ${#output_dirs[@]}." >&2
    exit 0
  fi
  ensemble_stamp=$(date +%Y%m%d_%H%M%S)
  ensemble_dir="${OUTPUT_ROOT}/${ENSEMBLE_NAME}_${ensemble_stamp}"
  ensemble_zip="${OUTPUT_ROOT}/${ENSEMBLE_NAME}_${ensemble_stamp}.zip"
  echo "============================================================"
  echo "[Ensemble] Averaging ${#output_dirs[@]} output dirs"
  echo "Output: ${ensemble_zip}"
  echo "============================================================"
  python tools/ensemble_submission_dirs.py \
    --input-dirs "${output_dirs[@]}" \
    --weights "${ENSEMBLE_WEIGHTS}" \
    --output-dir "${ensemble_dir}" \
    --archive-path "${ensemble_zip}" \
    --history-csv "${HISTORY_CSV}" \
    --model-name "${ENSEMBLE_NAME}" \
    --input-dir "${INPUT_DIR}" \
    --notes "${NOTES}; ensemble over ${#output_dirs[@]} sweep outputs"

  if [[ "${REMOVE_IMAGES_AFTER_ZIP}" == "1" ]]; then
    rm -rf "${ensemble_dir}"
  fi
fi

echo "Manual blur scene submission sweep finished."
