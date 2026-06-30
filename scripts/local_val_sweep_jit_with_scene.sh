#!/usr/bin/env bash
set -euo pipefail

# Local validation sweep based on submit_jit_with_scene.sh.
# It fixes a 100-image Drop/Clear subset, runs JiT submit-style inference,
# evaluates PSNR(Y), SSIM(Y), LPIPS, and writes a local sweep CSV.

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_ROOT=${VAL_ROOT:-${DATA_ROOT}/jit_local_val_200}
VAL_NUM_IMAGES=${VAL_NUM_IMAGES:-200}
VAL_SEED=${VAL_SEED:-2026}

JIT_CKPT=${JIT_CKPT:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SOURCE_SCENE_JSON=${SOURCE_SCENE_JSON:-}
SCENE_JSON=${SCENE_JSON:-}

OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/local_val_sweep_200}
LOCAL_CSV=${LOCAL_CSV:-${OUTPUT_ROOT}/local_val_sweep_200.csv}
SUBMIT_HISTORY_CSV=${SUBMIT_HISTORY_CSV:-${OUTPUT_ROOT}/submission_history_200.csv}

MODEL_NAME_PREFIX=${MODEL_NAME_PREFIX:-jit_local}
JIT_CKPT_TYPES=${JIT_CKPT_TYPES:-best,last}
STATE_KEYS=${STATE_KEYS:-model_ema1,model_ema2,model}
STEPS_LIST=${STEPS_LIST:-1,2,4}
STRIDES=${STRIDES:-128,64}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-8}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-8}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}
NOTES=${NOTES:-local_val_sweep}

split_list() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  SPLIT_RESULT=(${raw})
}

mkdir -p "${OUTPUT_ROOT}"

subset_scene_args=()
if [[ -n "${SOURCE_SCENE_JSON}" ]]; then
  subset_scene_args+=(--scene-json "${SOURCE_SCENE_JSON}" --output-scene-json "${VAL_ROOT}/$(basename "${SOURCE_SCENE_JSON}")")
fi

python tools/make_fixed_val_subset.py \
  --data-root "${DATA_PATH}" \
  --output-root "${VAL_ROOT}" \
  --num-samples "${VAL_NUM_IMAGES}" \
  --seed "${VAL_SEED}" \
  "${subset_scene_args[@]}"

INPUT_DIR="${VAL_ROOT}/Drop"
CLEAR_DIR="${VAL_ROOT}/Clear"

if [[ -z "${SCENE_JSON}" ]]; then
  if [[ -n "${SOURCE_SCENE_JSON}" ]]; then
    SCENE_JSON="${VAL_ROOT}/$(basename "${SOURCE_SCENE_JSON}")"
  else
    scene_stamp=$(date +%Y%m%d_%H%M%S)
    scene_dir="${OUTPUT_ROOT}/scene_predictions"
    mkdir -p "${scene_dir}"
    SCENE_JSON="${scene_dir}/${MODEL_NAME_PREFIX}_${scene_stamp}.json"
    SCENE_CSV="${scene_dir}/${MODEL_NAME_PREFIX}_${scene_stamp}.csv"

    echo "============================================================"
    echo "[Scene] Predict labels for fixed local validation subset"
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
        run_model_name="${MODEL_NAME_PREFIX}_${ckpt_type}_${state_key}_s${steps}_r${stride}"
        run_notes="${NOTES}; ckpt_type=${ckpt_type}; state_key=${state_key}; steps=${steps}; stride=${stride}"

        echo "============================================================"
        echo "[Local JiT] ${run_model_name}"
        echo "Checkpoint: ${JIT_CKPT}"
        echo "Scene JSON: ${SCENE_JSON}"
        echo "============================================================"

        before_count=$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${run_model_name}_${ckpt_type}_*" 2>/dev/null | wc -l || true)
        python submit_jit.py \
          --input-dir "${INPUT_DIR}" \
          --checkpoint "${JIT_CKPT}" \
          --ckpt_type "${ckpt_type}" \
          --output-root "${OUTPUT_ROOT}" \
          --history-csv "${SUBMIT_HISTORY_CSV}" \
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
          echo "Could not locate output dir for ${run_model_name}" >&2
          exit 2
        fi

        python tools/evaluate_submission_dir.py \
          --prediction-dir "${latest_dir}" \
          --clear-dir "${CLEAR_DIR}" \
          --csv "${LOCAL_CSV}" \
          --model-name "${run_model_name}" \
          --checkpoint "${JIT_CKPT}" \
          --ckpt-type "${ckpt_type}" \
          --state-key "${state_key}" \
          --steps "${steps}" \
          --stride "${stride}" \
          --tile-batch-size "${TILE_BATCH_SIZE}" \
          --scene-json "${SCENE_JSON}" \
          --device "${DEVICE}" \
          --notes "${run_notes}"
      done
    done
  done
done

echo "Local validation sweep finished: ${LOCAL_CSV}"
