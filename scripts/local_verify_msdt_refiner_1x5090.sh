#!/usr/bin/env bash
set -euo pipefail

# Local 5090 verification for "MSDT refiner fixed it or over-fixed it".
# It creates a fixed Drop/Clear validation subset, runs JiT-only and
# JiT+MSDT-refiner inference, then writes per-image delta metrics to CSV.
#
# Fast compare existing dirs:
# RUN_INFER=0 JIT_ONLY_DIR=... REFINER_DIR=... bash scripts/local_verify_msdt_refiner_1x5090.sh

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_ROOT=${VAL_ROOT:-${DATA_ROOT}/jit_refiner_local_val_200}
VAL_NUM_IMAGES=${VAL_NUM_IMAGES:-200}
VAL_SEED=${VAL_SEED:-2026}

OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/msdt_refiner_verify}
COMPARE_CSV=${COMPARE_CSV:-${OUTPUT_ROOT}/jit_vs_msdt_refiner_compare.csv}
SUMMARY_JSON=${SUMMARY_JSON:-${OUTPUT_ROOT}/jit_vs_msdt_refiner_compare.summary.json}

BASE_JIT_CKPT=${BASE_JIT_CKPT:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
REFINER_CKPT=${REFINER_CKPT:-run/train/b16_focus_2scene_msdt_refiner_1x5090/16}
BASE_CKPT_TYPE=${BASE_CKPT_TYPE:-best}
REFINER_CKPT_TYPE=${REFINER_CKPT_TYPE:-last}
STATE_KEY=${STATE_KEY:-model_ema2}

SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SOURCE_SCENE_JSON=${SOURCE_SCENE_JSON:-}
SCENE_JSON=${SCENE_JSON:-}

STEPS=${STEPS:-1}
STRIDE=${STRIDE:-64}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-8}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-8}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}
RUN_INFER=${RUN_INFER:-1}
JIT_ONLY_DIR=${JIT_ONLY_DIR:-}
REFINER_DIR=${REFINER_DIR:-}

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
    SCENE_JSON="${scene_dir}/refiner_verify_${scene_stamp}.json"
    SCENE_CSV="${scene_dir}/refiner_verify_${scene_stamp}.csv"

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

latest_output_dir() {
  local model_name="$1"
  local ckpt_type="$2"
  find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${model_name}_${ckpt_type}_*" 2>/dev/null | sort | tail -n 1
}

if [[ "${RUN_INFER}" == "1" ]]; then
  echo "============================================================"
  echo "[JiT-only local inference]"
  echo "ckpt=${BASE_JIT_CKPT}, ckpt_type=${BASE_CKPT_TYPE}, state=${STATE_KEY}"
  echo "scene=${SCENE_JSON}"
  echo "============================================================"
  python submit_jit.py \
    --input-dir "${INPUT_DIR}" \
    --checkpoint "${BASE_JIT_CKPT}" \
    --ckpt_type "${BASE_CKPT_TYPE}" \
    --output-root "${OUTPUT_ROOT}" \
    --history-csv "${OUTPUT_ROOT}/submission_history.csv" \
    --model-name "verify_jit_only" \
    --state-key "${STATE_KEY}" \
    --use-scene \
    --scene-json "${SCENE_JSON}" \
    --steps "${STEPS}" \
    --stride "${STRIDE}" \
    --tile-batch-size "${TILE_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --amp-dtype "${AMP_DTYPE}" \
    --notes "local_refiner_verify_jit_only"
  JIT_ONLY_DIR="$(latest_output_dir verify_jit_only "${BASE_CKPT_TYPE}")"

  echo "============================================================"
  echo "[JiT+MSDT-refiner local inference]"
  echo "ckpt=${REFINER_CKPT}, ckpt_type=${REFINER_CKPT_TYPE}, state=${STATE_KEY}"
  echo "scene=${SCENE_JSON}"
  echo "============================================================"
  python submit_jit.py \
    --input-dir "${INPUT_DIR}" \
    --checkpoint "${REFINER_CKPT}" \
    --ckpt_type "${REFINER_CKPT_TYPE}" \
    --output-root "${OUTPUT_ROOT}" \
    --history-csv "${OUTPUT_ROOT}/submission_history.csv" \
    --model-name "verify_msdt_refiner" \
    --state-key "${STATE_KEY}" \
    --use-scene \
    --scene-json "${SCENE_JSON}" \
    --steps "${STEPS}" \
    --stride "${STRIDE}" \
    --tile-batch-size "${TILE_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --amp-dtype "${AMP_DTYPE}" \
    --notes "local_refiner_verify_msdt_refiner"
  REFINER_DIR="$(latest_output_dir verify_msdt_refiner "${REFINER_CKPT_TYPE}")"
fi

echo "============================================================"
echo "[Compare]"
echo "drop=${INPUT_DIR}"
echo "clear=${CLEAR_DIR}"
echo "jit=${JIT_ONLY_DIR}"
echo "refiner=${REFINER_DIR}"
echo "csv=${COMPARE_CSV}"
echo "============================================================"

python tools/compare_jit_refiner_outputs.py \
  --drop-dir "${INPUT_DIR}" \
  --clear-dir "${CLEAR_DIR}" \
  --jit-dir "${JIT_ONLY_DIR}" \
  --refiner-dir "${REFINER_DIR}" \
  --csv "${COMPARE_CSV}" \
  --summary-json "${SUMMARY_JSON}" \
  --device "${DEVICE}"

echo "MSDT refiner local verification finished."
