#!/usr/bin/env bash
set -euo pipefail

# Single-GPU smoke test for the final pseudo-label JiT scripts.
#
# By default this runs P1/P2/P3/P4 with 1 epoch and 1 train step per stage.
# Set RUN_CASES to a subset when a checkpoint/model is unavailable or too large:
#   RUN_CASES="P1 P4" GPU=0 DATA_ROOT=/path/to/eccv_dn bash scripts_pseduo/smoke_test_single_gpu_final.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
GPU=${GPU:-0}
RUN_CASES=${RUN_CASES:-"P1 P2 P3 P4"}
RUN_INFER=${RUN_INFER:-1}
SMOKE_ROOT=${SMOKE_ROOT:-${ROOT_DIR}/run/train_smoke/final_single_gpu}

COMMON_ENV=(
  GPU="${GPU}"
  ROOT_DIR="${ROOT_DIR}"
  DATA_ROOT="${DATA_ROOT}"
  DATA_PATH="${DATA_PATH}"
  VAL_DATA_PATH="${DATA_PATH}"
  EVAL_EPOCH=1
  EVAL_NUM_IMAGES=1
  NUM_WORKERS=0
  SAVE_LAST_FREQ=1
  LOG_FREQ=1
  ONLINE_EVAL=1
  MAX_TRAIN_STEPS=1
)

has_case() {
  local wanted="$1"
  [[ " ${RUN_CASES} " == *" ${wanted} "* ]]
}

echo "============================================================"
echo "[Final single-GPU smoke]"
echo "ROOT_DIR=${ROOT_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "DATA_PATH=${DATA_PATH}"
echo "GPU=${GPU}"
echo "RUN_CASES=${RUN_CASES}"
echo "SMOKE_ROOT=${SMOKE_ROOT}"
echo "============================================================"

if has_case "P1"; then
  echo "============================================================"
  echo "[Smoke P1] B16 focus refiner finetune"
  echo "============================================================"
  env "${COMMON_ENV[@]}" \
    EPOCHS=1 \
    BATCH_SIZE="${P1_BATCH_SIZE:-1}" \
    OUTPUT_DIR="${SMOKE_ROOT}/p1_b16_focus_refiner/16" \
    bash "${SCRIPT_DIR}/p1_train_pseudo_1x5090_b16_focus_refiner_3h.sh"
fi

if has_case "P2"; then
  echo "============================================================"
  echo "[Smoke P2] B16 dn_blur4 JIT + refiner"
  echo "============================================================"
  env "${COMMON_ENV[@]}" \
    RUN_STAGES="JIT REFINER" \
    JIT_EPOCHS=1 \
    REFINER_EPOCHS=1 \
    JIT_BATCH_SIZE="${P2_JIT_BATCH_SIZE:-1}" \
    REFINER_BATCH_SIZE="${P2_REFINER_BATCH_SIZE:-1}" \
    JIT_OUTPUT_DIR="${SMOKE_ROOT}/p2_b16_dn_blur4_jit/16" \
    REFINER_OUTPUT_DIR="${SMOKE_ROOT}/p2_b16_dn_blur4_refiner/16" \
    REFINER_CKPT="${SMOKE_ROOT}/p2_b16_dn_blur4_jit/16/checkpoint-last.pth" \
    bash "${SCRIPT_DIR}/p2_train_pseudo_single_gpu_b16_dn_blur4.sh"
fi

if has_case "P3"; then
  echo "============================================================"
  echo "[Smoke P3] H16 blur2 JIT + refiner"
  echo "============================================================"
  env "${COMMON_ENV[@]}" \
    RUN_STAGES="JIT REFINER" \
    EPOCHS_JIT=1 \
    EPOCHS_REFINER=1 \
    BATCH_SIZE_JIT="${P3_JIT_BATCH_SIZE:-1}" \
    BATCH_SIZE_REFINER="${P3_REFINER_BATCH_SIZE:-1}" \
    JIT_OUTPUT_DIR="${SMOKE_ROOT}/p3_h16_blur2_jit/16" \
    REFINER_OUTPUT_DIR="${SMOKE_ROOT}/p3_h16_blur2_refiner/16" \
    CKPT_H_BLUR="${SMOKE_ROOT}/p3_h16_blur2_jit/16" \
    bash "${SCRIPT_DIR}/p3_train_pseudo_1xA100_h16_blur2_jit_refiner_36h.sh"
fi

if has_case "P4"; then
  echo "============================================================"
  echo "[Smoke P4] B16 focus JIT + refiner"
  echo "============================================================"
  env "${COMMON_ENV[@]}" \
    RUN_STAGES="JIT REFINER" \
    JIT_EPOCHS=1 \
    REFINER_EPOCHS=1 \
    JIT_BATCH_SIZE="${P4_JIT_BATCH_SIZE:-1}" \
    REFINER_BATCH_SIZE="${P4_REFINER_BATCH_SIZE:-1}" \
    JIT_OUTPUT_DIR="${SMOKE_ROOT}/p4_b16_focus_jit/16" \
    REFINER_OUTPUT_DIR="${SMOKE_ROOT}/p4_b16_focus_refiner/16" \
    REFINER_CKPT="${SMOKE_ROOT}/p4_b16_focus_jit/16/checkpoint-last.pth" \
    bash "${SCRIPT_DIR}/p4_train_pseudo_single_gpu_b16_focus_from_best_refiner_jit100_refiner300.sh"
fi

if [[ "${RUN_INFER}" == "1" ]]; then
  echo "============================================================"
  echo "[Smoke infer] one image through submit_jit_from_local_config.sh"
  echo "============================================================"
  smoke_input="${SMOKE_ROOT}/infer_input"
  mkdir -p "${smoke_input}"
  first_image=$(find "${DATA_ROOT}/test-input" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort | head -n 1 || true)
  if [[ -z "${first_image}" ]]; then
    first_image=$(find "${DATA_PATH}/Drop" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort | head -n 1 || true)
  fi
  if [[ -z "${first_image}" ]]; then
    echo "No image found for inference smoke under ${DATA_ROOT}/test-input or ${DATA_PATH}/Drop" >&2
    exit 2
  fi
  cp "${first_image}" "${smoke_input}/"

  infer_ckpt="${INFER_CKPT:-${SMOKE_ROOT}/p4_b16_focus_refiner/16}"
  if [[ ! -f "${infer_ckpt}" && ! -f "${infer_ckpt}/checkpoint-last.pth" ]]; then
    echo "Skipping inference smoke because INFER_CKPT is missing: ${infer_ckpt}" >&2
  else
    GPU="${GPU}" \
    DATA_ROOT="${DATA_ROOT}" \
    INPUT_DIR="${smoke_input}" \
    OUTPUT_ROOT="${SMOKE_ROOT}/infer" \
    HISTORY_CSV="${SMOKE_ROOT}/infer/submission_history.csv" \
    JIT_CKPT="${infer_ckpt}" \
    JIT_CKPT_TYPE=last \
    STATE_KEY=model_ema1 \
    STEPS=1 \
    STRIDE=256 \
    TILE_BATCH_SIZE=1 \
    TTA_HFLIP=0 \
    TTA_ROT90=0 \
    UPDATE_SCENE_FROM_FOCUS_PSEUDO=0 \
    SCENE_JSON="" \
    SCENE_BATCH_SIZE=1 \
    SCENE_NUM_WORKERS=0 \
    MODEL_NAME=smoke_final_single_gpu \
    REMOVE_IMAGES_AFTER_ZIP=1 \
    bash "${SCRIPT_DIR}/submit_jit_from_local_config.sh"
  fi
fi

echo "============================================================"
echo "Final single-GPU smoke finished."
echo "Artifacts: ${SMOKE_ROOT}"
echo "============================================================"
