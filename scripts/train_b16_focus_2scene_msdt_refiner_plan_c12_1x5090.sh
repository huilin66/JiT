#!/usr/bin/env bash
set -euo pipefail

# Follow-up sweep around the winning Plan C for focus-2scene JiT-B/16.
# C1 is stronger than Plan C; C2 sits between Plan A baseline and Plan C.
#
# Default runs both plans in order. Example:
#   RUN_PLANS="C2" bash scripts/train_b16_focus_2scene_msdt_refiner_plan_c12_1x5090.sh

BASE_SCRIPT=${BASE_SCRIPT:-scripts/train_b16_focus_2scene_msdt_refiner_1x5090.sh}
RUN_PLANS=${RUN_PLANS:-"C1 C2"}

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}
OUT_ROOT=${OUT_ROOT:-run/train}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-300}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}
RESUME_STATE_KEY=${RESUME_STATE_KEY:-model_ema2}

REFINER_BASE_DIM=${REFINER_BASE_DIM:-32}
REFINER_NUM_BLOCKS=${REFINER_NUM_BLOCKS:-2}
REFINER_USE_FREQUENCY=${REFINER_USE_FREQUENCY:-1}

run_exp() {
  local plan="$1"
  local name="$2"
  local lr="$3"
  local max_residual="$4"
  local edge_weight="$5"
  local freq_weight="$6"
  local eval_epoch="5"
  local output_dir="${OUT_ROOT}/b16_focus_2scene_msdt_refiner_plan_c12_${name}_1x5090/16"

  echo "============================================================"
  echo "[MSDT refiner plan_c12 ${plan}] ${name}"
  echo "output_dir=${output_dir}"
  echo "lr=${lr}, epochs=${EPOCHS}, eval_epoch=${eval_epoch}"
  echo "max_residual=${max_residual}, edge=${edge_weight}, freq=${freq_weight}"
  echo "============================================================"

  GPU="${GPU}" \
  DATA_ROOT="${DATA_ROOT}" \
  DATA_PATH="${DATA_PATH}" \
  VAL_DATA_PATH="${VAL_DATA_PATH}" \
  CKPT="${CKPT}" \
  OUTPUT_DIR="${output_dir}" \
  SCENE_FOCUS_2_PATH="${SCENE_FOCUS_2_PATH}" \
  MODEL="${MODEL}" \
  IMG_SIZE="${IMG_SIZE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LR="${lr}" \
  EPOCHS="${EPOCHS}" \
  WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
  EVAL_EPOCH="${eval_epoch}" \
  EVAL_NUM_IMAGES="${EVAL_NUM_IMAGES}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS}" \
  SAVE_LAST_FREQ="${SAVE_LAST_FREQ}" \
  LOG_FREQ="${LOG_FREQ}" \
  ONLINE_EVAL="${ONLINE_EVAL}" \
  RESUME_STATE_KEY="${RESUME_STATE_KEY}" \
  REFINER_BASE_DIM="${REFINER_BASE_DIM}" \
  REFINER_NUM_BLOCKS="${REFINER_NUM_BLOCKS}" \
  REFINER_USE_FREQUENCY="${REFINER_USE_FREQUENCY}" \
  REFINER_MAX_RESIDUAL="${max_residual}" \
  LOSS_EDGE_WEIGHT="${edge_weight}" \
  LOSS_FREQ_WEIGHT="${freq_weight}" \
  bash "${BASE_SCRIPT}"
}

for plan in ${RUN_PLANS}; do
  case "${plan}" in
    C1|c1)
      run_exp "01_C1" "01_c1_higher_than_c" "2.5e-4" "0.35" "0.10" "0.10"
      ;;
    C2|c2)
      run_exp "02_C2" "02_c2_between_a_and_c" "1.5e-4" "0.275" "0.065" "0.065"
      ;;
    *)
      echo "Unknown plan: ${plan}. Use RUN_PLANS=\"C1 C2\"." >&2
      exit 2
      ;;
  esac
done

echo "All requested MSDT refiner plan_c12 runs finished: ${RUN_PLANS}"
