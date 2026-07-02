#!/usr/bin/env bash
set -euo pipefail

# Run the three planned 1x5090 MSDT-refiner trainings for focus-2scene JiT.
# Default runs all plans in order. Example:
#   RUN_PLANS="A B" bash scripts/train_b16_focus_2scene_msdt_refiner_plans_1x5090.sh

BASE_SCRIPT=${BASE_SCRIPT:-scripts/train_b16_focus_2scene_msdt_refiner_1x5090.sh}
RUN_PLANS=${RUN_PLANS:-"A B C"}

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-8}
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

run_plan() {
  local plan="$1"
  local output_dir="$2"
  local lr="$3"
  local epochs="$4"
  local eval_epoch="$5"
  local max_residual="$6"
  local edge_weight="$7"
  local freq_weight="$8"

  echo "============================================================"
  echo "[MSDT refiner plan ${plan}]"
  echo "output_dir=${output_dir}"
  echo "lr=${lr}, epochs=${epochs}, eval_epoch=${eval_epoch}"
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
  EPOCHS="${epochs}" \
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
    A|a)
      run_plan \
        "A mid_conservative" \
        "run/train/b16_focus_2scene_msdt_refiner_plan_a_mid_conservative_1x5090/16" \
        "5e-5" \
        "160" \
        "2" \
        "0.15" \
        "0.03" \
        "0.01"
      ;;
    B|b)
      run_plan \
        "B conservative" \
        "run/train/b16_focus_2scene_msdt_refiner_plan_b_conservative_1x5090/16" \
        "3e-5" \
        "120" \
        "2" \
        "0.10" \
        "0.02" \
        "0.005"
      ;;
    C|c)
      run_plan \
        "C baseline" \
        "run/train/b16_focus_2scene_msdt_refiner_plan_c_baseline_1x5090/16" \
        "1e-4" \
        "300" \
        "5" \
        "0.25" \
        "0.05" \
        "0.05"
      ;;
    *)
      echo "Unknown plan: ${plan}. Use RUN_PLANS=\"A B C\"." >&2
      exit 2
      ;;
  esac
done

echo "All requested MSDT refiner plans finished: ${RUN_PLANS}"
