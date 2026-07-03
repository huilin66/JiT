#!/usr/bin/env bash
set -euo pipefail

# Balanced MSDT-refiner plan for the original focus-2scene JiT-B/16.
# This sits between the old strong refiner and the conservative Plan A:
#   old:    lr=1e-4, max_residual=0.25, edge/freq=0.05/0.05
#   Plan A: lr=5e-5, max_residual=0.15, edge/freq=0.03/0.01
#   this:   lr=7e-5, max_residual=0.20, edge/freq=0.04/0.02

BASE_SCRIPT=${BASE_SCRIPT:-scripts/train_b16_focus_2scene_msdt_refiner_1x5090.sh}

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

OUTPUT_DIR=${OUTPUT_DIR:-run/train/b16_focus_2scene_msdt_refiner_plan_d_balanced_1x5090/16}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-7e-5}
EPOCHS=${EPOCHS:-220}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-2}
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
REFINER_MAX_RESIDUAL=${REFINER_MAX_RESIDUAL:-0.20}
LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.04}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.02}

echo "============================================================"
echo "[MSDT refiner Plan D balanced] original focus 2scene JiT-B/16 on 1x5090"
echo "ckpt=${CKPT}"
echo "resume_state_key=${RESUME_STATE_KEY}"
echo "lr=${LR}, epochs=${EPOCHS}, eval_epoch=${EVAL_EPOCH}"
echo "max_residual=${REFINER_MAX_RESIDUAL}, edge=${LOSS_EDGE_WEIGHT}, freq=${LOSS_FREQ_WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "============================================================"

GPU="${GPU}" \
DATA_ROOT="${DATA_ROOT}" \
DATA_PATH="${DATA_PATH}" \
VAL_DATA_PATH="${VAL_DATA_PATH}" \
CKPT="${CKPT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SCENE_FOCUS_2_PATH="${SCENE_FOCUS_2_PATH}" \
MODEL="${MODEL}" \
IMG_SIZE="${IMG_SIZE}" \
BATCH_SIZE="${BATCH_SIZE}" \
LR="${LR}" \
EPOCHS="${EPOCHS}" \
WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
EVAL_EPOCH="${EVAL_EPOCH}" \
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
REFINER_MAX_RESIDUAL="${REFINER_MAX_RESIDUAL}" \
LOSS_EDGE_WEIGHT="${LOSS_EDGE_WEIGHT}" \
LOSS_FREQ_WEIGHT="${LOSS_FREQ_WEIGHT}" \
bash "${BASE_SCRIPT}"

echo "MSDT refiner Plan D balanced training finished."
