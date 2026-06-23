#!/usr/bin/env bash
set -euo pipefail

# Stage 1: load an existing JiT checkpoint, freeze JiT, and train only the
# MSDT-style detail refiner. Run from the project root.
#
# Example:
# CKPT=/path/to/trained_jit_b16 \
# DATA_PATH=/path/to/RainDrop_Train2 \
# bash scripts/train_b16_msdt_refiner_3x3090.sh

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-29630}

DATA_PATH=${DATA_PATH:-/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/jit-b-16}
OUTPUT_DIR=${OUTPUT_DIR:-run/train/jit_b16_msdt_refiner_stage1/16}

SCENE_TRAIN_PATH=${SCENE_TRAIN_PATH:-}
SCENE_VAL_PATH=${SCENE_VAL_PATH:-${SCENE_TRAIN_PATH}}
USE_SCENE_DATASET=${USE_SCENE_DATASET:-0}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-300}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}

REFINER_BASE_DIM=${REFINER_BASE_DIM:-32}
REFINER_NUM_BLOCKS=${REFINER_NUM_BLOCKS:-2}
REFINER_USE_FREQUENCY=${REFINER_USE_FREQUENCY:-1}
REFINER_MAX_RESIDUAL=${REFINER_MAX_RESIDUAL:-0.25}
LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.05}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.05}

scene_args=()
if [[ -n "${SCENE_TRAIN_PATH}" ]]; then
  scene_args+=(--scene_train_path "${SCENE_TRAIN_PATH}")
fi
if [[ -n "${SCENE_VAL_PATH}" ]]; then
  scene_args+=(--scene_val_path "${SCENE_VAL_PATH}")
fi

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  main_jit.py \
  --model "${MODEL}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lr_schedule cosine \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --eval_epoch "${EVAL_EPOCH}" \
  --eval_num_images "${EVAL_NUM_IMAGES}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
  --cfg 1.0 \
  --num_workers "${NUM_WORKERS}" \
  --save_last_freq 5 \
  --log_freq 50 \
  --output_dir "${OUTPUT_DIR}" \
  --use_bg_subnet 0 \
  --use_detail_refiner 1 \
  --freeze_jit 1 \
  --refiner_base_dim "${REFINER_BASE_DIM}" \
  --refiner_num_blocks "${REFINER_NUM_BLOCKS}" \
  --refiner_use_frequency "${REFINER_USE_FREQUENCY}" \
  --refiner_max_residual "${REFINER_MAX_RESIDUAL}" \
  --loss_edge_weight "${LOSS_EDGE_WEIGHT}" \
  --loss_freq_weight "${LOSS_FREQ_WEIGHT}" \
  --use_scene_dataset "${USE_SCENE_DATASET}" \
  --data_path "${DATA_PATH}" \
  --val_data_path "${VAL_DATA_PATH}" \
  --resume "${CKPT}" \
  --resume_state_key model_ema1 \
  --resume_optimizer 0 \
  --online_eval \
  "${scene_args[@]}"
