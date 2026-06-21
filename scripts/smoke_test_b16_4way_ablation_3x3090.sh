#!/usr/bin/env bash
set -euo pipefail

# Each setting runs two epochs with one train iteration per epoch.
# Epoch 1 activates LPIPS and tests its backward-memory peak.
#
# 在项目根目录执行：
# bash scripts/smoke_test_b16_4way_ablation_3x3090.sh
#
# 也可以覆盖默认配置：
# DATA_ROOT=/data/eccv_dn \
# CKPT=/data/jit-b-16 \
# bash scripts/smoke_test_b16_4way_ablation_3x3090.sh

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}

DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29710}

CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train_smoke/jit_b16}

SCENE_TRAIN_PATH=${SCENE_TRAIN_PATH:-${DATA_PATH}/Drop_scen_pred.json}
SCENE_VAL_PATH=${SCENE_VAL_PATH:-${SCENE_TRAIN_PATH}}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}

scene_args=()

if [[ -n "${SCENE_TRAIN_PATH}" ]]; then
  scene_args+=(--scene_train_path "${SCENE_TRAIN_PATH}")
fi

if [[ -n "${SCENE_VAL_PATH}" ]]; then
  scene_args+=(--scene_val_path "${SCENE_VAL_PATH}")
fi

run_smoke() {
  local name="$1"
  local use_scene="$2"
  local use_head="$3"
  local port="$4"
  local output_dir="${OUT_ROOT}/${name}/16"

  echo "============================================================"
  echo "[Smoke] ${name}"
  echo "use_scene_dataset=${use_scene}"
  echo "use_bg_subnet=${use_head}"
  echo "Epoch 0 tests normal loss; epoch 1 tests LPIPS backward."
  echo "data_path=${DATA_PATH}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    main_jit.py \
    --model "${MODEL}" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr_schedule cosine \
    --epochs 2 \
    --warmup_epochs 1 \
    --eval_epoch 1 \
    --eval_num_images "${EVAL_NUM_IMAGES}" \
    --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --cfg 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --save_last_freq 1 \
    --log_freq 1 \
    --output_dir "${output_dir}" \
    --use_bg_subnet "${use_head}" \
    --use_scene_dataset "${use_scene}" \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT}" \
    --online_eval \
    "${scene_args[@]}"
}

run_smoke "b16_no_scene_no_head" 0 0 "$((MASTER_PORT_BASE + 0))"
run_smoke "b16_scene_no_head"    1 0 "$((MASTER_PORT_BASE + 1))"
run_smoke "b16_no_scene_head"    0 1 "$((MASTER_PORT_BASE + 2))"
run_smoke "b16_scene_head"       1 1 "$((MASTER_PORT_BASE + 3))"

echo "All four smoke tests passed, including LPIPS backward."
