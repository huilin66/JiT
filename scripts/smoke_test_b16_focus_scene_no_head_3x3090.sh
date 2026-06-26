#!/usr/bin/env bash
set -euo pipefail

# Smoke test 2scene and 4scene focus-label no-head runs on 3 GPUs.

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}

DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29720}

CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train_smoke/jit_b16_focus_scene_no_head}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}

run_smoke() {
  local scene_name="$1"
  local scene_path="$2"
  local port="$3"
  local output_dir="${OUT_ROOT}/b16_focus_${scene_name}_no_head/16"

  echo "============================================================"
  echo "[Smoke ${scene_name}] b16_focus_${scene_name}_no_head"
  echo "use_scene_dataset=1"
  echo "use_bg_subnet=0"
  echo "scene_path=${scene_path}"
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
    --use_bg_subnet 0 \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT}" \
    --online_eval \
    --scene_train_path "${scene_path}" \
    --scene_val_path "${scene_path}"
}

run_smoke "2scene" "${SCENE_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}" "$((MASTER_PORT_BASE + 0))"
run_smoke "4scene" "${SCENE_4_PATH:-${DATA_PATH}/Drop_focus_4scene.json}" "$((MASTER_PORT_BASE + 1))"

echo "2scene and 4scene focus no-head smoke tests passed."
