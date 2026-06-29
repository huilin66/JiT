#!/usr/bin/env bash
set -euo pipefail

# Train JiT-B/16 no-head runs with manually checked blur scene labels.
#
# Labels are generated from C:/Users/USER/Downloads/sample_check.csv by
# tools/data_tools.py manual-blur-scenes and live under RainDrop_Train:
#   Drop_blur_2scene.json       0=not_blur, 1=blur
#   Drop_dn_blur_4scene.json    0=night_not_blur, 1=night_blur, 2=day_not_blur, 3=day_blur
#
# Example:
# DATA_ROOT=/data/eccv_dn CKPT=/data/jit-b-16 bash scripts/train_b16_manual_blur_scene_no_head_3x3090.sh

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29820}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train/ablation_b16_manual_blur_scene_no_head_3x3090}

SCENE_BLUR_2_PATH=${SCENE_BLUR_2_PATH:-${DATA_PATH}/Drop_blur_2scene.json}
SCENE_DN_BLUR_4_PATH=${SCENE_DN_BLUR_4_PATH:-${DATA_PATH}/Drop_dn_blur_4scene.json}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-5e-5}
EPOCHS=${EPOCHS:-600}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}
RESUME_OPTIMIZER=${RESUME_OPTIMIZER:-0}

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

run_exp() {
  local scene_name="$1"
  local scene_path="$2"
  local port="$3"
  local output_dir="${OUT_ROOT}/b16_${scene_name}_no_head/16"

  if [[ ! -f "${scene_path}" ]]; then
    echo "Missing scene label file: ${scene_path}" >&2
    exit 2
  fi

  echo "============================================================"
  echo "[Manual blur ${scene_name}] b16_${scene_name}_no_head"
  echo "use_scene_dataset=1"
  echo "use_bg_subnet=0"
  echo "scene_path=${scene_path}"
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
    --epochs "${EPOCHS}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --eval_epoch "${EVAL_EPOCH}" \
    --eval_num_images "${EVAL_NUM_IMAGES}" \
    --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
    --cfg 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --save_last_freq "${SAVE_LAST_FREQ}" \
    --log_freq "${LOG_FREQ}" \
    --output_dir "${output_dir}" \
    --use_bg_subnet 0 \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT}" \
    --resume_optimizer "${RESUME_OPTIMIZER}" \
    --scene_train_path "${scene_path}" \
    --scene_val_path "${scene_path}" \
    "${online_eval_args[@]}"
}

run_exp "blur_2scene" "${SCENE_BLUR_2_PATH:-${DATA_PATH}/Drop_blur_2scene.json}" "$((MASTER_PORT_BASE + 0))"
run_exp "dn_blur_4scene" "${SCENE_DN_BLUR_4_PATH:-${DATA_PATH}/Drop_dn_blur_4scene.json}" "$((MASTER_PORT_BASE + 1))"

echo "Manual blur scene training finished."
