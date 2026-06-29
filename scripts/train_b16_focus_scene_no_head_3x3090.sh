#!/usr/bin/env bash
set -euo pipefail

# Train focus/scene no-head runs on 3 GPUs.
#
# Override paths from the command line, for example:
# DATA_PATH=/data/RainDrop_Train CKPT=/data/jit-b-16 bash scripts/train_b16_focus_scene_no_head_3x3090.sh

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29620}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}

DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train/ablation_b16_focus_scene_no_head_3x3090}

SCENE_2_PATH=${SCENE_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}
SCENE_4_PATH=${SCENE_4_PATH:-${DATA_PATH}/Drop_focus_4scene.json}
SCENE_DN_2_PATH=${SCENE_DN_2_PATH:-${DATA_PATH}/Drop_dn_2scene.json}

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
  local output_dir="${OUT_ROOT}/b16_focus_${scene_name}_no_head/16"

  echo "============================================================"
  echo "[Focus ${scene_name}] b16_focus_${scene_name}_no_head"
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

run_exp "2scene" "${SCENE_2_PATH}" "$((MASTER_PORT_BASE + 0))"
run_exp "4scene" "${SCENE_4_PATH}" "$((MASTER_PORT_BASE + 1))"
run_exp "dn_2scene" "${SCENE_DN_2_PATH}" "$((MASTER_PORT_BASE + 2))"

echo "2scene, 4scene, and dn_2scene no-head training finished."
