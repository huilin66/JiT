#!/usr/bin/env bash
set -euo pipefail

# Train JiT-L/16 and JiT-H/16 focus-2scene no-head runs on 2 x RTX 6000 48GB.
# Use run_exp lines at the bottom to control which experiments run.
#
# Example:
# DATA_PATH=/scrinvme/huilin/tp/eccv_dn/RainDrop_Train \
# CKPT_L=/scrinvme/huilin/tp/eccv_dn/jit-l-16 \
# CKPT_H=/scrinvme/huilin/tp/eccv_dn/jit-h-16 \
# bash scripts/train_focus_2scene_jit_lh_2xrtx6000_48g.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0}
NPROC=${NPROC:-1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30320}

DATA_PATH=${DATA_PATH:-/data/huilin/scrinvme/huilin/tp/eccv_dn/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

CKPT_L=${CKPT_L:-ckpt/jit-l-16}
CKPT_H=${CKPT_H:-ckpt/jit-h-16}
OUT_ROOT=${OUT_ROOT:-run/train/focus_2scene_h_no_head_1xA100_48g}

IMG_SIZE=${IMG_SIZE:-256}
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

BATCH_SIZE_L=${BATCH_SIZE_L:-8}
BATCH_SIZE_H=${BATCH_SIZE_H:-16}
LR_L=${LR_L:-3e-5}
LR_H=${LR_H:-2e-5}

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

run_exp() {
  local exp_name="$1"
  local model="$2"
  local ckpt="$3"
  local batch_size="$4"
  local lr="$5"
  local port="$6"
  local output_dir="${OUT_ROOT}/${exp_name}/16"

  echo "============================================================"
  echo "[1xA100 focus 2scene] ${exp_name}"
  echo "model=${model}"
  echo "ckpt=${ckpt}"
  echo "GPUS=${GPUS}, nproc=${NPROC}, batch=${batch_size}, lr=${lr}"
  echo "scene_path=${SCENE_FOCUS_2_PATH}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    main_jit.py \
    --model "${model}" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${batch_size}" \
    --lr "${lr}" \
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
    --resume "${ckpt}" \
    --resume_optimizer "${RESUME_OPTIMIZER}" \
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}" \
    "${online_eval_args[@]}"
}

run_exp "jit_h16_focus_2scene_no_head" "JiT-H/16" "${CKPT_H}" "${BATCH_SIZE_H}" "${LR_H}" "$((MASTER_PORT_BASE + 1))"

echo "JiT/H focus_2scene 1xA100 training finished."
