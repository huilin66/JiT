#!/usr/bin/env bash
set -euo pipefail

# JiT-B/16 no-head focus 2-scene training on 3 x RTX 3090.
# Based on train_b16_dn_blur_4scene_aug_1x5090.sh.
#
# Labels:
#   Drop_focus_2scene.json
#   0=background-focus, 1=raindrop-focus
#
# Example:
# DATA_ROOT=/root/huilin/data/eccv_dn \
# CKPT=ckpt/jit-b-16 \
# bash scripts/train_b16_focus_2scene_aug_3x3090.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30050}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train/ablation_b16_focus_2scene_aug_no_head_3x3090}

SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

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

MORE_AUG=${MORE_AUG:-1}
LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.03}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.01}

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

run_exp() {
  local scene_name="$1"
  local scene_path="$2"
  local port="$3"
  local edge_weight="$4"
  local freq_weight="$5"
  local output_dir="${OUT_ROOT}/b16_focus_${scene_name}_aug_no_head/16"

  if [[ ! -f "${scene_path}" ]]; then
    echo "Missing scene label file: ${scene_path}" >&2
    exit 2
  fi

  echo "============================================================"
  echo "[Focus ${scene_name}] b16_focus_${scene_name}_aug_no_head"
  echo "GPUS=${GPUS}, nproc=${NPROC}, batch=${BATCH_SIZE}, lr=${LR}"
  echo "use_scene_dataset=1"
  echo "use_bg_subnet=0"
  echo "scene_path=${scene_path}"
  echo "more_aug=${MORE_AUG}"
  echo "loss_edge_weight=${edge_weight}"
  echo "loss_freq_weight=${freq_weight}"
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
    --more_aug "${MORE_AUG}" \
    --loss_edge_weight "${edge_weight}" \
    --loss_freq_weight "${freq_weight}" \
    "${online_eval_args[@]}"
}

run_exp "2scene" "${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}" "$((MASTER_PORT_BASE + 0))" 0.0 0.0
run_exp "2scene_edge_freq" "${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}" "$((MASTER_PORT_BASE + 1))" "${LOSS_EDGE_WEIGHT}" "${LOSS_FREQ_WEIGHT}"

echo "focus_2scene augmented 3x3090 training finished."
