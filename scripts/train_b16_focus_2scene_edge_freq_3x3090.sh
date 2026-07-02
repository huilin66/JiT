#!/usr/bin/env bash
set -euo pipefail

# JiT-B/16 no-head focus 2-scene training on 3 x RTX 3090.
# No extra data augmentation; edge/frequency losses are enabled.
#
# Labels:
#   Drop_focus_2scene.json
#   0=background-focus, 1=raindrop-focus
#
# Example:
# DATA_ROOT=/root/huilin/data/eccv_dn \
# CKPT=ckpt/jit-b-16 \
# bash scripts/train_b16_focus_2scene_edge_freq_3x3090.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-30060}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train/ablation_b16_focus_2scene_edge_freq_no_head_3x3090}

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

LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.03}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.01}

OUTPUT_DIR=${OUTPUT_DIR:-${OUT_ROOT}/b16_focus_2scene_edge_freq_no_head/16}

if [[ ! -f "${SCENE_FOCUS_2_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_FOCUS_2_PATH}" >&2
  exit 2
fi

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

echo "============================================================"
echo "[Focus 2scene edge/freq] b16_focus_2scene_edge_freq_no_head"
echo "GPUS=${GPUS}, nproc=${NPROC}, batch=${BATCH_SIZE}, lr=${LR}"
echo "scene_path=${SCENE_FOCUS_2_PATH}"
echo "loss_edge_weight=${LOSS_EDGE_WEIGHT}"
echo "loss_freq_weight=${LOSS_FREQ_WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
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
  --output_dir "${OUTPUT_DIR}" \
  --use_bg_subnet 0 \
  --use_scene_dataset 1 \
  --data_path "${DATA_PATH}" \
  --val_data_path "${VAL_DATA_PATH}" \
  --resume "${CKPT}" \
  --resume_optimizer "${RESUME_OPTIMIZER}" \
  --scene_train_path "${SCENE_FOCUS_2_PATH}" \
  --scene_val_path "${SCENE_FOCUS_2_PATH}" \
  --loss_edge_weight "${LOSS_EDGE_WEIGHT}" \
  --loss_freq_weight "${LOSS_FREQ_WEIGHT}" \
  "${online_eval_args[@]}"

echo "focus_2scene edge/freq 3x3090 training finished."
