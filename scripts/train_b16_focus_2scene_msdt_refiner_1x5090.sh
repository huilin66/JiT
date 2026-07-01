#!/usr/bin/env bash
set -euo pipefail

# Stage-2 refiner training on one RTX 5090:
# load an existing focus-2scene JiT checkpoint, freeze JiT, and train only the
# MSDT-style detail refiner.
#
# Example:
# DATA_PATH=D:/zhl/data/eccv_dn/RainDrop_Train \
# CKPT=run/ablation_b16_3x3090/b16_focus_2scene_no_head \
# bash scripts/train_b16_focus_2scene_msdt_refiner_1x5090.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPU=${GPU:-0}

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

CKPT=${CKPT:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
OUTPUT_DIR=${OUTPUT_DIR:-run/train/b16_focus_2scene_msdt_refiner_1x5090/16}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-300}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
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
REFINER_MAX_RESIDUAL=${REFINER_MAX_RESIDUAL:-0.25}
LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.05}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.05}

if [[ ! -f "${CKPT}" && ! -f "${CKPT}/checkpoint-last.pth" ]]; then
  echo "Cannot find checkpoint file: ${CKPT} (or ${CKPT}/checkpoint-last.pth)" >&2
  exit 2
fi

if [[ ! -f "${SCENE_FOCUS_2_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_FOCUS_2_PATH}" >&2
  exit 2
fi

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

echo "============================================================"
echo "[MSDT refiner] focus 2scene on 1x5090"
echo "GPU=${GPU}, batch=${BATCH_SIZE}, lr=${LR}, epochs=${EPOCHS}"
echo "ckpt=${CKPT}"
echo "resume_state_key=${RESUME_STATE_KEY}"
echo "scene_path=${SCENE_FOCUS_2_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "refiner_base_dim=${REFINER_BASE_DIM}, blocks=${REFINER_NUM_BLOCKS}"
echo "loss_edge_weight=${LOSS_EDGE_WEIGHT}, loss_freq_weight=${LOSS_FREQ_WEIGHT}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

CUDA_VISIBLE_DEVICES="${GPU}" python main_jit.py \
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
  --save_last_freq "${SAVE_LAST_FREQ}" \
  --log_freq "${LOG_FREQ}" \
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
  --use_scene_dataset 1 \
  --data_path "${DATA_PATH}" \
  --val_data_path "${VAL_DATA_PATH}" \
  --resume "${CKPT}" \
  --resume_state_key "${RESUME_STATE_KEY}" \
  --resume_optimizer 0 \
  --scene_train_path "${SCENE_FOCUS_2_PATH}" \
  --scene_val_path "${SCENE_FOCUS_2_PATH}" \
  "${online_eval_args[@]}"

echo "focus_2scene MSDT refiner 1x5090 training finished."
