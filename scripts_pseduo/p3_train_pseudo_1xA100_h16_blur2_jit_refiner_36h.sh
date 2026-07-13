#!/usr/bin/env bash
set -euo pipefail

# 1xA100 experiment:
# Stage 1: lightly fine-tune JiT-H/16 from the best focus-2scene H16-refiner
#          checkpoint to blur_2scene on pseudo-augmented data.
# Stage 2: spend most of the budget training a frozen-JiT MSDT refiner from the
#          Stage-1 checkpoint.
#
# Timing target from prior runs:
#   H16 JIT:     50 epochs ~= 6.5h
#   H16 refiner: 50 epochs ~= 3.5h  -> 330 epochs ~= 23.1h
# Total default target: about 30h.

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0}
NPROC=${NPROC:-1}
MASTER_PORT_JIT=${MASTER_PORT_JIT:-30720}
MASTER_PORT_REFINER=${MASTER_PORT_REFINER:-30721}
RUN_STAGES=${RUN_STAGES:-"JIT REFINER"}

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/data/huilin/scrinvme/huilin/tp/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

CKPT_H_FOCUS=${CKPT_H_FOCUS:-run/train/focus_2scene_msdt_refiner_h_1xA100_48g/h16_refiner_higher_than_c1/16}
JIT_OUTPUT_DIR=${JIT_OUTPUT_DIR:-run/train_pseudo/h16_blur_2scene_from_refiner_higher_jit_ft_50ep_1xA100/16}
REFINER_OUTPUT_DIR=${REFINER_OUTPUT_DIR:-run/train_pseudo/h16_blur_2scene_from_refiner_higher_refiner_330ep_1xA100/16}
CKPT_H_BLUR=${CKPT_H_BLUR:-${JIT_OUTPUT_DIR}}
SCENE_BLUR_2_PATH=${SCENE_BLUR_2_PATH:-${DATA_PATH}/Drop_blur_2scene_test_pseudo.json}

IMG_SIZE=${IMG_SIZE:-256}
EPOCHS_JIT=${EPOCHS_JIT:-50}
EPOCHS_REFINER=${EPOCHS_REFINER:-330}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}

BATCH_SIZE_JIT=${BATCH_SIZE_JIT:-16}
LR_JIT=${LR_JIT:-2e-5}
RESUME_STATE_KEY_JIT=${RESUME_STATE_KEY_JIT:-model_ema1}

BATCH_SIZE_REFINER=${BATCH_SIZE_REFINER:-32}
LR_REFINER=${LR_REFINER:-2.5e-4}
RESUME_STATE_KEY_REFINER=${RESUME_STATE_KEY_REFINER:-model_ema2}
REFINER_BASE_DIM=${REFINER_BASE_DIM:-32}
REFINER_NUM_BLOCKS=${REFINER_NUM_BLOCKS:-2}
REFINER_USE_FREQUENCY=${REFINER_USE_FREQUENCY:-1}
REFINER_MAX_RESIDUAL=${REFINER_MAX_RESIDUAL:-0.35}
LOSS_EDGE_WEIGHT_REFINER=${LOSS_EDGE_WEIGHT_REFINER:-0.10}
LOSS_FREQ_WEIGHT_REFINER=${LOSS_FREQ_WEIGHT_REFINER:-0.10}

stage_enabled() {
  local wanted="$1"
  for stage in ${RUN_STAGES}; do
    if [[ "${stage}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

missing_paths=()
check_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    missing_paths+=("${label}: ${path}")
  fi
}
check_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    missing_paths+=("${label}: ${path}")
  fi
}
check_ckpt() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" && ! -f "${path}/checkpoint-last.pth" ]]; then
    missing_paths+=("${label}: ${path} or ${path}/checkpoint-last.pth")
  fi
}

check_dir "ROOT_DIR" "${ROOT_DIR}"
check_file "main_jit.py" "${ROOT_DIR}/main_jit.py"
check_dir "DATA_PATH" "${DATA_PATH}"
check_dir "DATA_PATH/Drop" "${DATA_PATH}/Drop"
check_dir "DATA_PATH/Clear" "${DATA_PATH}/Clear"
check_file "SCENE_BLUR_2_PATH" "${SCENE_BLUR_2_PATH}"
if stage_enabled "JIT"; then
  check_ckpt "CKPT_H_FOCUS" "${CKPT_H_FOCUS}"
else
  check_ckpt "CKPT_H_BLUR" "${CKPT_H_BLUR}"
fi
if [[ "${#missing_paths[@]}" -gt 0 ]]; then
  echo "Missing required paths:" >&2
  printf '  - %s\n' "${missing_paths[@]}" >&2
  exit 2
fi

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

if stage_enabled "JIT"; then
  echo "============================================================"
  echo "[Pseudo 1xA100] Stage 1/2: H16 blur2 JIT finetune"
  echo "DATA_PATH=${DATA_PATH}"
  echo "CKPT_H_FOCUS=${CKPT_H_FOCUS}"
  echo "SCENE_BLUR_2_PATH=${SCENE_BLUR_2_PATH}"
  echo "JIT_OUTPUT_DIR=${JIT_OUTPUT_DIR}"
  echo "GPUS=${GPUS}, batch=${BATCH_SIZE_JIT}, epochs=${EPOCHS_JIT}, lr=${LR_JIT}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT_JIT}" \
    main_jit.py \
    --model "JiT-H/16" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE_JIT}" \
    --lr "${LR_JIT}" \
    --lr_schedule cosine \
    --epochs "${EPOCHS_JIT}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --eval_epoch "${EVAL_EPOCH}" \
    --eval_num_images "${EVAL_NUM_IMAGES}" \
    --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
    --cfg 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --save_last_freq "${SAVE_LAST_FREQ}" \
    --log_freq "${LOG_FREQ}" \
    --output_dir "${JIT_OUTPUT_DIR}" \
    --use_bg_subnet 0 \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT_H_FOCUS}" \
    --resume_state_key "${RESUME_STATE_KEY_JIT}" \
    --resume_optimizer 0 \
    --scene_train_path "${SCENE_BLUR_2_PATH}" \
    --scene_val_path "${SCENE_BLUR_2_PATH}" \
    "${online_eval_args[@]}"
fi

if stage_enabled "REFINER"; then
  echo "============================================================"
  echo "[Pseudo 1xA100] Stage 2/2: H16 blur2 refiner"
  echo "CKPT_H_BLUR=${CKPT_H_BLUR}"
  echo "SCENE_BLUR_2_PATH=${SCENE_BLUR_2_PATH}"
  echo "REFINER_OUTPUT_DIR=${REFINER_OUTPUT_DIR}"
  echo "GPUS=${GPUS}, batch=${BATCH_SIZE_REFINER}, epochs=${EPOCHS_REFINER}, lr=${LR_REFINER}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT_REFINER}" \
    main_jit.py \
    --model "JiT-H/16" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE_REFINER}" \
    --lr "${LR_REFINER}" \
    --lr_schedule cosine \
    --epochs "${EPOCHS_REFINER}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --eval_epoch "${EVAL_EPOCH}" \
    --eval_num_images "${EVAL_NUM_IMAGES}" \
    --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
    --cfg 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --save_last_freq "${SAVE_LAST_FREQ}" \
    --log_freq "${LOG_FREQ}" \
    --output_dir "${REFINER_OUTPUT_DIR}" \
    --use_bg_subnet 0 \
    --use_detail_refiner 1 \
    --freeze_jit 1 \
    --refiner_base_dim "${REFINER_BASE_DIM}" \
    --refiner_num_blocks "${REFINER_NUM_BLOCKS}" \
    --refiner_use_frequency "${REFINER_USE_FREQUENCY}" \
    --refiner_max_residual "${REFINER_MAX_RESIDUAL}" \
    --loss_edge_weight "${LOSS_EDGE_WEIGHT_REFINER}" \
    --loss_freq_weight "${LOSS_FREQ_WEIGHT_REFINER}" \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT_H_BLUR}" \
    --resume_state_key "${RESUME_STATE_KEY_REFINER}" \
    --resume_optimizer 0 \
    --scene_train_path "${SCENE_BLUR_2_PATH}" \
    --scene_val_path "${SCENE_BLUR_2_PATH}" \
    "${online_eval_args[@]}"
fi

echo "Pseudo H16 blur2 JIT/refiner requested stages finished: ${RUN_STAGES}"
