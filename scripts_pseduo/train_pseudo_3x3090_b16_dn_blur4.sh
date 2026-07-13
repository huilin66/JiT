#!/usr/bin/env bash
set -euo pipefail

# 3x3090 experiment:
# Train JiT-B/16 dn_blur_4scene with the focus-2scene B16 winning base
# hyperparameters on the pseudo-augmented RainDrop_Train.
#
# Prior timing: 3x3090 B16 50 epochs ~= 1.5h. Default is a full 600 epoch run.

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-30620}
ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

CKPT=${CKPT:-ckpt/jit-b-16}
OUTPUT_DIR=${OUTPUT_DIR:-run/train_pseudo/b16_dn_blur_4scene_focus_hparams_no_head_3x3090/16}
SCENE_DN_BLUR_4_PATH=${SCENE_DN_BLUR_4_PATH:-${DATA_PATH}/Drop_dn_blur_4scene_test_pseudo.json}

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
check_file "SCENE_DN_BLUR_4_PATH" "${SCENE_DN_BLUR_4_PATH}"
check_ckpt "CKPT" "${CKPT}"
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

echo "============================================================"
echo "[Pseudo 3x3090] B16 dn_blur_4scene, focus-B16 hparams"
echo "DATA_PATH=${DATA_PATH}"
echo "CKPT=${CKPT}"
echo "SCENE_DN_BLUR_4_PATH=${SCENE_DN_BLUR_4_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "GPUS=${GPUS}, nproc=${NPROC}, batch=${BATCH_SIZE}, epochs=${EPOCHS}, lr=${LR}"
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
  --scene_train_path "${SCENE_DN_BLUR_4_PATH}" \
  --scene_val_path "${SCENE_DN_BLUR_4_PATH}" \
  "${online_eval_args[@]}"

echo "Pseudo B16 dn_blur_4scene training finished."
