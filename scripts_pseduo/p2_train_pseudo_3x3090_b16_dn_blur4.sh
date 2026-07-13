#!/usr/bin/env bash
set -euo pipefail

# P2 3x3090 experiment:
#   Stage 1: train JiT-B/16 on pseudo-augmented dn_blur_4scene.
#   Stage 2: train the MSDT detail refiner from Stage-1 checkpoint-last.
#
# Best-history anchors from submission_history_jit.csv:
#   JiT-B base: b16_focus_2scene_no_head checkpoint-last, score 33.9609.
#     Training style: batch 16, lr 5e-5, 600 epochs, warmup 5,
#     no LPIPS/edge/frequency extra losses.
#   JiT-B refiner: plan_c12 C1 checkpoint-last, score 34.4282.
#     Training style: single-card batch 32, lr 2.5e-4, 300 epochs, warmup 5,
#     resume_state_key model_ema2, max_residual 0.35, edge/freq 0.10.
#     On 3x3090, batch_size is per GPU; default to per-GPU 8 as in the
#     existing 3x3090 B16 refiner script instead of accidentally using
#     global batch 96.

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_JIT=${MASTER_PORT_JIT:-30620}
MASTER_PORT_REFINER=${MASTER_PORT_REFINER:-30621}
ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
RUN_STAGES=${RUN_STAGES:-"JIT REFINER"}

INIT_CKPT=${INIT_CKPT:-ckpt/jit-b-16}
SCENE_DN_BLUR_4_PATH=${SCENE_DN_BLUR_4_PATH:-${DATA_PATH}/Drop_dn_blur_4scene_test_pseudo.json}
OUT_ROOT=${OUT_ROOT:-run/train_pseudo}
JIT_OUTPUT_DIR=${JIT_OUTPUT_DIR:-${OUT_ROOT}/p2_b16_dn_blur_4scene_jit_best_base_hparams_3x3090/16}
REFINER_OUTPUT_DIR=${REFINER_OUTPUT_DIR:-${OUT_ROOT}/p2_b16_dn_blur_4scene_refiner_c1_from_jit_last_3x3090/16}
REFINER_CKPT=${REFINER_CKPT:-${JIT_OUTPUT_DIR}/checkpoint-last.pth}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}

JIT_BATCH_SIZE=${JIT_BATCH_SIZE:-16}
JIT_LR=${JIT_LR:-5e-5}
JIT_EPOCHS=${JIT_EPOCHS:-600}
JIT_WARMUP_EPOCHS=${JIT_WARMUP_EPOCHS:-5}
JIT_EVAL_EPOCH=${JIT_EVAL_EPOCH:-5}
JIT_EVAL_NUM_IMAGES=${JIT_EVAL_NUM_IMAGES:-100}
JIT_RESUME_OPTIMIZER=${JIT_RESUME_OPTIMIZER:-0}

REFINER_BATCH_SIZE=${REFINER_BATCH_SIZE:-8}
REFINER_LR=${REFINER_LR:-2.5e-4}
REFINER_EPOCHS=${REFINER_EPOCHS:-300}
REFINER_WARMUP_EPOCHS=${REFINER_WARMUP_EPOCHS:-5}
REFINER_EVAL_EPOCH=${REFINER_EVAL_EPOCH:-5}
REFINER_EVAL_NUM_IMAGES=${REFINER_EVAL_NUM_IMAGES:-100}
REFINER_RESUME_STATE_KEY=${REFINER_RESUME_STATE_KEY:-model_ema2}
REFINER_BASE_DIM=${REFINER_BASE_DIM:-32}
REFINER_NUM_BLOCKS=${REFINER_NUM_BLOCKS:-2}
REFINER_USE_FREQUENCY=${REFINER_USE_FREQUENCY:-1}
REFINER_MAX_RESIDUAL=${REFINER_MAX_RESIDUAL:-0.35}
LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.10}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.10}

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
has_stage() {
  local stage="$1"
  [[ " ${RUN_STAGES} " == *" ${stage} "* ]]
}

check_dir "ROOT_DIR" "${ROOT_DIR}"
check_file "main_jit.py" "${ROOT_DIR}/main_jit.py"
check_dir "DATA_PATH" "${DATA_PATH}"
check_dir "DATA_PATH/Drop" "${DATA_PATH}/Drop"
check_dir "DATA_PATH/Clear" "${DATA_PATH}/Clear"
check_file "SCENE_DN_BLUR_4_PATH" "${SCENE_DN_BLUR_4_PATH}"
if has_stage "JIT"; then
  check_ckpt "INIT_CKPT" "${INIT_CKPT}"
fi
if has_stage "REFINER" && ! has_stage "JIT"; then
  check_ckpt "REFINER_CKPT" "${REFINER_CKPT}"
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

echo "============================================================"
echo "[P2 pseudo 3x3090] B16 dn_blur_4scene JIT + refiner"
echo "RUN_STAGES=${RUN_STAGES}"
echo "DATA_PATH=${DATA_PATH}"
echo "SCENE_DN_BLUR_4_PATH=${SCENE_DN_BLUR_4_PATH}"
echo "GPUS=${GPUS}, nproc=${NPROC}"
echo "JIT: init=${INIT_CKPT}, output=${JIT_OUTPUT_DIR}, batch=${JIT_BATCH_SIZE}, lr=${JIT_LR}, epochs=${JIT_EPOCHS}"
echo "REFINER: ckpt=${REFINER_CKPT}, output=${REFINER_OUTPUT_DIR}, batch=${REFINER_BATCH_SIZE}, lr=${REFINER_LR}, epochs=${REFINER_EPOCHS}"
echo "REFINER: state=${REFINER_RESUME_STATE_KEY}, max_residual=${REFINER_MAX_RESIDUAL}, edge=${LOSS_EDGE_WEIGHT}, freq=${LOSS_FREQ_WEIGHT}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

if has_stage "JIT"; then
  echo "============================================================"
  echo "[P2 Stage 1] Train JiT-B/16 dn_blur_4scene"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT_JIT}" \
    main_jit.py \
    --model "${MODEL}" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${JIT_BATCH_SIZE}" \
    --lr "${JIT_LR}" \
    --lr_schedule cosine \
    --epochs "${JIT_EPOCHS}" \
    --warmup_epochs "${JIT_WARMUP_EPOCHS}" \
    --eval_epoch "${JIT_EVAL_EPOCH}" \
    --eval_num_images "${JIT_EVAL_NUM_IMAGES}" \
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
    --resume "${INIT_CKPT}" \
    --resume_optimizer "${JIT_RESUME_OPTIMIZER}" \
    --scene_train_path "${SCENE_DN_BLUR_4_PATH}" \
    --scene_val_path "${SCENE_DN_BLUR_4_PATH}" \
    "${online_eval_args[@]}"
fi

if has_stage "REFINER"; then
  if [[ ! -f "${REFINER_CKPT}" && ! -f "${REFINER_CKPT}/checkpoint-last.pth" ]]; then
    echo "Cannot find refiner base checkpoint after Stage 1: ${REFINER_CKPT} (or ${REFINER_CKPT}/checkpoint-last.pth)" >&2
    exit 2
  fi

  echo "============================================================"
  echo "[P2 Stage 2] Train MSDT refiner from JiT-B last"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT_REFINER}" \
    main_jit.py \
    --model "${MODEL}" \
    --img_size "${IMG_SIZE}" \
    --batch_size "${REFINER_BATCH_SIZE}" \
    --lr "${REFINER_LR}" \
    --lr_schedule cosine \
    --epochs "${REFINER_EPOCHS}" \
    --warmup_epochs "${REFINER_WARMUP_EPOCHS}" \
    --eval_epoch "${REFINER_EVAL_EPOCH}" \
    --eval_num_images "${REFINER_EVAL_NUM_IMAGES}" \
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
    --loss_edge_weight "${LOSS_EDGE_WEIGHT}" \
    --loss_freq_weight "${LOSS_FREQ_WEIGHT}" \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${REFINER_CKPT}" \
    --resume_state_key "${REFINER_RESUME_STATE_KEY}" \
    --resume_optimizer 0 \
    --scene_train_path "${SCENE_DN_BLUR_4_PATH}" \
    --scene_val_path "${SCENE_DN_BLUR_4_PATH}" \
    "${online_eval_args[@]}"
fi

echo "P2 pseudo B16 dn_blur_4scene JIT/refiner training finished."
