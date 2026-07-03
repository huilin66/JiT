#!/usr/bin/env bash
set -euo pipefail

# Smoke test JiT-L/16 and JiT-H/16 focus-2scene training on 3 x RTX 3090.
# Each model runs two epochs with one train iteration per epoch.
# Epoch 1 activates LPIPS and tests its backward-memory peak.
#
# Run from repo root:
#   bash scripts/smoke_test_focus_2scene_jit_lh_3x3090.sh
#
# Optional:
#   RUN_MODELS="L" bash scripts/smoke_test_focus_2scene_jit_lh_3x3090.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30220}
RUN_MODELS=${RUN_MODELS:-"L H"}

CKPT_L=${CKPT_L:-ckpt/jit-l-16}
CKPT_H=${CKPT_H:-ckpt/jit-h-16}
OUT_ROOT=${OUT_ROOT:-run/train_smoke/jit_lh_focus_2scene_no_head_3x3090}

IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE_L=${BATCH_SIZE_L:-4}
BATCH_SIZE_H=${BATCH_SIZE_H:-2}
LR_L=${LR_L:-3e-5}
LR_H=${LR_H:-2e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}

print_gpu_snapshot() {
  local title="$1"
  echo "---------------- GPU memory: ${title} ----------------"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  else
    echo "nvidia-smi not found; skip GPU memory snapshot."
  fi
}

start_gpu_memory_log() {
  local log_path="$1"
  mkdir -p "$(dirname "${log_path}")"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv -l 1 > "${log_path}" &
    MEM_LOG_PID=$!
  else
    MEM_LOG_PID=""
  fi
}

stop_gpu_memory_log() {
  local log_path="$1"
  if [[ -n "${MEM_LOG_PID:-}" ]]; then
    kill "${MEM_LOG_PID}" >/dev/null 2>&1 || true
    wait "${MEM_LOG_PID}" >/dev/null 2>&1 || true
    MEM_LOG_PID=""
    echo "GPU memory log: ${log_path}"
  fi
}

run_smoke() {
  local tag="$1"
  local model="$2"
  local ckpt="$3"
  local batch_size="$4"
  local lr="$5"
  local port="$6"
  local output_dir="${OUT_ROOT}/jit_${tag}_focus_2scene_no_head/16"
  local mem_log="${output_dir}/gpu_memory_${tag}.csv"

  echo "============================================================"
  echo "[Smoke JiT-${tag}] focus 2scene no-head"
  echo "model=${model}"
  echo "ckpt=${ckpt}"
  echo "batch_size=${batch_size}, lr=${lr}"
  echo "Epoch 0 tests normal loss; epoch 1 tests LPIPS backward."
  echo "scene_path=${SCENE_FOCUS_2_PATH}"
  echo "data_path=${DATA_PATH}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  print_gpu_snapshot "before ${tag}"
  start_gpu_memory_log "${mem_log}"

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
    --resume "${ckpt}" \
    --resume_optimizer 0 \
    --online_eval \
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}"

  stop_gpu_memory_log "${mem_log}"
  print_gpu_snapshot "after ${tag}"

  if [[ ! -f "${output_dir}/checkpoint-last.pth" ]]; then
    echo "Smoke test failed: checkpoint-last.pth was not created for JiT-${tag}" >&2
    exit 1
  fi
}

for model_tag in ${RUN_MODELS}; do
  case "${model_tag}" in
    L|l)
      run_smoke "l16" "JiT-L/16" "${CKPT_L}" "${BATCH_SIZE_L}" "${LR_L}" "$((MASTER_PORT_BASE + 0))"
      ;;
    H|h)
      run_smoke "h16" "JiT-H/16" "${CKPT_H}" "${BATCH_SIZE_H}" "${LR_H}" "$((MASTER_PORT_BASE + 1))"
      ;;
    *)
      echo "Unknown model tag: ${model_tag}. Use RUN_MODELS=\"L H\"." >&2
      exit 2
      ;;
  esac
done

echo "JiT-L/H focus-2scene smoke tests passed, including LPIPS backward."
