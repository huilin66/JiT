#!/usr/bin/env bash
set -euo pipefail

# Two-epoch smoke test for JiT-B/L MSDT-refiner post-training on 3 x RTX 3090.
# Each epoch runs one train step. Epoch 1 activates LPIPS backward, so this is
# useful for checking GPU memory before launching the full training script.
#
# Examples:
#   RUN_EXPS="L_C1" BATCH_SIZE_L=1 bash scripts/smoke_test_focus_2scene_msdt_refiner_bl_3x3090.sh
#   RUN_EXPS="B_HIGHER" BATCH_SIZE_B=8 bash scripts/smoke_test_focus_2scene_msdt_refiner_bl_3x3090.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30820}
RUN_EXPS=${RUN_EXPS:-"B_HIGHER L_C1 L_HIGHER"}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

CKPT_B=${CKPT_B:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
CKPT_L=${CKPT_L:-run/lh16/l16_rtx6000}
OUT_ROOT=${OUT_ROOT:-run/train_smoke/focus_2scene_msdt_refiner_bl_3x3090}

IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE_B=${BATCH_SIZE_B:-4}
BATCH_SIZE_L=${BATCH_SIZE_L:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
RESUME_STATE_KEY=${RESUME_STATE_KEY:-model_ema2}

REFINER_BASE_DIM=${REFINER_BASE_DIM:-32}
REFINER_NUM_BLOCKS=${REFINER_NUM_BLOCKS:-2}
REFINER_USE_FREQUENCY=${REFINER_USE_FREQUENCY:-1}

LR_C1=${LR_C1:-2.5e-4}
MAX_RESIDUAL_C1=${MAX_RESIDUAL_C1:-0.35}
EDGE_WEIGHT_C1=${EDGE_WEIGHT_C1:-0.10}
FREQ_WEIGHT_C1=${FREQ_WEIGHT_C1:-0.10}

LR_HIGHER=${LR_HIGHER:-3e-4}
MAX_RESIDUAL_HIGHER=${MAX_RESIDUAL_HIGHER:-0.40}
EDGE_WEIGHT_HIGHER=${EDGE_WEIGHT_HIGHER:-0.12}
FREQ_WEIGHT_HIGHER=${FREQ_WEIGHT_HIGHER:-0.12}

check_required_path() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" && ! -f "${path}/checkpoint-last.pth" ]]; then
    echo "Cannot find ${label} checkpoint: ${path} (or ${path}/checkpoint-last.pth)" >&2
    exit 2
  fi
}

check_required_path "JiT-B" "${CKPT_B}"
check_required_path "JiT-L" "${CKPT_L}"

if [[ ! -f "${SCENE_FOCUS_2_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_FOCUS_2_PATH}" >&2
  exit 2
fi

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
  local max_residual="$6"
  local edge_weight="$7"
  local freq_weight="$8"
  local port="$9"
  local output_dir="${OUT_ROOT}/${tag}/16"
  local mem_log="${output_dir}/gpu_memory_${tag}.csv"

  echo "============================================================"
  echo "[Smoke MSDT refiner 3x3090] ${tag}"
  echo "model=${model}"
  echo "ckpt=${ckpt}"
  echo "resume_state_key=${RESUME_STATE_KEY}"
  echo "GPUS=${GPUS}, nproc=${NPROC}, batch_per_gpu=${batch_size}, global_batch=$((batch_size * NPROC))"
  echo "lr=${lr}, max_residual=${max_residual}, edge=${edge_weight}, freq=${freq_weight}"
  echo "Epoch 0 tests normal loss; epoch 1 tests LPIPS backward."
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
    --use_detail_refiner 1 \
    --freeze_jit 1 \
    --refiner_base_dim "${REFINER_BASE_DIM}" \
    --refiner_num_blocks "${REFINER_NUM_BLOCKS}" \
    --refiner_use_frequency "${REFINER_USE_FREQUENCY}" \
    --refiner_max_residual "${max_residual}" \
    --loss_edge_weight "${edge_weight}" \
    --loss_freq_weight "${freq_weight}" \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${ckpt}" \
    --resume_state_key "${RESUME_STATE_KEY}" \
    --resume_optimizer 0 \
    --online_eval \
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}"

  stop_gpu_memory_log "${mem_log}"
  print_gpu_snapshot "after ${tag}"

  if [[ ! -f "${output_dir}/checkpoint-last.pth" ]]; then
    echo "Smoke test failed: checkpoint-last.pth was not created for ${tag}" >&2
    exit 1
  fi
}

for exp in ${RUN_EXPS}; do
  case "${exp}" in
    B_HIGHER|b_higher)
      run_smoke \
        "b16_refiner_higher_than_c1" \
        "JiT-B/16" \
        "${CKPT_B}" \
        "${BATCH_SIZE_B}" \
        "${LR_HIGHER}" \
        "${MAX_RESIDUAL_HIGHER}" \
        "${EDGE_WEIGHT_HIGHER}" \
        "${FREQ_WEIGHT_HIGHER}" \
        "$((MASTER_PORT_BASE + 0))"
      ;;
    L_C1|l_c1)
      run_smoke \
        "l16_refiner_c1" \
        "JiT-L/16" \
        "${CKPT_L}" \
        "${BATCH_SIZE_L}" \
        "${LR_C1}" \
        "${MAX_RESIDUAL_C1}" \
        "${EDGE_WEIGHT_C1}" \
        "${FREQ_WEIGHT_C1}" \
        "$((MASTER_PORT_BASE + 1))"
      ;;
    L_HIGHER|l_higher)
      run_smoke \
        "l16_refiner_higher_than_c1" \
        "JiT-L/16" \
        "${CKPT_L}" \
        "${BATCH_SIZE_L}" \
        "${LR_HIGHER}" \
        "${MAX_RESIDUAL_HIGHER}" \
        "${EDGE_WEIGHT_HIGHER}" \
        "${FREQ_WEIGHT_HIGHER}" \
        "$((MASTER_PORT_BASE + 2))"
      ;;
    *)
      echo "Unknown experiment: ${exp}. Use RUN_EXPS=\"B_HIGHER L_C1 L_HIGHER\"." >&2
      exit 2
      ;;
  esac
done

echo "JiT-B/L MSDT-refiner smoke tests passed, including LPIPS backward."
