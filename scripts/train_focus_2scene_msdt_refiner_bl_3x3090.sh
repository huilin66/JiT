#!/usr/bin/env bash
set -euo pipefail

# Stage-2 MSDT-refiner sweep for focus-2scene JiT-B/16 and JiT-L/16 on 3 x RTX 3090.
# JiT is loaded from a trained checkpoint, frozen, and only the detail refiner is trained.
#
# Default experiments:
#   B_HIGHER  - JiT-B/16 with a stronger-than-C1 refiner strategy
#   L_C1      - JiT-L/16 with the C1 refiner strategy
#   L_HIGHER  - JiT-L/16 with the stronger-than-C1 refiner strategy
#
# Examples:
#   RUN_EXPS="B_HIGHER" bash scripts/train_focus_2scene_msdt_refiner_bl_3x3090.sh
#   BATCH_SIZE_L=1 RUN_EXPS="L_C1 L_HIGHER" bash scripts/train_focus_2scene_msdt_refiner_bl_3x3090.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30720}
RUN_EXPS=${RUN_EXPS:-"B_HIGHER L_C1 L_HIGHER"}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

CKPT_B=${CKPT_B:-run/ablation_b16_3x3090/b16_focus_2scene_no_head}
CKPT_L=${CKPT_L:-run/lh16/l16_rtx6000}
OUT_ROOT=${OUT_ROOT:-run/train/focus_2scene_msdt_refiner_bl_3x3090}

IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE_B=${BATCH_SIZE_B:-8}
BATCH_SIZE_L=${BATCH_SIZE_L:-2}
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

# C1: lr=2.5e-4, max_residual=0.35, edge/freq=0.10/0.10.
# HIGHER probes one step above C1 without a large jump.
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

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

run_exp() {
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

  echo "============================================================"
  echo "[MSDT refiner 3x3090] ${tag}"
  echo "model=${model}"
  echo "ckpt=${ckpt}"
  echo "resume_state_key=${RESUME_STATE_KEY}"
  echo "GPUS=${GPUS}, nproc=${NPROC}, batch_per_gpu=${batch_size}, global_batch=$((batch_size * NPROC))"
  echo "lr=${lr}, epochs=${EPOCHS}, eval_epoch=${EVAL_EPOCH}"
  echo "max_residual=${max_residual}, edge=${edge_weight}, freq=${freq_weight}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

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
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}" \
    "${online_eval_args[@]}"
}

for exp in ${RUN_EXPS}; do
  case "${exp}" in
    B_HIGHER|b_higher)
      run_exp \
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
      run_exp \
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
      run_exp \
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

echo "Requested JiT-B/L MSDT-refiner 3x3090 trainings finished: ${RUN_EXPS}"
