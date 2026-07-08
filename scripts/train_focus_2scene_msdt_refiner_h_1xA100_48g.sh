#!/usr/bin/env bash
set -euo pipefail

# Stage-2 MSDT-refiner sweep for focus-2scene JiT-H/16 on one A100.
# JiT-H is loaded from a trained focus-2scene checkpoint, frozen, and only the
# detail refiner is trained.
#
# Default experiments:
#   H_C1      - JiT-H/16 with the C1 refiner strategy
#   H_HIGHER  - JiT-H/16 with the stronger-than-C1 refiner strategy
#
# Examples:
#   RUN_EXPS="H_C1" bash scripts/train_focus_2scene_msdt_refiner_h_1xA100_48g.sh
#   BATCH_SIZE_H=2 RUN_EXPS="H_HIGHER" bash scripts/train_focus_2scene_msdt_refiner_h_1xA100_48g.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0}
NPROC=${NPROC:-1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30920}
RUN_EXPS=${RUN_EXPS:-"H_C1 H_HIGHER"}

DATA_PATH=${DATA_PATH:-/data/huilin/scrinvme/huilin/tp/eccv_dn/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

CKPT_H=${CKPT_H:-run/train/focus_2scene_h_no_head_1xA100_48g/jit_h16_focus_2scene_no_head/16}
OUT_ROOT=${OUT_ROOT:-run/train/focus_2scene_msdt_refiner_h_1xA100_48g}

IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE_H=${BATCH_SIZE_H:-32}
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

LR_C1=${LR_C1:-2.5e-4}
MAX_RESIDUAL_C1=${MAX_RESIDUAL_C1:-0.35}
EDGE_WEIGHT_C1=${EDGE_WEIGHT_C1:-0.10}
FREQ_WEIGHT_C1=${FREQ_WEIGHT_C1:-0.10}

LR_HIGHER=${LR_HIGHER:-3e-4}
MAX_RESIDUAL_HIGHER=${MAX_RESIDUAL_HIGHER:-0.40}
EDGE_WEIGHT_HIGHER=${EDGE_WEIGHT_HIGHER:-0.12}
FREQ_WEIGHT_HIGHER=${FREQ_WEIGHT_HIGHER:-0.12}

if [[ ! -f "${CKPT_H}" && ! -f "${CKPT_H}/checkpoint-last.pth" ]]; then
  echo "Cannot find JiT-H checkpoint: ${CKPT_H} (or ${CKPT_H}/checkpoint-last.pth)" >&2
  exit 2
fi

if [[ ! -f "${SCENE_FOCUS_2_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_FOCUS_2_PATH}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"
VISIBLE_GPU_COUNT=$(CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; print(torch.cuda.device_count())")
if (( NPROC > VISIBLE_GPU_COUNT )); then
  echo "NPROC=${NPROC} is larger than visible GPU count ${VISIBLE_GPU_COUNT} from GPUS=${GPUS}" >&2
  exit 2
fi

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

run_exp() {
  local tag="$1"
  local lr="$2"
  local max_residual="$3"
  local edge_weight="$4"
  local freq_weight="$5"
  local port="$6"
  local output_dir="${OUT_ROOT}/${tag}/16"

  echo "============================================================"
  echo "[MSDT refiner 1xA100 JiT-H] ${tag}"
  echo "ckpt=${CKPT_H}"
  echo "resume_state_key=${RESUME_STATE_KEY}"
  echo "GPUS=${GPUS}, nproc=${NPROC}, batch=${BATCH_SIZE_H}, lr=${lr}"
  echo "epochs=${EPOCHS}, eval_epoch=${EVAL_EPOCH}"
  echo "max_residual=${max_residual}, edge=${edge_weight}, freq=${freq_weight}"
  echo "scene_path=${SCENE_FOCUS_2_PATH}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    main_jit.py \
    --model "JiT-H/16" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE_H}" \
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
    --resume "${CKPT_H}" \
    --resume_state_key "${RESUME_STATE_KEY}" \
    --resume_optimizer 0 \
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}" \
    "${online_eval_args[@]}"
}

for exp in ${RUN_EXPS}; do
  case "${exp}" in
    H_C1|h_c1)
      run_exp \
        "h16_refiner_c1" \
        "${LR_C1}" \
        "${MAX_RESIDUAL_C1}" \
        "${EDGE_WEIGHT_C1}" \
        "${FREQ_WEIGHT_C1}" \
        "$((MASTER_PORT_BASE + 0))"
      ;;
    H_HIGHER|h_higher)
      run_exp \
        "h16_refiner_higher_than_c1" \
        "${LR_HIGHER}" \
        "${MAX_RESIDUAL_HIGHER}" \
        "${EDGE_WEIGHT_HIGHER}" \
        "${FREQ_WEIGHT_HIGHER}" \
        "$((MASTER_PORT_BASE + 1))"
      ;;
    *)
      echo "Unknown experiment: ${exp}. Use RUN_EXPS=\"H_C1 H_HIGHER\"." >&2
      exit 2
      ;;
  esac
done

echo "Requested JiT-H MSDT-refiner 1xA100 trainings finished: ${RUN_EXPS}"
