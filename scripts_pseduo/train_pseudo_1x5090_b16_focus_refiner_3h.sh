#!/usr/bin/env bash
set -euo pipefail

# 1x5090 experiment:
# Fine-tune from the best focus-2scene JiT-B/16 refiner checkpoint on the
# pseudo-augmented RainDrop_Train. Runtime target: about 3h.
#
# The current best B16 refiner submission in submission_history_jit.csv is:
#   score=34.4282
#   run=train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16
#   ckpt_type=last, state_key=model_ema1
#
# Estimate from prior runs: B16 refiner 50 epochs ~= 2h on 1x5090.
# Instead of betting the whole 3h budget on one LR, the default runs three
# short pseudo-finetune plans from the same C1 checkpoint:
#   SAFE: 25 epochs, lr=1e-5
#   MID:  25 epochs, lr=2.5e-5
#   HIGH: 25 epochs, lr=5e-5

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPU=${GPU:-0}
RUN_PLANS=${RUN_PLANS:-"SAFE MID HIGH"}
ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

CKPT=${CKPT:-run/train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16}
OUT_ROOT=${OUT_ROOT:-run/train_pseudo}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene_test_pseudo.json}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS_PER_PLAN=${EPOCHS_PER_PLAN:-25}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}
RESUME_STATE_KEY=${RESUME_STATE_KEY:-model_ema1}

REFINER_BASE_DIM=${REFINER_BASE_DIM:-32}
REFINER_NUM_BLOCKS=${REFINER_NUM_BLOCKS:-2}
REFINER_USE_FREQUENCY=${REFINER_USE_FREQUENCY:-1}
REFINER_MAX_RESIDUAL=${REFINER_MAX_RESIDUAL:-0.35}
LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.10}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.10}
LR_SAFE=${LR_SAFE:-1e-5}
LR_MID=${LR_MID:-2.5e-5}
LR_HIGH=${LR_HIGH:-5e-5}

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
check_file "SCENE_FOCUS_2_PATH" "${SCENE_FOCUS_2_PATH}"
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
echo "[Pseudo 1x5090] B16 focus2scene refiner sweep, target ~=3h"
echo "DATA_PATH=${DATA_PATH}"
echo "CKPT=${CKPT}"
echo "SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "RUN_PLANS=${RUN_PLANS}"
echo "GPU=${GPU}, batch=${BATCH_SIZE}, epochs_per_plan=${EPOCHS_PER_PLAN}"
echo "refiner_max_residual=${REFINER_MAX_RESIDUAL}, edge=${LOSS_EDGE_WEIGHT}, freq=${LOSS_FREQ_WEIGHT}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

run_plan() {
  local plan="$1"
  local lr="$2"
  local output_dir="${OUT_ROOT}/b16_focus_2scene_refiner_c1_pseudo_ft_${plan,,}_${EPOCHS_PER_PLAN}ep_1x5090/16"

  echo "============================================================"
  echo "[Pseudo 1x5090] plan=${plan}, lr=${lr}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python main_jit.py \
    --model "${MODEL}" \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${lr}" \
    --lr_schedule cosine \
    --epochs "${EPOCHS_PER_PLAN}" \
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
}

for plan in ${RUN_PLANS}; do
  case "${plan}" in
    SAFE|safe)
      run_plan "safe" "${LR_SAFE}"
      ;;
    MID|mid)
      run_plan "mid" "${LR_MID}"
      ;;
    HIGH|high)
      run_plan "high" "${LR_HIGH}"
      ;;
    *)
      echo "Unknown plan: ${plan}. Use RUN_PLANS=\"SAFE MID HIGH\"." >&2
      exit 2
      ;;
  esac
done

echo "Pseudo B16 focus2scene refiner sweep finished: ${RUN_PLANS}"
