#!/usr/bin/env bash
set -euo pipefail

# Windows compatibility settings
export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
# 更新为新的环境变量名
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

# Four-way JiT-B/16 ablation on one RTX 5090 32GB.
# Experiments run sequentially on GPU 0.
#
# Example:
# DATA_PATH=/data/RainDrop_Train2 \
# CKPT=/data/jit-b-16 \
# SCENE_TRAIN_PATH=/data/RainDrop_Train2/Drop_scen_pred.json \
# bash scripts/train_b16_4way_ablation_1x5090.sh

GPU=${GPU:-0}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30010}

DATA_PATH=${DATA_PATH:-D:/zhl/data/eccv_dn/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/ablation_b16_1x5090}

SCENE_TRAIN_PATH=${SCENE_TRAIN_PATH:-${DATA_PATH}/Drop_scen_pred.json}
SCENE_VAL_PATH=${SCENE_VAL_PATH:-${SCENE_TRAIN_PATH}}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-2.5e-5}
EPOCHS=${EPOCHS:-600}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

if [[ ! -d "${DATA_PATH}/Drop" || ! -d "${DATA_PATH}/Clear" ]]; then
  echo "Missing training folders: ${DATA_PATH}/Drop and ${DATA_PATH}/Clear" >&2
  exit 1
fi

if [[ ! -d "${VAL_DATA_PATH}/Drop" || ! -d "${VAL_DATA_PATH}/Clear" ]]; then
  echo "Missing validation folders: ${VAL_DATA_PATH}/Drop and ${VAL_DATA_PATH}/Clear" >&2
  exit 1
fi

if [[ ! -d "${CKPT}" ]]; then
  echo "Missing pretrained checkpoint directory: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "${SCENE_TRAIN_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_TRAIN_PATH}" >&2
  echo "Generate it with: python data_tools.py pseudo-scene --data-root ${DATA_PATH}" >&2
  exit 1
fi

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

run_exp() {
  local name="$1"
  local use_scene="$2"
  local use_head="$3"
  local port="$4"
  local output_dir="${OUT_ROOT}/${name}/16"
  local scene_args=()

  if [[ "${use_scene}" == "1" ]]; then
    scene_args+=(
      --scene_train_path "${SCENE_TRAIN_PATH}"
      --scene_val_path "${SCENE_VAL_PATH}"
    )
  fi

  echo "============================================================"
  echo "[Ablation] ${name}"
  echo "GPU=${GPU}, batch=${BATCH_SIZE}, lr=${LR}"
  echo "use_scene_dataset=${use_scene}, use_bg_subnet=${use_head}"
  echo "Output: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python main_jit.py \
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
    --output_dir "${output_dir}" \
    --use_bg_subnet "${use_head}" \
    --use_scene_dataset "${use_scene}" \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT}" \
    "${scene_args[@]}" \
    "${online_eval_args[@]}"
}

run_exp "b16_no_scene_no_head" 0 0 "$((MASTER_PORT_BASE + 0))"
run_exp "b16_scene_no_head"    1 0 "$((MASTER_PORT_BASE + 1))"
run_exp "b16_no_scene_head"    0 1 "$((MASTER_PORT_BASE + 2))"
run_exp "b16_scene_head"       1 1 "$((MASTER_PORT_BASE + 3))"

echo "All four RTX 5090 ablation runs finished."
