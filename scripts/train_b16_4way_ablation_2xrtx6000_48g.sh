#!/usr/bin/env bash
set -euo pipefail

# Four-way JiT-B/16 ablation on 2 x RTX 6000 48GB.
# The four experiments run sequentially and each experiment uses both GPUs.
# Default per-GPU batch is 64, giving a global batch of 128.
#
# Example:
# DATA_PATH=/data/RainDrop_Train2 \
# CKPT=/data/jit-b-16 \
# SCENE_TRAIN_PATH=/data/RainDrop_Train2/Drop_scen_pred.json \
# bash scripts/train_b16_4way_ablation_2xrtx6000_48g.sh

GPUS=${GPUS:-0,1}
NPROC=${NPROC:-2}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29810}

DATA_PATH=${DATA_PATH:-/scrinvme/huilin/tp/eccv_dn/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-/scrinvme/huilin/tp/eccv_dn//jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train/ablation_b16_2xrtx6000_48g}

SCENE_TRAIN_PATH=${SCENE_TRAIN_PATH:-${DATA_PATH}/Drop_scen_pred.json}
SCENE_VAL_PATH=${SCENE_VAL_PATH:-${SCENE_TRAIN_PATH}}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-600}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}

if [[ ! -d "${DATA_PATH}/Drop" || ! -d "${DATA_PATH}/Clear" ]]; then
  echo "Missing training folders: ${DATA_PATH}/Drop and ${DATA_PATH}/Clear" >&2
  exit 1
fi

if [[ ! -d "${CKPT}" ]]; then
  echo "Missing pretrained checkpoint directory: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "${SCENE_TRAIN_PATH}" ]]; then
  echo "Warning: scene label file not found: ${SCENE_TRAIN_PATH}" >&2
  echo "The two scene experiments will fail unless this path is corrected." >&2
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
  echo "GPUs=${GPUS}, batch_per_gpu=${BATCH_SIZE}, global_batch=$((BATCH_SIZE * NPROC))"
  echo "use_scene_dataset=${use_scene}, use_bg_subnet=${use_head}"
  echo "Output: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
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

echo "All four RTX 6000 ablation runs finished."
