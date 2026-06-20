#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for the four JiT-B/16 ablation settings on
# 2 x RTX 6000 48GB. Each setting runs two train iterations and a tiny
# online evaluation, then writes checkpoint-last.pth and checkpoint-best.pth.
#
# Example:
# DATA_PATH=/data/RainDrop_Train2 \
# CKPT=/data/jit-b-16 \
# SCENE_TRAIN_PATH=/data/RainDrop_Train2/Drop_scen_pred.json \
# bash scripts/smoke_test_b16_4way_ablation_2xrtx6000_48g.sh

GPUS=${GPUS:-0,1}
NPROC=${NPROC:-2}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29910}

DATA_PATH=${DATA_PATH:-/data/RainDrop_Train2}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-/data/jit-b-16}
OUT_ROOT=${OUT_ROOT:-/tmp/jit_b16_2xrtx6000_smoke}

SCENE_TRAIN_PATH=${SCENE_TRAIN_PATH:-${DATA_PATH}/Drop_scen_pred.json}
SCENE_VAL_PATH=${SCENE_VAL_PATH:-${SCENE_TRAIN_PATH}}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-2}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}

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

run_smoke() {
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
  echo "[Smoke] ${name}"
  echo "use_scene_dataset=${use_scene}, use_bg_subnet=${use_head}"
  echo "train_steps=${MAX_TRAIN_STEPS}, eval_images=${EVAL_NUM_IMAGES}"
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
    --epochs 1 \
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
    --use_bg_subnet "${use_head}" \
    --use_scene_dataset "${use_scene}" \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT}" \
    --online_eval \
    "${scene_args[@]}"

  if [[ ! -f "${output_dir}/checkpoint-last.pth" ]]; then
    echo "Smoke test failed: checkpoint-last.pth was not created for ${name}" >&2
    exit 1
  fi

  echo "[PASS] ${name}"
}

run_smoke "b16_no_scene_no_head" 0 0 "$((MASTER_PORT_BASE + 0))"
run_smoke "b16_scene_no_head"    1 0 "$((MASTER_PORT_BASE + 1))"
run_smoke "b16_no_scene_head"    0 1 "$((MASTER_PORT_BASE + 2))"
run_smoke "b16_scene_head"       1 1 "$((MASTER_PORT_BASE + 3))"

echo "All four RTX 6000 smoke tests passed."
