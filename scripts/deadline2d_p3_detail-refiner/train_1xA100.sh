#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=${GPU:-0}; DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn/RainDrop_Train}
OUT_ROOT=${OUT_ROOT:-${ROOT}/run/deadline2d_p3_detail-refiner}; EPOCHS=${EPOCHS:-20}; BATCH_SIZE=${BATCH_SIZE:-8}; NUM_WORKERS=${NUM_WORKERS:-12}
JIT_B_CKPT=${JIT_B_CKPT:-${ROOT}/run/train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16/checkpoint-last.pth}
JIT_STATE_KEY=${JIT_STATE_KEY:-model_ema1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}; MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-0}
[[ -d "${DATA_ROOT}/Blur" && -d "${DATA_ROOT}/Clear" ]] || { echo "Missing Blur/Clear under ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${JIT_B_CKPT}" ]] || { echo "Missing JiT-B checkpoint: ${JIT_B_CKPT}" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="${GPU}" python -u "${ROOT}/tools/blur_clear_refiner.py" train \
  --kind detail --data-root "${DATA_ROOT}" --output-dir "${OUT_ROOT}" --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" --val-batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --patch-size 256 --base-dim 32 --blocks 2 --frequency 1 --max-residual 0.25 \
  --lr 1e-4 --min-lr 1e-6 --edge-weight 0.05 --freq-weight 0.02 --device cuda:0 --amp-dtype bf16 \
  --init-jit-checkpoint "${JIT_B_CKPT}" --init-state-key "${JIT_STATE_KEY}" \
  --max-train-steps "${MAX_TRAIN_STEPS}" --max-val-batches "${MAX_VAL_BATCHES}"
[[ -s "${OUT_ROOT}/model_best_score.pth" && -s "${OUT_ROOT}/model_best_psnr.pth" ]] || {
  echo 'Training ended without both score-best and PSNR-best checkpoints' >&2; exit 2;
}
