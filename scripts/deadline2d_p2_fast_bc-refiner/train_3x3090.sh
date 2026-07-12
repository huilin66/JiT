#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn/RainDrop_Train}
OUT_ROOT=${OUT_ROOT:-${ROOT}/run/deadline2d_p2_fast_bc-refiner}
JIT_B_CKPT=${JIT_B_CKPT:-${ROOT}/run/train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16/checkpoint-last.pth}
JIT_STATE_KEY=${JIT_STATE_KEY:-model_ema1}
GPUS=${GPUS:-"0 1 2"}; EPOCHS=${EPOCHS:-100}; BATCH_SIZE=${BATCH_SIZE:-16}; NUM_WORKERS=${NUM_WORKERS:-8}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}; MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-0}
read -r -a gpu <<<"${GPUS}"; [[ ${#gpu[@]} -eq 3 ]] || { echo 'GPUS needs exactly 3 ids' >&2; exit 2; }
[[ -d "${DATA_ROOT}/Blur" && -d "${DATA_ROOT}/Clear" ]] || { echo "Missing Blur/Clear under ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${JIT_B_CKPT}" ]] || { echo "Missing JiT-B checkpoint: ${JIT_B_CKPT}" >&2; exit 2; }
names=(conservative balanced edge_freq); edge=(0.02 0.05 0.08); freq=(0.005 0.01 0.02)
mkdir -p "${OUT_ROOT}/logs"; pids=()
for i in 0 1 2; do
  echo "[launch] ${names[$i]} GPU=${gpu[$i]}"
  CUDA_VISIBLE_DEVICES="${gpu[$i]}" python -u "${ROOT}/tools/blur_clear_refiner.py" train \
    --kind detail --data-root "${DATA_ROOT}" --output-dir "${OUT_ROOT}/${names[$i]}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --val-batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" --patch-size 256 --base-dim 32 --blocks 2 --frequency 1 \
    --lr 1e-4 --min-lr 1e-6 --edge-weight "${edge[$i]}" --freq-weight "${freq[$i]}" \
    --max-residual 0.25 --init-jit-checkpoint "${JIT_B_CKPT}" --init-state-key "${JIT_STATE_KEY}" \
    --device cuda:0 --amp-dtype bf16 \
    --max-train-steps "${MAX_TRAIN_STEPS}" --max-val-batches "${MAX_VAL_BATCHES}" \
    >"${OUT_ROOT}/logs/${names[$i]}.log" 2>&1 & pids+=("$!")
done
status=0; for i in 0 1 2; do wait "${pids[$i]}" || status=1; done
for name in "${names[@]}"; do
  [[ -s "${OUT_ROOT}/${name}/model_best_score.pth" && -s "${OUT_ROOT}/${name}/model_best_psnr.pth" ]] || status=1
  tail -n 3 "${OUT_ROOT}/logs/${name}.log" || true
done
exit "${status}"
