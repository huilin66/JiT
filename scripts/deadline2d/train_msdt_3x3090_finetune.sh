#!/usr/bin/env bash
set -euo pipefail

# Run three comparable short MSDT no-scene fine-tunes, one per RTX 3090.
# The source model is identical; only edge/frequency loss weights differ.

JIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MSDT_ROOT=${MSDT_ROOT:-"$(cd "${JIT_ROOT}/../MSDT" && pwd)"}
DATA_PATH=${DATA_PATH:-D:/zhl/data/eccv_dn/RainDrop_Train}
SOURCE_CKPT=${SOURCE_CKPT:-${MSDT_ROOT}/checkpoints/msdt_1x5090/no_scene/model_best.pth}
OUT_ROOT=${OUT_ROOT:-${MSDT_ROOT}/checkpoints/deadline2d_3x3090}

GPUS=${GPUS:-"0 1 2"}
EPOCHS=${EPOCHS:-8}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-6}

read -r -a gpu_list <<< "${GPUS}"
if [[ ${#gpu_list[@]} -ne 3 ]]; then
  echo "GPUS must contain exactly three GPU ids, e.g. GPUS=\"0 1 2\"" >&2
  exit 2
fi
for required in "${SOURCE_CKPT}" "${MSDT_ROOT}/train_raindrop.py" "${MSDT_ROOT}/splits/raindrop_split.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATA_PATH}/Drop" || ! -d "${DATA_PATH}/Clear" ]]; then
  echo "DATA_PATH must contain Drop and Clear: ${DATA_PATH}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"
PREPARED_CKPT="${OUT_ROOT}/prepared_finetune_init.pth"
python "${JIT_ROOT}/tools/prepare_msdt_finetune_checkpoint.py" \
  --input "${SOURCE_CKPT}" \
  --output "${PREPARED_CKPT}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --epochs "${EPOCHS}"

names=(baseline edge edge_freq)
configs=(
  "${JIT_ROOT}/configs/deadline2d/msdt_finetune_baseline.yaml"
  "${JIT_ROOT}/configs/deadline2d/msdt_finetune_edge.yaml"
  "${JIT_ROOT}/configs/deadline2d/msdt_finetune_edge_freq.yaml"
)
pids=()

cd "${MSDT_ROOT}"
for index in 0 1 2; do
  name=${names[$index]}
  gpu=${gpu_list[$index]}
  config=${configs[$index]}
  output_dir="${OUT_ROOT}/${name}"
  log_path="${OUT_ROOT}/logs/${name}.log"
  mkdir -p "${output_dir}"
  echo "[launch] ${name}: physical GPU ${gpu}, output=${output_dir}, log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train_raindrop.py \
    --config "${config}" \
    --data-root "${DATA_PATH}" \
    --output-dir "${output_dir}" \
    --resume "${PREPARED_CKPT}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --stop-after-epoch "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1 2; do
  if wait "${pids[$index]}"; then
    echo "[completed] ${names[$index]}"
  else
    echo "[failed] ${names[$index]} -- inspect ${OUT_ROOT}/logs/${names[$index]}.log" >&2
    status=1
  fi
done

echo "Metrics:"
for name in "${names[@]}"; do
  metrics="${OUT_ROOT}/${name}/metrics.csv"
  if [[ -f "${metrics}" ]]; then
    echo "  ${name}: ${metrics}"
    tail -n 2 "${metrics}"
  fi
done
exit "${status}"
