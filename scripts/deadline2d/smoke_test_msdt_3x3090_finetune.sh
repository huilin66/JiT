#!/usr/bin/env bash
set -euo pipefail

# Fast end-to-end validation for train_msdt_3x3090_finetune.sh.
# Each RTX 3090 runs exactly one train step and validates one image.
# Success requires finite metrics, a non-zero gradient, an optimizer update,
# and both latest/best checkpoints from all three independent processes.

JIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MSDT_ROOT=${MSDT_ROOT:-"$(cd "${JIT_ROOT}/../MSDT" && pwd)"}
DATA_PATH=${DATA_PATH:-/root/huilin/data/eccv_dn/RainDrop_Train}
SOURCE_CKPT=${SOURCE_CKPT:-${MSDT_ROOT}/checkpoints/msdt_1x5090/no_scene/model_best.pth}
SMOKE_ROOT=${SMOKE_ROOT:-${MSDT_ROOT}/checkpoints/deadline2d_smoke_$(date +%Y%m%d_%H%M%S)}

GPUS=${GPUS:-"0 1 2"}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

read -r -a gpu_list <<< "${GPUS}"
if [[ ${#gpu_list[@]} -ne 3 ]]; then
  echo "GPUS must contain exactly three ids, e.g. GPUS=\"0 1 2\"" >&2
  exit 2
fi

required_paths=(
  "${MSDT_ROOT}/train_raindrop.py"
  "${MSDT_ROOT}/configs/raindrop_no_scene.yaml"
  "${SOURCE_CKPT}"
  "${DATA_PATH}/Drop"
  "${DATA_PATH}/Clear"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[preflight failed] missing: ${path}" >&2
    exit 2
  fi
done

echo "[preflight] creating or validating the deterministic group split"
python "${JIT_ROOT}/tools/ensure_msdt_split.py" \
  --msdt-root "${MSDT_ROOT}" \
  --data-root "${DATA_PATH}" \
  --config configs/raindrop_no_scene.yaml

visible_gpus=$(IFS=,; echo "${gpu_list[*]}")
echo "[preflight] checking CUDA devices ${visible_gpus}"
CUDA_VISIBLE_DEVICES="${visible_gpus}" python - <<'PY'
import json
import torch

count = torch.cuda.device_count()
if count != 3:
    raise RuntimeError(f"Expected 3 visible CUDA devices, found {count}")
devices = []
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    if "3090" not in props.name:
        raise RuntimeError(f"Visible device {index} is not an RTX 3090: {props.name}")
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"BF16 is unavailable on visible device {index}: {props.name}")
        value = torch.randn(64, 64, device=f"cuda:{index}", dtype=torch.bfloat16)
        result = value @ value
        if not torch.isfinite(result).all():
            raise RuntimeError(f"BF16 matmul is non-finite on device {index}")
    devices.append({
        "visible_index": index,
        "name": props.name,
        "memory_gib": round(props.total_memory / 1024 ** 3, 2),
        "capability": list(torch.cuda.get_device_capability(index)),
    })
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices}))
PY

cd "${MSDT_ROOT}"
echo "[preflight] importing training stack and initializing AlexNet LPIPS on CPU"
python - <<'PY'
import torch
from raindrop_metrics import ValidationMetrics
from raindrop_utils import load_config, build_model

config = load_config("configs/raindrop_no_scene.yaml")
model = build_model(config)
if not sum(parameter.numel() for parameter in model.parameters()) > 0:
    raise RuntimeError("MSDT contains no parameters")
ValidationMetrics(torch.device("cpu"))
print("TRAINING_STACK_OK")
PY

mkdir -p "${SMOKE_ROOT}/logs"
PREPARED_CKPT="${SMOKE_ROOT}/prepared_smoke_init.pth"
python "${JIT_ROOT}/tools/prepare_msdt_finetune_checkpoint.py" \
  --input "${SOURCE_CKPT}" \
  --output "${PREPARED_CKPT}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --epochs 1

names=(baseline edge edge_freq)
configs=(
  "${JIT_ROOT}/configs/deadline2d/msdt_finetune_baseline.yaml"
  "${JIT_ROOT}/configs/deadline2d/msdt_finetune_edge.yaml"
  "${JIT_ROOT}/configs/deadline2d/msdt_finetune_edge_freq.yaml"
)
pids=()

for index in 0 1 2; do
  name=${names[$index]}
  gpu=${gpu_list[$index]}
  output_dir="${SMOKE_ROOT}/${name}"
  log_path="${SMOKE_ROOT}/logs/${name}.log"
  mkdir -p "${output_dir}"
  echo "[launch] ${name} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train_raindrop.py \
    --config "${configs[$index]}" \
    --data-root "${DATA_PATH}" \
    --output-dir "${output_dir}" \
    --resume "${PREPARED_CKPT}" \
    --device cuda:0 \
    --epochs 1 \
    --stop-after-epoch 1 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --max-train-steps 1 \
    --max-val-images 1 \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

process_status=0
for index in 0 1 2; do
  if wait "${pids[$index]}"; then
    echo "[process completed] ${names[$index]}"
  else
    echo "[process failed] ${names[$index]}" >&2
    process_status=1
  fi
done

validation_status=${process_status}
for name in "${names[@]}"; do
  output_dir="${SMOKE_ROOT}/${name}"
  log_path="${SMOKE_ROOT}/logs/${name}.log"
  echo "---------------- ${name} log tail ----------------"
  tail -n 30 "${log_path}" || true

  for artifact in model_latest.pth model_best.pth metrics.csv; do
    if [[ ! -s "${output_dir}/${artifact}" ]]; then
      echo "[validation failed] ${name} missing/non-empty artifact: ${artifact}" >&2
      validation_status=1
    fi
  done
  if ! grep -q '"optimizer_update_verified": true' "${log_path}"; then
    echo "[validation failed] ${name} did not verify optimizer update" >&2
    validation_status=1
  fi
  if grep -Eq 'Traceback|CUDA out of memory|non-finite|FloatingPointError' "${log_path}"; then
    echo "[validation failed] ${name} log contains a fatal/numerical error" >&2
    validation_status=1
  fi
done

SMOKE_ROOT="${SMOKE_ROOT}" python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["SMOKE_ROOT"])
summary = {}
for name in ("baseline", "edge", "edge_freq"):
    path = root / name / "metrics.csv"
    if not path.is_file():
        continue
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metrics row for {name}, found {len(rows)}")
    row = rows[0]
    required = ("loss", "PSNR_Y", "SSIM_Y", "LPIPS", "Score")
    values = {key: float(row[key]) for key in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"Non-finite smoke metrics for {name}: {values}")
    summary[name] = values
print(json.dumps({"smoke_root": str(root), "metrics": summary}, indent=2))
PY

if [[ "${validation_status}" -ne 0 ]]; then
  echo "SMOKE_TEST_FAILED: inspect ${SMOKE_ROOT}/logs" >&2
  exit 1
fi

echo "SMOKE_TEST_PASSED"
echo "All three GPUs completed forward/backward/optimizer/validation/checkpoint paths."
echo "Artifacts: ${SMOKE_ROOT}"
