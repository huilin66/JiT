#!/usr/bin/env bash
set -euo pipefail

# Sequential JiT-B/L inference on one RTX 5090. By default, predicts a fresh
# two-class scene JSON once and reuses it for both restoration models.

JIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${JIT_ROOT}"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/test-input}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/deadline2d/jit}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/submission_history.csv}
SCENE_JSON=${SCENE_JSON:-}
SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE:-128}
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}

JIT_B_CKPT=${JIT_B_CKPT:-run/train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16}
JIT_L_CKPT=${JIT_L_CKPT:-run/ablation_b16_3x3090/l16_refiner_higher_than_c1}
RUN_B=${RUN_B:-1}
RUN_L=${RUN_L:-1}
STATE_KEY=${STATE_KEY:-model_ema1}
STRIDE=${STRIDE:-16}
EXPECTED_COUNT=${EXPECTED_COUNT:-592}

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Missing input directory: ${INPUT_DIR}" >&2
  exit 2
fi
actual_count=$(find "${INPUT_DIR}" -maxdepth 1 -type f -iname '*.png' | wc -l)
if [[ "${actual_count}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "Expected ${EXPECTED_COUNT} input PNGs, found ${actual_count}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
env_path="${OUTPUT_ROOT}/jit_outputs.env"
: >"${env_path}"

if [[ -z "${SCENE_JSON}" ]]; then
  if [[ ! -f "${SCENE_CKPT}" ]]; then
    echo "Missing scene checkpoint: ${SCENE_CKPT}" >&2
    exit 2
  fi
  scene_dir="${OUTPUT_ROOT}/scene_predictions"
  mkdir -p "${scene_dir}"
  SCENE_JSON="${scene_dir}/deadline_focus_2scene.json"
  echo "Predicting fresh two-class scene JSON with ${SCENE_CKPT}"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m scene_tools.infer_scene_convnext \
    --input-dir "${INPUT_DIR}" \
    --checkpoint "${SCENE_CKPT}" \
    --output-json "${SCENE_JSON}" \
    --output-csv "${scene_dir}/deadline_focus_2scene.csv" \
    --batch-size "${SCENE_BATCH_SIZE}" \
    --num-workers "${SCENE_NUM_WORKERS}" \
    --device cuda:0 \
    --amp-dtype auto \
    --no-recursive
else
  echo "Using supplied scene JSON: ${SCENE_JSON}"
fi

python - "${INPUT_DIR}" "${SCENE_JSON}" "${EXPECTED_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

input_dir, json_path, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if not json_path.is_file():
    raise SystemExit(f"Missing scene JSON: {json_path}")
labels = json.loads(json_path.read_text(encoding="utf-8"))
names = {path.name for path in input_dir.glob("*.png")}
keys = set(labels)
classes = {int(value) for value in labels.values()}
if len(labels) != expected or keys != names:
    raise SystemExit(
        f"Scene JSON coverage mismatch: labels={len(labels)}, inputs={len(names)}, "
        f"missing={len(names - keys)}, extra={len(keys - names)}"
    )
if not classes <= {0, 1}:
    raise SystemExit(f"Expected two-class scene labels 0/1, got {sorted(classes)}")
counts = {class_id: sum(int(value) == class_id for value in labels.values()) for class_id in (0, 1)}
print(f"Validated scene JSON: {json_path}; class counts={counts}")
PY
printf 'JIT_SCENE_JSON=%q\n' "${SCENE_JSON}" >>"${env_path}"

run_jit() {
  local tag="$1"
  local checkpoint="$2"
  local tile_batch="$3"
  if [[ ! -e "${checkpoint}" ]]; then
    echo "Missing JiT checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  DATA_ROOT="${DATA_ROOT}" INPUT_DIR="${INPUT_DIR}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
  HISTORY_CSV="${HISTORY_CSV}" JIT_CKPT="${checkpoint}" JIT_CKPT_TYPE=best \
  STATE_KEY="${STATE_KEY}" STEPS=1 STRIDE="${STRIDE}" TILE_BATCH_SIZE="${tile_batch}" \
  SCENE_JSON="${SCENE_JSON}" MODEL_NAME="deadline_${tag}" \
  TTA_HFLIP=1 TTA_VFLIP=0 TTA_ROT90=1 TTA_ROT180=0 TTA_ROT270=0 SCALES=1 \
  bash scripts/submit_jit_from_local_config.sh

  shopt -s nullglob
  local matches=("${OUTPUT_ROOT}/deadline_${tag}_best_"*)
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "Cannot locate output directory for deadline_${tag}" >&2
    exit 2
  fi
  local latest=${matches[$((${#matches[@]} - 1))]}
  local variable
  variable=$(printf '%s' "JIT_${tag^^}_DIR" | tr '-' '_')
  printf '%s=%q\n' "${variable}" "${latest}" >>"${env_path}"
  echo "${variable}=${latest}"
}

if [[ "${RUN_B}" == "1" ]]; then run_jit b "${JIT_B_CKPT}" 32; fi
if [[ "${RUN_L}" == "1" ]]; then run_jit l "${JIT_L_CKPT}" 16; fi
echo "JiT output manifest: ${env_path}"
