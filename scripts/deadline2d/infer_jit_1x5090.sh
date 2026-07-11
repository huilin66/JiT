#!/usr/bin/env bash
set -euo pipefail

# Sequential JiT-B/L inference on one RTX 5090. Uses the supplied two-class JSON.

JIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${JIT_ROOT}"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/test-input}
SCENE_JSON=${SCENE_JSON:-${DATA_ROOT}/jit_submit_best_model_ema1_s1_r16_hflip_rot90_20260710_151804.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/deadline2d/jit}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/submission_history.csv}

JIT_B_CKPT=${JIT_B_CKPT:-run/train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16}
JIT_L_CKPT=${JIT_L_CKPT:-run/ablation_b16_3x3090/l16_refiner_higher_than_c1}
RUN_B=${RUN_B:-1}
RUN_L=${RUN_L:-1}
STATE_KEY=${STATE_KEY:-model_ema1}
STRIDE=${STRIDE:-16}
EXPECTED_COUNT=${EXPECTED_COUNT:-592}

for required in "${INPUT_DIR}" "${SCENE_JSON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
actual_count=$(find "${INPUT_DIR}" -maxdepth 1 -type f -iname '*.png' | wc -l)
if [[ "${actual_count}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "Expected ${EXPECTED_COUNT} input PNGs, found ${actual_count}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
env_path="${OUTPUT_ROOT}/jit_outputs.env"
: >"${env_path}"

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
