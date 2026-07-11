#!/usr/bin/env bash
set -euo pipefail

# Sequential original/fine-tuned MSDT inference on one RTX 5090.

JIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MSDT_ROOT=${MSDT_ROOT:-"$(cd "${JIT_ROOT}/../MSDT" && pwd)"}
cd "${MSDT_ROOT}"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/test-input}
SCENE_JSON=${SCENE_JSON:-${JIT_ROOT}/submissions/deadline2d/jit/scene_predictions/deadline_focus_2scene.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${MSDT_ROOT}/submissions/deadline2d}
FT_ROOT=${FT_ROOT:-${JIT_ROOT}/run/deadline2d_3x3090}
EXPECTED_COUNT=${EXPECTED_COUNT:-592}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/submission_history.csv}

ORIGINAL_NO_SCENE=${ORIGINAL_NO_SCENE:-${MSDT_ROOT}/checkpoints/msdt_1x5090/no_scene/model_latest.pth}
ORIGINAL_SCENE=${ORIGINAL_SCENE:-${MSDT_ROOT}/checkpoints/msdt_1x5090/scene/model_latest.pth}
RUN_ORIGINAL=${RUN_ORIGINAL:-1}
RUN_FINETUNES=${RUN_FINETUNES:-1}
RUN_SCENE=${RUN_SCENE:-1}

if [[ ! -d "${INPUT_DIR}" ]]; then echo "Missing INPUT_DIR: ${INPUT_DIR}" >&2; exit 2; fi
actual_count=$(find "${INPUT_DIR}" -maxdepth 1 -type f -iname '*.png' | wc -l)
if [[ "${actual_count}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "Expected ${EXPECTED_COUNT} input PNGs, found ${actual_count}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
env_path="${OUTPUT_ROOT}/msdt_outputs.env"
: >"${env_path}"

run_msdt() {
  local tag="$1"
  local config="$2"
  local weights="$3"
  local use_scene="$4"
  if [[ ! -f "${weights}" ]]; then
    echo "[skip] missing ${tag} checkpoint: ${weights}" >&2
    return 0
  fi
  local output_dir="${OUTPUT_ROOT}/${tag}"
  local archive_path="${OUTPUT_ROOT}/${tag}.zip"
  local scene_args=()
  if [[ "${use_scene}" == "1" ]]; then
    if [[ ! -f "${SCENE_JSON}" ]]; then echo "Missing SCENE_JSON: ${SCENE_JSON}" >&2; exit 2; fi
    scene_args+=(--scene-json "${SCENE_JSON}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python -u infer_raindrop.py \
    --config "${config}" --weights "${weights}" --input "${INPUT_DIR}" \
    --output-dir "${output_dir}" --device cuda:0 --tile-size 0 --tile-overlap 16 \
    --scale 1 --vflip --hflip --rot90 --rot180 --flatten-output \
    --archive-path "${archive_path}" --history-csv "${HISTORY_CSV}" \
    --model-name "${tag}" --notes "deadline2d_1x5090" "${scene_args[@]}"
  local variable
  variable=$(printf '%s' "MSDT_${tag^^}_DIR" | tr '-' '_')
  printf '%s=%q\n' "${variable}" "${output_dir}" >>"${env_path}"
}

if [[ "${RUN_ORIGINAL}" == "1" ]]; then
  run_msdt original_no_scene configs/raindrop_no_scene.yaml "${ORIGINAL_NO_SCENE}" 0
fi
if [[ "${RUN_FINETUNES}" == "1" ]]; then
  run_msdt ft_baseline configs/raindrop_no_scene.yaml "${FT_ROOT}/baseline/model_best.pth" 0
  run_msdt ft_edge configs/raindrop_no_scene.yaml "${FT_ROOT}/edge/model_best.pth" 0
  run_msdt ft_edge_freq configs/raindrop_no_scene.yaml "${FT_ROOT}/edge_freq/model_best.pth" 0
fi
if [[ "${RUN_SCENE}" == "1" ]]; then
  run_msdt original_scene configs/raindrop_scene.yaml "${ORIGINAL_SCENE}" 1
fi
echo "MSDT output manifest: ${env_path}"
