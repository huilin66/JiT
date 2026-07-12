#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=${GPU:-0}; INPUT_DIR=${INPUT_DIR:-}; DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}; TEST_INPUT_DIR=${TEST_INPUT_DIR:-${DATA_ROOT}/test-input}
BEST_KIND=${BEST_KIND:-score}; [[ "${BEST_KIND}" == "score" || "${BEST_KIND}" == "psnr" ]] || { echo 'BEST_KIND must be score or psnr' >&2; exit 2; }
CHECKPOINT=${CHECKPOINT:-${ROOT}/run/deadline2d_p3_detail-refiner/model_best_${BEST_KIND}.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/submissions/deadline2d_p3_detail-refiner}; EXPECTED_COUNT=${EXPECTED_COUNT:-592}; STRENGTHS=${STRENGTHS:-"0.75 1.0"}
JIT_B_CKPT=${JIT_B_CKPT:-${ROOT}/run/train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16/checkpoint-last.pth}
JIT_STATE_KEY=${JIT_STATE_KEY:-model_ema1}; SCENE_JSON=${SCENE_JSON:-}; BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT:-${OUTPUT_ROOT}/jit_b_base}
[[ -s "${CHECKPOINT}" ]] || { echo "Missing ${CHECKPOINT}" >&2; exit 2; }; mkdir -p "${OUTPUT_ROOT}"
if [[ -z "${INPUT_DIR}" ]]; then
  [[ -s "${JIT_B_CKPT}" ]] || { echo "Missing JiT-B checkpoint: ${JIT_B_CKPT}" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES="${GPU}" DATA_ROOT="${DATA_ROOT}" INPUT_DIR="${TEST_INPUT_DIR}" OUTPUT_ROOT="${BASE_OUTPUT_ROOT}" \
    JIT_CKPT="${JIT_B_CKPT}" JIT_CKPT_TYPE=last STATE_KEY="${JIT_STATE_KEY}" SCENE_JSON="${SCENE_JSON}" \
    MODEL_NAME=p3_jit_b_base STEPS=1 STRIDE=16 TTA_HFLIP=1 TTA_ROT90=1 TILE_BATCH_SIZE=32 \
    bash "${ROOT}/scripts/submit_jit_from_local_config.sh"
  shopt -s nullglob; candidates=("${BASE_OUTPUT_ROOT}/p3_jit_b_base_last_"*); shopt -u nullglob; dirs=()
  for candidate in "${candidates[@]}"; do [[ -d "${candidate}" ]] && dirs+=("${candidate}"); done
  [[ ${#dirs[@]} -gt 0 ]] || { echo 'Cannot locate generated P3 JiT-B base directory' >&2; exit 2; }
  INPUT_DIR=${dirs[$((${#dirs[@]} - 1))]}
fi
echo "P3 base predictions: ${INPUT_DIR}"
for strength in ${STRENGTHS}; do tag="detail_${BEST_KIND}_s${strength/./p}"
  CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/tools/blur_clear_refiner.py" infer --checkpoint "${CHECKPOINT}" \
    --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_ROOT}/${tag}" --archive-path "${OUTPUT_ROOT}/${tag}.zip" \
    --strength "${strength}" --expected-count "${EXPECTED_COUNT}" --device cuda:0 --amp-dtype bf16
done
