#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
INPUT_DIR=${INPUT_DIR:-${DATA_ROOT}/test-input}
MSDT_ROOT=${MSDT_ROOT:-"$(cd "${ROOT}/../MSDT" && pwd)"}
JIT_OUTPUT_ROOT=${JIT_OUTPUT_ROOT:-${ROOT}/submissions/deadline2d/jit}
MSDT_OUTPUT_ROOT=${MSDT_OUTPUT_ROOT:-${MSDT_ROOT}/submissions/deadline2d}
FUSION_OUTPUT_ROOT=${FUSION_OUTPUT_ROOT:-${ROOT}/submissions/deadline2d/fusion}
FT_ROOT=${FT_ROOT:-${MSDT_ROOT}/checkpoints/deadline2d_3x3090}

GPU="${GPU}" DATA_ROOT="${DATA_ROOT}" INPUT_DIR="${INPUT_DIR}" \
  OUTPUT_ROOT="${JIT_OUTPUT_ROOT}" \
  bash scripts/deadline2d/infer_jit_1x5090.sh

GPU="${GPU}" DATA_ROOT="${DATA_ROOT}" INPUT_DIR="${INPUT_DIR}" MSDT_ROOT="${MSDT_ROOT}" \
  OUTPUT_ROOT="${MSDT_OUTPUT_ROOT}" FT_ROOT="${FT_ROOT}" \
  SCENE_JSON="${JIT_OUTPUT_ROOT}/scene_predictions/deadline_focus_2scene.json" \
  bash scripts/deadline2d/infer_msdt_1x5090.sh

INPUT_DIR="${INPUT_DIR}" \
  JIT_ENV="${JIT_OUTPUT_ROOT}/jit_outputs.env" \
  MSDT_ENV="${MSDT_OUTPUT_ROOT}/msdt_outputs.env" \
  OUTPUT_ROOT="${FUSION_OUTPUT_ROOT}" \
  bash scripts/deadline2d/build_fusion_candidates.sh
