#!/usr/bin/env bash
set -euo pipefail

# Usage: bash .../run_scene_postprocess.sh PRED_DIR [PRED_DIR ...]
# Optional: MODEL_WEIGHTS=0.5,0.5 SCENE_WEIGHT=1.0 OUTPUT_ROOT=...
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 1 ]]; then echo "Usage: $0 PRED_DIR [PRED_DIR ...]" >&2; exit 2; fi
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/submissions/deadline2d_p1_post}
NAME=${NAME:-scene_consensus}
EXPECTED_COUNT=${EXPECTED_COUNT:-592}
SCENE_WEIGHT=${SCENE_WEIGHT:-1.0}
SCENE_METHOD=${SCENE_METHOD:-median}
MODEL_METHOD=${MODEL_METHOD:-weighted_mean}
args=(--pred-dirs "$@" --output-dir "${OUTPUT_ROOT}/${NAME}" --archive-path "${OUTPUT_ROOT}/${NAME}.zip"
      --expected-count "${EXPECTED_COUNT}" --scene-method "${SCENE_METHOD}"
      --model-method "${MODEL_METHOD}" --scene-weight "${SCENE_WEIGHT}")
if [[ -n "${MODEL_WEIGHTS:-}" ]]; then args+=(--model-weights "${MODEL_WEIGHTS}"); fi
python "${ROOT}/tools/scene_consensus_postprocess.py" "${args[@]}"

