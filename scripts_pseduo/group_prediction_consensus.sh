#!/usr/bin/env bash
set -euo pipefail

# Post-process a flat test prediction folder with test_group consensus.
# Required:
#   PRED_DIR=/path/to/flat_predictions bash scripts_pseduo/group_prediction_consensus.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# PRED_DIR=${PRED_DIR:-D:/zhl/project/MSDT/submissions/msdt_1x5090/msdt_no_scene_best_20260714_225746}
# PRED_DIR=${PRED_DIR:-submissions_test/jit_l_e1_s1_r16_last_20260715_124201}
# PRED_DIR=${PRED_DIR:-submissions_test/jit_l_e1_s1_r16_last_20260715_134946}
PRED_DIR=${PRED_DIR:-submissions_test/ensemble_submit/submission_ensemble_uniform_20260715_153015}
TEST_GROUP_DIR=${TEST_GROUP_DIR:-demo/manual_group_pseudo/test_group_pure}
# closest_to_median: 32.3888
# sharpest: 32.2588
# 32.6659
METHOD=${METHOD:-inverse_median_distance}
TRIM_FRACTION=${TRIM_FRACTION:-0.2}
GROUP_WEIGHT=${GROUP_WEIGHT:-1.0}
MIN_GROUP_SIZE=${MIN_GROUP_SIZE:-2}
EXPECTED_COUNT=${EXPECTED_COUNT:-592}
COPY_UNGROUPED=${COPY_UNGROUPED:-1}
MAKE_ZIP=${MAKE_ZIP:-1}
OVERWRITE=${OVERWRITE:-1}

method_tag="${METHOD}_w${GROUP_WEIGHT//./p}"
OUTPUT_DIR=${OUTPUT_DIR:-${PRED_DIR%/}_group_${method_tag}}
ARCHIVE_PATH=${ARCHIVE_PATH:-${OUTPUT_DIR}.zip}

missing=()
if [[ -z "${PRED_DIR}" ]]; then
  missing+=("PRED_DIR environment variable")
elif [[ ! -d "${PRED_DIR}" ]]; then
  missing+=("PRED_DIR directory: ${PRED_DIR}")
fi
if [[ ! -d "${TEST_GROUP_DIR}" ]]; then
  missing+=("TEST_GROUP_DIR directory: ${TEST_GROUP_DIR}")
fi
if [[ ! -f "tools/group_prediction_consensus.py" ]]; then
  missing+=("tools/group_prediction_consensus.py")
fi
if (( ${#missing[@]} > 0 )); then
  echo "Missing required paths:" >&2
  for item in "${missing[@]}"; do
    echo "  - ${item}" >&2
  done
  exit 2
fi

args=(
  --pred-dir "${PRED_DIR}"
  --test-group-dir "${TEST_GROUP_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --method "${METHOD}"
  --trim-fraction "${TRIM_FRACTION}"
  --group-weight "${GROUP_WEIGHT}"
  --min-group-size "${MIN_GROUP_SIZE}"
)

if [[ "${EXPECTED_COUNT}" != "0" ]]; then
  args+=(--expected-count "${EXPECTED_COUNT}")
fi
if [[ "${COPY_UNGROUPED}" == "1" ]]; then
  args+=(--copy-ungrouped)
else
  args+=(--no-copy-ungrouped)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${MAKE_ZIP}" == "1" ]]; then
  args+=(--archive-path "${ARCHIVE_PATH}")
fi

echo "Group prediction consensus"
echo "  PRED_DIR       : ${PRED_DIR}"
echo "  TEST_GROUP_DIR : ${TEST_GROUP_DIR}"
echo "  OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "  METHOD         : ${METHOD}"
echo "  GROUP_WEIGHT   : ${GROUP_WEIGHT}"
echo "  EXPECTED_COUNT : ${EXPECTED_COUNT}"
if [[ "${MAKE_ZIP}" == "1" ]]; then
  echo "  ARCHIVE_PATH   : ${ARCHIVE_PATH}"
fi

python tools/group_prediction_consensus.py "${args[@]}"
