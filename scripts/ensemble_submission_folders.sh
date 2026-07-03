#!/usr/bin/env bash
set -euo pipefail

# Ensemble official submission prediction folders and package a ZIP.
# It does not require Clear images and does not run local evaluation.
#
# Example:
# INPUT_DIRS="submissions/jit_best,submissions/msdt_best" \
# MODEL_NAME="jit_msdt_ensemble" \
# bash scripts/ensemble_submission_folders.sh
#
# Try multiple two-model weights:
# INPUT_DIRS="submissions/jit_best,submissions/msdt_best" \
# WEIGHTS_LIST="0.5,0.5 0.6,0.4 0.4,0.6" \
# bash scripts/ensemble_submission_folders.sh

OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/ensemble_submit}
MODEL_NAME=${MODEL_NAME:-submission_ensemble}
INPUT_DIRS=${INPUT_DIRS:-}
WEIGHTS=${WEIGHTS:-}
WEIGHTS_LIST=${WEIGHTS_LIST:-}
HISTORY_CSV=${HISTORY_CSV:-${OUTPUT_ROOT}/ensemble_history.csv}
REMOVE_IMAGES_AFTER_ZIP=${REMOVE_IMAGES_AFTER_ZIP:-0}
NOTES=${NOTES:-official_submission_folder_ensemble}

if [[ -z "${INPUT_DIRS}" ]]; then
  echo "Set INPUT_DIRS to comma-separated prediction folders." >&2
  exit 2
fi

raw_dirs="${INPUT_DIRS//,/ }"
# shellcheck disable=SC2206
input_dirs=(${raw_dirs})
if [[ "${#input_dirs[@]}" -lt 2 ]]; then
  echo "Need at least two prediction folders; got ${#input_dirs[@]}." >&2
  exit 2
fi

if [[ -z "${WEIGHTS_LIST}" ]]; then
  if [[ -n "${WEIGHTS}" ]]; then
    WEIGHTS_LIST="${WEIGHTS}"
  else
    WEIGHTS_LIST="uniform"
  fi
fi

mkdir -p "${OUTPUT_ROOT}"

sanitize_weights() {
  local raw="$1"
  if [[ "${raw}" == "uniform" ]]; then
    echo "uniform"
    return
  fi
  raw="${raw//,/__}"
  raw="${raw//./p}"
  echo "w${raw}"
}

for weights_item in ${WEIGHTS_LIST}; do
  stamp=$(date +%Y%m%d_%H%M%S)
  weights_arg=()
  weights_note="uniform"
  weights_tag=$(sanitize_weights "${weights_item}")
  if [[ "${weights_item}" != "uniform" ]]; then
    weights_arg+=(--weights "${weights_item}")
    weights_note="${weights_item}"
  fi

  run_name="${MODEL_NAME}_${weights_tag}_${stamp}"
  ensemble_dir="${OUTPUT_ROOT}/${run_name}"
  ensemble_zip="${OUTPUT_ROOT}/${run_name}.zip"
  remove_args=()
  if [[ "${REMOVE_IMAGES_AFTER_ZIP}" == "1" ]]; then
    remove_args+=(--remove-images-after-zip)
  fi

  echo "============================================================"
  echo "[Submission Ensemble] ${run_name}"
  echo "weights=${weights_note}"
  printf '  %s\n' "${input_dirs[@]}"
  echo "zip=${ensemble_zip}"
  echo "============================================================"

  python tools/ensemble_submission_dirs.py \
    --input-dirs "${input_dirs[@]}" \
    "${weights_arg[@]}" \
    --output-dir "${ensemble_dir}" \
    --archive-path "${ensemble_zip}" \
    --history-csv "${HISTORY_CSV}" \
    --model-name "${run_name}" \
    --notes "${NOTES}; weights=${weights_note}; inputs=${INPUT_DIRS}" \
    "${remove_args[@]}"
done

echo "Submission ensemble finished: ${OUTPUT_ROOT}"
