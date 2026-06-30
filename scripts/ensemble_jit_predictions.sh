#!/usr/bin/env bash
set -euo pipefail

# Multi-output ensemble for JiT predictions.
# Either pass INPUT_DIRS as a comma-separated list of prediction dirs, or
# provide LOCAL_CSV and TOP_K to select the best local validation outputs.

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
VAL_ROOT=${VAL_ROOT:-${DATA_ROOT}/jit_local_val_100}
DROP_DIR=${DROP_DIR:-${VAL_ROOT}/Drop}
CLEAR_DIR=${CLEAR_DIR:-${VAL_ROOT}/Clear}

OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/ensemble}
MODEL_NAME=${MODEL_NAME:-jit_ensemble}
LOCAL_CSV=${LOCAL_CSV:-}
TOP_K=${TOP_K:-5}
MIN_SCORE=${MIN_SCORE:-}
INPUT_DIRS=${INPUT_DIRS:-}
WEIGHTS=${WEIGHTS:-}
EVALUATE=${EVALUATE:-1}
DEVICE=${DEVICE:-cuda:0}

mkdir -p "${OUTPUT_ROOT}"

input_dirs=()
if [[ -n "${INPUT_DIRS}" ]]; then
  raw="${INPUT_DIRS//,/ }"
  # shellcheck disable=SC2206
  input_dirs=(${raw})
elif [[ -n "${LOCAL_CSV}" ]]; then
  top_args=(--csv "${LOCAL_CSV}" --top-k "${TOP_K}")
  if [[ -n "${MIN_SCORE}" ]]; then
    top_args+=(--min-score "${MIN_SCORE}")
  fi
  while IFS= read -r line; do
    input_dirs+=("${line}")
  done < <(python tools/top_local_sweep_dirs.py "${top_args[@]}")
else
  echo "Provide INPUT_DIRS or LOCAL_CSV." >&2
  exit 2
fi

if [[ "${#input_dirs[@]}" -lt 2 ]]; then
  echo "Need at least two prediction dirs for ensemble; got ${#input_dirs[@]}." >&2
  exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
ensemble_dir="${OUTPUT_ROOT}/${MODEL_NAME}_${stamp}"
ensemble_zip="${OUTPUT_ROOT}/${MODEL_NAME}_${stamp}.zip"
history_csv="${OUTPUT_ROOT}/ensemble_history.csv"

echo "============================================================"
echo "[Ensemble] ${#input_dirs[@]} prediction dirs"
printf '  %s\n' "${input_dirs[@]}"
echo "Output: ${ensemble_zip}"
echo "============================================================"

python tools/ensemble_submission_dirs.py \
  --input-dirs "${input_dirs[@]}" \
  --weights "${WEIGHTS}" \
  --output-dir "${ensemble_dir}" \
  --archive-path "${ensemble_zip}" \
  --history-csv "${history_csv}" \
  --model-name "${MODEL_NAME}" \
  --input-dir "${DROP_DIR}" \
  --notes "ensemble over ${#input_dirs[@]} dirs"

if [[ "${EVALUATE}" == "1" && -d "${CLEAR_DIR}" ]]; then
  python tools/evaluate_submission_dir.py \
    --prediction-dir "${ensemble_dir}" \
    --clear-dir "${CLEAR_DIR}" \
    --csv "${OUTPUT_ROOT}/ensemble_local_eval.csv" \
    --model-name "${MODEL_NAME}" \
    --checkpoint "$(IFS='|'; echo "${input_dirs[*]}")" \
    --ckpt-type "ensemble" \
    --state-key "ensemble" \
    --steps "ensemble" \
    --stride "ensemble" \
    --scene-json "" \
    --device "${DEVICE}" \
    --notes "ensemble over ${#input_dirs[@]} dirs"
fi

echo "Ensemble finished: ${ensemble_zip}"
