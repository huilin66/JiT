#!/usr/bin/env bash
set -euo pipefail

# Build a deliberately small candidate set from completed JiT/MSDT PNG folders.

JIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${JIT_ROOT}"

INPUT_DIR=${INPUT_DIR:-D:/zhl/data/eccv_dn/test-input}
EXPECTED_COUNT=${EXPECTED_COUNT:-592}
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/deadline2d/fusion}
JIT_ENV=${JIT_ENV:-submissions/deadline2d/jit/jit_outputs.env}
MSDT_ENV=${MSDT_ENV:-../MSDT/submissions/deadline2d/msdt_outputs.env}

if [[ -f "${JIT_ENV}" ]]; then source "${JIT_ENV}"; fi
if [[ -f "${MSDT_ENV}" ]]; then source "${MSDT_ENV}"; fi
: "${JIT_B_DIR:?Set JIT_B_DIR or provide JIT_ENV}"
: "${JIT_L_DIR:?Set JIT_L_DIR or provide JIT_ENV}"

# Older infer_jit_1x5090.sh manifests could accidentally record the archive
# instead of its sibling PNG directory because both matched the same glob.
normalize_jit_dir() {
  local path="$1"
  if [[ -f "${path}" && "${path}" == *.zip && -d "${path%.zip}" ]]; then
    path="${path%.zip}"
  fi
  if [[ ! -d "${path}" ]]; then
    echo "JiT PNG directory not found: ${path}" >&2
    return 2
  fi
  printf '%s\n' "${path}"
}
JIT_B_DIR=$(normalize_jit_dir "${JIT_B_DIR}")
JIT_L_DIR=$(normalize_jit_dir "${JIT_L_DIR}")

if [[ -z "${MSDT_PRIMARY_DIR:-}" ]]; then
  MSDT_PRIMARY_DIR=${MSDT_FT_EDGE_DIR:-${MSDT_FT_BASELINE_DIR:-${MSDT_ORIGINAL_NO_SCENE_DIR:-}}}
fi
: "${MSDT_PRIMARY_DIR:?Set MSDT_PRIMARY_DIR or provide MSDT_ENV}"

mkdir -p "${OUTPUT_ROOT}"
history="${OUTPUT_ROOT}/ensemble_history.csv"

make_fixed() {
  local name="$1"
  local weights="$2"
  shift 2
  python tools/ensemble_submission_dirs.py \
    --input-dirs "$@" --weights "${weights}" \
    --output-dir "${OUTPUT_ROOT}/${name}" --archive-path "${OUTPUT_ROOT}/${name}.zip" \
    --history-csv "${history}" --model-name "${name}" --input-dir "${INPUT_DIR}" \
    --notes "deadline2d fixed fusion weights=${weights}"
}

make_fixed fixed_030_030_040 "0.30,0.30,0.40" "${JIT_B_DIR}" "${JIT_L_DIR}" "${MSDT_PRIMARY_DIR}"
make_fixed fixed_040_025_035 "0.40,0.25,0.35" "${JIT_B_DIR}" "${JIT_L_DIR}" "${MSDT_PRIMARY_DIR}"
make_fixed jit_pair_060_040 "0.60,0.40" "${JIT_B_DIR}" "${JIT_L_DIR}"

python tools/deadline_postprocess.py adaptive \
  --input-dir "${INPUT_DIR}" --jit-dir "${OUTPUT_ROOT}/jit_pair_060_040" \
  --msdt-dir "${MSDT_PRIMARY_DIR}" --threshold 0.02 \
  --low-jit-weight 0.50 --high-jit-weight 0.65 --beta 1.0 \
  --output-dir "${OUTPUT_ROOT}/adaptive_050_065" \
  --archive-path "${OUTPUT_ROOT}/adaptive_050_065.zip" --expected-count "${EXPECTED_COUNT}"

for beta in 0.95 0.90; do
  suffix=${beta/./p}
  python tools/deadline_postprocess.py backblend \
    --input-dir "${INPUT_DIR}" --restored-dir "${OUTPUT_ROOT}/fixed_030_030_040" \
    --beta "${beta}" --output-dir "${OUTPUT_ROOT}/fixed_030_030_040_beta${suffix}" \
    --archive-path "${OUTPUT_ROOT}/fixed_030_030_040_beta${suffix}.zip" \
    --expected-count "${EXPECTED_COUNT}"
done

echo "Built candidates under ${OUTPUT_ROOT}:"
find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name '*.zip' -print
