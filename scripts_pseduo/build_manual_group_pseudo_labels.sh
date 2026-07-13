#!/usr/bin/env bash
set -euo pipefail

# Build manually matched group pseudo labels, then optionally append them to
# RainDrop_Train/Drop and RainDrop_Train/Clear with updated scene JSON files.
#
# Example:
#   DATA_ROOT=/root/huilin/data/eccv_dn \
#   DROP_PRED_DIR=/root/huilin/data/eccv_dn/jit_submit_best_model_ema1_s1_r16_hflip_rot90_best_20260708_101616 \
#   TEST_PRED_DIR=/root/huilin/data/eccv_dn/jit_submit_best_model_ema1_s1_r16_hflip_rot90_best_20260710_151811 \
#   bash scripts_pseduo/build_manual_group_pseudo_labels.sh

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
# DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}

DROP_INPUT_DIR=${DROP_INPUT_DIR:-${DATA_ROOT}/Drop}
TEST_INPUT_DIR=${TEST_INPUT_DIR:-${DATA_ROOT}/test-input}
DROP_PRED_DIR=${DROP_PRED_DIR:-demo/ensemble_submit/submission_ensemble_w0p45__0p55_20260708_001359}
TEST_PRED_DIR=${TEST_PRED_DIR:-demo/ensemble_submit/submission_ensemble_uniform_20260710_000724}

MANUAL_PAIR_DIR=${MANUAL_PAIR_DIR:-${ROOT_DIR}/demo/test_pari_manual}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT_DIR}/demo/manual_group_pseudo}
PSEUDO_INPUT_DIR=${PSEUDO_INPUT_DIR:-}
PSEUDO_LABEL_IMAGE_DIR=${PSEUDO_LABEL_IMAGE_DIR:-}
PSEUDO_LABEL_DIR=${PSEUDO_LABEL_DIR:-}
PSEUDO_LABEL_COMPARE_DIR=${PSEUDO_LABEL_COMPARE_DIR:-}

RAIN_TRAIN_DIR=${RAIN_TRAIN_DIR:-${DATA_ROOT}/RainDrop_Train}
FOCUS2SCENE_JSON=${FOCUS2SCENE_JSON:-${RAIN_TRAIN_DIR}/Drop_focus_2scene.json}
BLUR2SCENE_JSON=${BLUR2SCENE_JSON:-${RAIN_TRAIN_DIR}/Drop_blur_2scene.json}
DN_BLUR_4SCENE_JSON=${DN_BLUR_4SCENE_JSON:-${RAIN_TRAIN_DIR}/Drop_dn_blur_4scene.json}
OUTPUT_FOCUS2SCENE_JSON=${OUTPUT_FOCUS2SCENE_JSON:-${RAIN_TRAIN_DIR}/Drop_focus_2scene_test_pseudo.json}
OUTPUT_BLUR2SCENE_JSON=${OUTPUT_BLUR2SCENE_JSON:-${RAIN_TRAIN_DIR}/Drop_blur_2scene_test_pseudo.json}
OUTPUT_DN_BLUR_4SCENE_JSON=${OUTPUT_DN_BLUR_4SCENE_JSON:-${RAIN_TRAIN_DIR}/Drop_dn_blur_4scene_test_pseudo.json}

TEST_GROUPS_CSV=${TEST_GROUPS_CSV:-${ROOT_DIR}/demo/test_group_from_folders.csv}
DROP_GROUPS_CSV=${DROP_GROUPS_CSV:-}
DROP_GROUP_MODE=${DROP_GROUP_MODE:-visual}
TEST_GROUP_MODE=${TEST_GROUP_MODE:-numeric-gap}
FEATURE_SIZE=${FEATURE_SIZE:-48}
NUMERIC_GAP=${NUMERIC_GAP:-1}
MAX_UNIQUE_IMAGES=${MAX_UNIQUE_IMAGES:-0}
UNIQUE_METHOD=${UNIQUE_METHOD:-closest_to_median}
TRAIN_COPY_PREFIX=${TRAIN_COPY_PREFIX:-test_pseudo_}
OVERWRITE=${OVERWRITE:-1}
INCLUDE_TARGET_TEST_GROUPS=${INCLUDE_TARGET_TEST_GROUPS:-1}
RESIZE_PSEUDO=${RESIZE_PSEUDO:-1}

missing_paths=()
check_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    missing_paths+=("${label}: ${path}")
  fi
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    missing_paths+=("${label}: ${path}")
  fi
}

check_dir "ROOT_DIR" "${ROOT_DIR}"
check_file "python tool" "${ROOT_DIR}/tools/build_manual_group_pseudo_labels.py"
check_file "pairing helper" "${ROOT_DIR}/tools/pair_test_input_to_clear.py"
check_dir "DROP_INPUT_DIR" "${DROP_INPUT_DIR}"
check_dir "TEST_INPUT_DIR" "${TEST_INPUT_DIR}"
check_dir "DROP_PRED_DIR" "${DROP_PRED_DIR}"
check_dir "TEST_PRED_DIR" "${TEST_PRED_DIR}"
check_dir "MANUAL_PAIR_DIR" "${MANUAL_PAIR_DIR}"
check_dir "RAIN_TRAIN_DIR" "${RAIN_TRAIN_DIR}"
check_dir "RainDrop_Train/Drop" "${RAIN_TRAIN_DIR}/Drop"
check_dir "RainDrop_Train/Clear" "${RAIN_TRAIN_DIR}/Clear"
check_file "FOCUS2SCENE_JSON" "${FOCUS2SCENE_JSON}"
check_file "BLUR2SCENE_JSON" "${BLUR2SCENE_JSON}"
check_file "DN_BLUR_4SCENE_JSON" "${DN_BLUR_4SCENE_JSON}"
if [[ -n "${TEST_GROUPS_CSV}" ]]; then
  check_file "TEST_GROUPS_CSV" "${TEST_GROUPS_CSV}"
fi
if [[ -n "${DROP_GROUPS_CSV}" ]]; then
  check_file "DROP_GROUPS_CSV" "${DROP_GROUPS_CSV}"
fi

if [[ "${#missing_paths[@]}" -gt 0 ]]; then
  echo "Missing required paths:" >&2
  printf '  - %s\n' "${missing_paths[@]}" >&2
  exit 2
fi

args=(
  --drop-input-dir "${DROP_INPUT_DIR}"
  --test-input-dir "${TEST_INPUT_DIR}"
  --drop-pred-dir "${DROP_PRED_DIR}"
  --test-pred-dir "${TEST_PRED_DIR}"
  --manual-pair-dir "${MANUAL_PAIR_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --rain-train-dir "${RAIN_TRAIN_DIR}"
  --focus2scene-json "${FOCUS2SCENE_JSON}"
  --blur2scene-json "${BLUR2SCENE_JSON}"
  --dn-blur-4scene-json "${DN_BLUR_4SCENE_JSON}"
  --output-focus2scene-json "${OUTPUT_FOCUS2SCENE_JSON}"
  --output-blur2scene-json "${OUTPUT_BLUR2SCENE_JSON}"
  --output-dn-blur-4scene-json "${OUTPUT_DN_BLUR_4SCENE_JSON}"
  --test-groups-csv "${TEST_GROUPS_CSV}"
  --drop-group-mode "${DROP_GROUP_MODE}"
  --test-group-mode "${TEST_GROUP_MODE}"
  --feature-size "${FEATURE_SIZE}"
  --numeric-gap "${NUMERIC_GAP}"
  --max-unique-images "${MAX_UNIQUE_IMAGES}"
  --unique-method "${UNIQUE_METHOD}"
  --train-copy-prefix "${TRAIN_COPY_PREFIX}"
)

if [[ -n "${PSEUDO_LABEL_DIR}" ]]; then
  args+=(--pseudo-label-dir "${PSEUDO_LABEL_DIR}")
fi
if [[ -n "${PSEUDO_INPUT_DIR}" ]]; then
  args+=(--pseudo-input-dir "${PSEUDO_INPUT_DIR}")
fi
if [[ -n "${PSEUDO_LABEL_IMAGE_DIR}" ]]; then
  args+=(--pseudo-label-image-dir "${PSEUDO_LABEL_IMAGE_DIR}")
fi
if [[ -n "${PSEUDO_LABEL_COMPARE_DIR}" ]]; then
  args+=(--pseudo-label-compare-dir "${PSEUDO_LABEL_COMPARE_DIR}")
fi
if [[ -n "${DROP_GROUPS_CSV}" ]]; then
  args+=(--drop-groups-csv "${DROP_GROUPS_CSV}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${INCLUDE_TARGET_TEST_GROUPS}" != "1" ]]; then
  args+=(--no-include-target-test-groups)
fi
if [[ "${RESIZE_PSEUDO}" != "1" ]]; then
  args+=(--no-resize-pseudo)
fi

echo "============================================================"
echo "[Manual group pseudo labels]"
echo "DATA_ROOT=${DATA_ROOT}"
echo "DROP_INPUT_DIR=${DROP_INPUT_DIR}"
echo "TEST_INPUT_DIR=${TEST_INPUT_DIR}"
echo "DROP_PRED_DIR=${DROP_PRED_DIR}"
echo "TEST_PRED_DIR=${TEST_PRED_DIR}"
echo "MANUAL_PAIR_DIR=${MANUAL_PAIR_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RAIN_TRAIN_DIR=${RAIN_TRAIN_DIR}"
echo "OUTPUT_FOCUS2SCENE_JSON=${OUTPUT_FOCUS2SCENE_JSON}"
echo "OUTPUT_BLUR2SCENE_JSON=${OUTPUT_BLUR2SCENE_JSON}"
echo "OUTPUT_DN_BLUR_4SCENE_JSON=${OUTPUT_DN_BLUR_4SCENE_JSON}"
echo "============================================================"

cd "${ROOT_DIR}"
python tools/build_manual_group_pseudo_labels.py "${args[@]}"
