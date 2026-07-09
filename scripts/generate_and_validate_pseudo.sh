#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 1: Generate mask-blended pseudo labels from teacher ensemble.
# Step 2: Validate pseudo quality against real GT (train set subset).
# Step 3: (Optional) Filter high-variance pseudo samples.
#
# This script should be run BEFORE any pseudo fine-tuning to verify that the
# mask-blended pseudo strategy is not degrading quality vs pure teacher output.
#
# Usage:
#   DATA_ROOT=D:/zhl/data/eccv_dn \
#   TEACHER_DIRS="submissions/run1 submissions/run2 submissions/run3" \
#   bash scripts/generate_and_validate_pseudo.sh
# ============================================================================

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_ROOT=${VAL_ROOT:-${DATA_ROOT}/jit_local_val_100}

# Teacher ensemble prediction directories (space-separated).
# These should be existing prediction outputs from your best models.
TEACHER_DIRS=${TEACHER_DIRS:-}

# Output locations.
PSEUDO_ROOT=${PSEUDO_ROOT:-${DATA_PATH}/pseudo_labels}
PSEUDO_OUTPUT=${PSEUDO_OUTPUT:-${PSEUDO_ROOT}/PseudoGT}
MASK_OUTPUT=${MASK_OUTPUT:-${PSEUDO_ROOT}/masks}
TTA_VARIANCE_CSV=${TTA_VARIANCE_CSV:-${PSEUDO_ROOT}/tta_variance.csv}
VALIDATE_OUTPUT=${VALIDATE_OUTPUT:-${PSEUDO_ROOT}/validation}

# Parameters.
FUSION=${FUSION:-trimmed_mean}
TRIM_LOW=${TRIM_LOW:-5.0}
TRIM_HIGH=${TRIM_HIGH:-95.0}
MASK_T1=${MASK_T1:-0.02}
MASK_T2=${MASK_T2:-0.08}
BLUR_SIGMA=${BLUR_SIGMA:-3.0}
BLEND_INPUT=${BLEND_INPUT:-0.85}
FILTER_TOP_PCT=${FILTER_TOP_PCT:-10.0}
SCENE_JSON=${SCENE_JSON:-}
SCENE_MEDIAN_WEIGHT=${SCENE_MEDIAN_WEIGHT:-0.0}
SCENE_MEDIAN_ALIGNED=${SCENE_MEDIAN_ALIGNED:-0}

# Validation subset (use a small subset for quick turn-around).
VAL_NUM=${VAL_NUM:-100}
VAL_SEED=${VAL_SEED:-2026}

mkdir -p "${PSEUDO_ROOT}" "${VALIDATE_OUTPUT}"

# ------------------------------------------------------------------
# Check prerequisites.
# ------------------------------------------------------------------
if [[ -z "${TEACHER_DIRS}" ]]; then
  echo "Error: TEACHER_DIRS must be set to one or more prediction directories." >&2
  echo "Example: TEACHER_DIRS='submissions/model1 submissions/model2'" >&2
  exit 2
fi

# ------------------------------------------------------------------
# Step 1: Generate pseudo labels for the training set.
# ------------------------------------------------------------------
echo "============================================================"
echo "[Step 1] Generate mask-blended pseudo labels"
echo "Input:      ${DATA_PATH}/Drop"
echo "Teachers:   ${TEACHER_DIRS}"
echo "Pseudo out: ${PSEUDO_OUTPUT}"
echo "Mask out:   ${MASK_OUTPUT}"
echo "============================================================"

scene_median_args=()
if [[ "${SCENE_MEDIAN_ALIGNED}" == "1" ]]; then
  scene_median_args+=(--scene-median-aligned)
fi

# shellcheck disable=SC2086
python tools/pseudo_label.py generate \
  --input-dir "${DATA_PATH}/Drop" \
  --pred-dirs ${TEACHER_DIRS} \
  --output-dir "${PSEUDO_OUTPUT}" \
  --mask-dir "${MASK_OUTPUT}" \
  --fusion "${FUSION}" \
  --trim-low "${TRIM_LOW}" \
  --trim-high "${TRIM_HIGH}" \
  --mask-t1 "${MASK_T1}" \
  --mask-t2 "${MASK_T2}" \
  --blur-sigma "${BLUR_SIGMA}" \
  --blend-input "${BLEND_INPUT}" \
  --tta-variance-csv "${TTA_VARIANCE_CSV}" \
  --scene-json "${SCENE_JSON}" \
  --scene-median-weight "${SCENE_MEDIAN_WEIGHT}" \
  "${scene_median_args[@]}"

# ------------------------------------------------------------------
# Step 2: Validate pseudo quality on a small GT subset.
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[Step 2] Validate pseudo strategies against GT"
echo "============================================================"

# Create a small fixed validation subset if it doesn't exist.
python tools/make_fixed_val_subset.py \
  --data-root "${DATA_PATH}" \
  --output-root "${VAL_ROOT}" \
  --num-samples "${VAL_NUM}" \
  --seed "${VAL_SEED}"

# Run validation: compare pure teacher, TTA-fused, and mask-blended vs real GT.
# shellcheck disable=SC2086
python tools/pseudo_label.py validate \
  --input-dir "${VAL_ROOT}/Drop" \
  --pred-dirs ${TEACHER_DIRS} \
  --gt-clear-dir "${VAL_ROOT}/Clear" \
  --output-dir "${VALIDATE_OUTPUT}" \
  --fusion "${FUSION}" \
  --trim-low "${TRIM_LOW}" \
  --trim-high "${TRIM_HIGH}" \
  --mask-t1 "${MASK_T1}" \
  --mask-t2 "${MASK_T2}" \
  --blur-sigma "${BLUR_SIGMA}" \
  --blend-input "${BLEND_INPUT}" \
  --save-viz

# ------------------------------------------------------------------
# Step 3: (Optional) Filter high-variance pseudo samples.
# ------------------------------------------------------------------
if [[ "${FILTER_TOP_PCT}" != "0" ]]; then
  echo ""
  echo "============================================================"
  echo "[Step 3] Filter pseudo samples by TTA variance"
  echo "Filtering top ${FILTER_TOP_PCT}% highest-variance samples"
  echo "============================================================"

  # Create filtered pseudo directory for V2 training.
  FILTERED_PSEUDO_DIR="${PSEUDO_ROOT}/PseudoGT_filtered"
  FILTERED_MASK_DIR="${PSEUDO_ROOT}/masks_filtered"

  python tools/pseudo_label.py filter \
    --tta-variance-csv "${TTA_VARIANCE_CSV}" \
    --output-dir "${PSEUDO_ROOT}" \
    --pseudo-dir "${PSEUDO_OUTPUT}" \
    --mask-dir "${MASK_OUTPUT}" \
    --filter-top-pct "${FILTER_TOP_PCT}"

  echo "Filtered pseudo: ${FILTERED_PSEUDO_DIR}"
  echo "Filtered masks:  ${FILTERED_MASK_DIR}"
fi

echo ""
echo "============================================================"
echo "Pseudo label pipeline complete."
echo ""
echo "Outputs:"
echo "  Pseudo GT:       ${PSEUDO_OUTPUT}"
echo "  Masks:           ${MASK_OUTPUT}"
echo "  TTA variance:    ${TTA_VARIANCE_CSV}"
echo "  Validation CSV:  ${VALIDATE_OUTPUT}/pseudo_validation.csv"
echo ""
echo "Next: run V1 training with:"
echo "  PSEUDO_DATA_PATH=${PSEUDO_OUTPUT} PSEUDO_MASK_DIR=${MASK_OUTPUT} VERSION=v1 bash scripts/train_pseudo.sh"
echo "============================================================"
