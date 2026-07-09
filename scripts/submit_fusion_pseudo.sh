#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Submission fusion: blend original ensemble with pseudo-finetuned model output.
#
# Two-step process:
#   1. Run inference with the pseudo-finetuned checkpoint on the TEST set.
#   2. Fuse the new predictions with the original best ensemble.
#
# Default fusion: final = W_ENS * ensemble + (1-W_ENS) * pseudo_model
# V1 default: W_ENS = 0.70 (conservative)
# V2 default: W_ENS = 0.60
# V3 default: W_ENS = 0.60
#
# Also supports multi-checkpoint ensemble from V1/V2/V3 training.
#
# Usage:
#   DATA_ROOT=D:/zhl/data/eccv_dn \
#   TEST_DIR=D:/zhl/data/eccv_dn/Test/Drop \
#   PSEUDO_CKPT=run/train/v1_pseudo_refiner/16 \
#   ENSEMBLE_DIR=submissions/best_ensemble \
#   SCENE_CKPT=run/scene_convnext_focus_2scene_v1/checkpoint-best.pth \
#   bash scripts/submit_fusion_pseudo.sh
# ============================================================================

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
TEST_DIR=${TEST_DIR:-${DATA_ROOT}/Test/Drop}

# --- Pseudo-finetuned model ---
PSEUDO_CKPT=${PSEUDO_CKPT:-run/train/v1_pseudo_refiner/16}
PSEUDO_CKPT_TYPE=${PSEUDO_CKPT_TYPE:-best}
PSEUDO_STATE_KEY=${PSEUDO_STATE_KEY:-model_ema2}
PSEUDO_STEPS=${PSEUDO_STEPS:-1}
PSEUDO_STRIDE=${PSEUDO_STRIDE:-64}
PSEUDO_SCALES=${PSEUDO_SCALES:-1.0}
PSEUDO_TTA_HFLIP=${PSEUDO_TTA_HFLIP:-1}

# --- Scene model ---
SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SCENE_JSON=${SCENE_JSON:-}

# --- Original ensemble ---
# One or more prediction directories from the best ensemble models.
ENSEMBLE_DIRS=${ENSEMBLE_DIRS:-}

# --- Fusion weights ---
# W_ENS = weight on original ensemble. (1 - W_ENS) = weight on pseudo model.
W_ENS=${W_ENS:-0.70}
VERSION_TAG=${VERSION_TAG:-v1}
ENSEMBLE_FUSION=${ENSEMBLE_FUSION:-weighted_mean}

# --- Output ---
OUTPUT_ROOT=${OUTPUT_ROOT:-submissions/fusion_${VERSION_TAG}}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
TILE_BATCH_SIZE=${TILE_BATCH_SIZE:-8}

# ===================================================================
if [[ ! -f "${SCENE_CKPT}" ]]; then
  echo "Scene checkpoint not found: ${SCENE_CKPT}" >&2; exit 2
fi
if [[ ! -f "${PSEUDO_CKPT}" && ! -d "${PSEUDO_CKPT}" ]]; then
  echo "Pseudo-finetuned checkpoint not found: ${PSEUDO_CKPT}" >&2; exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

# ------------------------------------------------------------------
# Step 1: Scene label prediction for test set.
# ------------------------------------------------------------------
if [[ -z "${SCENE_JSON}" ]]; then
  SCENE_JSON="${OUTPUT_ROOT}/scene_test.json"
  SCENE_CSV="${OUTPUT_ROOT}/scene_test.csv"
  echo "============================================================"
  echo "[Fusion Step 1] Predict scene labels for test set"
  echo "============================================================"
  python scene_tools/infer_scene_convnext.py \
    --input-dir "${TEST_DIR}" \
    --checkpoint "${SCENE_CKPT}" \
    --output-json "${SCENE_JSON}" \
    --output-csv "${SCENE_CSV}" \
    --batch-size 128 \
    --num-workers 8 \
    --device "${DEVICE}" \
    --amp-dtype "${AMP_DTYPE}"
fi

# ------------------------------------------------------------------
# Step 2: Inference with pseudo-finetuned model.
# ------------------------------------------------------------------
PSEUDO_PRED_DIR="${OUTPUT_ROOT}/pseudo_model_predictions"
PSEUDO_MODEL_NAME="pseudo_${VERSION_TAG}"

echo ""
echo "============================================================"
echo "[Fusion Step 2] JiT inference with pseudo-finetuned model"
echo "Checkpoint: ${PSEUDO_CKPT}"
echo "============================================================"

tta_args=()
if [[ "${PSEUDO_TTA_HFLIP}" == "1" ]]; then
  tta_args+=(--tta-hflip)
fi

python submit_jit.py \
  --input-dir "${TEST_DIR}" \
  --checkpoint "${PSEUDO_CKPT}" \
  --ckpt_type "${PSEUDO_CKPT_TYPE}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-name "${PSEUDO_MODEL_NAME}" \
  --state-key "${PSEUDO_STATE_KEY}" \
  --use-scene \
  --scene-json "${SCENE_JSON}" \
  --steps "${PSEUDO_STEPS}" \
  --stride "${PSEUDO_STRIDE}" \
  --tile-batch-size "${TILE_BATCH_SIZE}" \
  --scales "${PSEUDO_SCALES}" \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  "${tta_args[@]}"

# Find the actual prediction directory.
PSEUDO_PRED_DIR=$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${PSEUDO_MODEL_NAME}_*" | sort | tail -n 1)
if [[ ! -d "${PSEUDO_PRED_DIR}" ]]; then
  echo "Could not find pseudo model prediction directory" >&2; exit 2
fi
echo "Pseudo model predictions: ${PSEUDO_PRED_DIR}"

# ------------------------------------------------------------------
# Step 3: Fuse with original ensemble.
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[Fusion Step 3] Weighted fusion"
echo "W_ENS=${W_ENS} (original ensemble) + (1-W_ENS)=$(python -c "print(1-${W_ENS})") (pseudo model)"
echo "============================================================"

if [[ -n "${ENSEMBLE_DIRS}" ]]; then
  # Fuse with provided ensemble directories.
  FUSION_NAME="fusion_${VERSION_TAG}_w${W_ENS/./p}"

  if [[ "${W_ENS}" == "1.0" ]]; then
    # Ensemble-only (no pseudo model).
    echo "W_ENS=1.0, generating ensemble-only submission."
    FUSION_DIR="${OUTPUT_ROOT}/${FUSION_NAME}"
    # shellcheck disable=SC2086
    python tools/ensemble_submission_dirs.py \
      --input-dirs ${ENSEMBLE_DIRS} \
      --fusion "${ENSEMBLE_FUSION}" \
      --output-dir "${FUSION_DIR}" \
      --archive-path "${OUTPUT_ROOT}/${FUSION_NAME}.zip" \
      --model-name "${FUSION_NAME}" \
      --notes "fusion_${VERSION_TAG}: ensemble only; method=${ENSEMBLE_FUSION}"
  elif [[ "${W_ENS}" == "0.0" ]]; then
    # Pseudo model only.
    echo "W_ENS=0.0, generating pseudo-model-only submission."
    FUSION_DIR="${OUTPUT_ROOT}/${FUSION_NAME}"
    python tools/ensemble_submission_dirs.py \
      --input-dirs "${PSEUDO_PRED_DIR}" \
      --output-dir "${FUSION_DIR}" \
      --archive-path "${OUTPUT_ROOT}/${FUSION_NAME}.zip" \
      --model-name "${FUSION_NAME}" \
      --notes "fusion_${VERSION_TAG}: pseudo model only"
  else
    # Weighted average: first average the ensemble dirs, then fuse with pseudo.
    # Build ensemble input args.
    ENSEMBLE_INPUT_DIRS=""
    for d in ${ENSEMBLE_DIRS}; do
      ENSEMBLE_INPUT_DIRS="${ENSEMBLE_INPUT_DIRS} ${d}"
    done

    # Step 3a: Average all ensemble dirs into one.
    ENSEMBLE_AVG_DIR="${OUTPUT_ROOT}/ensemble_averaged"
    mkdir -p "${ENSEMBLE_AVG_DIR}"
    # shellcheck disable=SC2086
    python tools/ensemble_submission_dirs.py \
      --input-dirs ${ENSEMBLE_DIRS} \
      --fusion "${ENSEMBLE_FUSION}" \
      --output-dir "${ENSEMBLE_AVG_DIR}" \
      --archive-path "${OUTPUT_ROOT}/ensemble_averaged.zip"

    # Step 3b: Fuse averaged ensemble with pseudo model.
    # Need to find the actual prediction subdir inside ensemble_averaged...
    # ensemble_submission_dirs outputs directly to --output-dir.
    # Actually, it outputs to --archive-path as ZIP and --output-dir as PNGs.
    # Let me just re-use the PNG dir approach.

    FUSION_DIR="${OUTPUT_ROOT}/${FUSION_NAME}"
    mkdir -p "${FUSION_DIR}"

    # shellcheck disable=SC2086
    python tools/ensemble_submission_dirs.py \
      --input-dirs ${ENSEMBLE_AVG_DIR} "${PSEUDO_PRED_DIR}" \
      --weights "${W_ENS},$(python -c "print(1-${W_ENS})")" \
      --output-dir "${FUSION_DIR}" \
      --archive-path "${OUTPUT_ROOT}/${FUSION_NAME}.zip" \
      --model-name "${FUSION_NAME}" \
      --notes "fusion_${VERSION_TAG}: W_ENS=${W_ENS} * ensemble + (1-W_ENS) * pseudo_refiner"

    echo "Fusion submission: ${OUTPUT_ROOT}/${FUSION_NAME}.zip"
  fi
else
  echo "Warning: ENSEMBLE_DIRS not set. Only pseudo model output generated."
  echo "To fuse, copy the desired ensemble prediction dirs and run:"
  echo "  python tools/ensemble_submission_dirs.py --input-dirs <ensemble_dirs> ${PSEUDO_PRED_DIR} --weights ${W_ENS},$(python -c "print(1-${W_ENS})") ..."
fi

echo ""
echo "============================================================"
echo "Fusion complete."
echo ""
echo "Outputs:"
echo "  Pseudo model preds:  ${PSEUDO_PRED_DIR}"
echo "  Fusion ZIP:          ${OUTPUT_ROOT}/fusion_${VERSION_TAG}_w*.zip"
echo ""
echo "To evaluate locally:"
echo "  python tools/evaluate_submission_dir.py --prediction-dir <dir> --clear-dir <gt_dir>"
echo "============================================================"
