#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Smoke test for the pseudo-label fine-tuning pipeline.
#
# Verifies (in order):
#   1. Pseudo-label generation on a 4-image subset.
#   2. Pseudo validation against GT.
#   3. V1 training — 1 epoch, 5 steps.
#   4. V2 training — 1 epoch, 5 steps.
#   5. V3 training — 1 epoch, 5 steps (only if UNFREEZE_BLOCKS >= 1).
#   6. Single-image inference with a pseudo-finetuned checkpoint.
#   7. Fusion of ensemble + pseudo model on the 4-image subset.
#
# Run this BEFORE any real training to catch config / code issues early.
# Expected runtime: ~2-5 minutes.
#
# Usage:
#   DATA_ROOT=D:/zhl/data/eccv_dn \
#   CKPT=run/train/best_msdt_refiner/16 \
#   TEACHER_DIRS="submissions/model1 submissions/model2" \
#   bash scripts/smoke_test_pseudo.sh
# ============================================================================

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

CKPT=${CKPT:-run/train/focus_2scene_msdt_refiner_h_1xA100_48g/h16_refiner_c1/16}
SCENE_CKPT=${SCENE_CKPT:-run/scene_convnext_focus_2scene_v1/checkpoint-best.pth}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}
TEACHER_DIRS=${TEACHER_DIRS:-}

GPU=${GPU:-0}
DEVICE=${DEVICE:-cuda:0}
SMOKE_ROOT=${SMOKE_ROOT:-run/smoke_test/pseudo}
UNFREEZE_BLOCKS=${UNFREEZE_BLOCKS:-4}

# Clean start
rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}"

# ---- tiny 4-image val subset ------------------------------------------
SUBSET_ROOT="${SMOKE_ROOT}/subset"
echo "=== [1/7] Creating 4-image subset ==="
python tools/make_fixed_val_subset.py \
  --data-root "${DATA_PATH}" \
  --output-root "${SUBSET_ROOT}" \
  --num-samples 4 \
  --seed 42 \
  --scene-json "${SCENE_FOCUS_2_PATH}" \
  --output-scene-json "${SUBSET_ROOT}/Drop_focus_2scene.json" \
  --overwrite

ls -la "${SUBSET_ROOT}/Drop/"  | head -10
echo "Subset created: ${SUBSET_ROOT}"

# ---- generate pseudo labels on the subset -----------------------------
PSEUDO_ROOT="${SMOKE_ROOT}/pseudo"
echo ""
echo "=== [2/7] Generating pseudo labels (4 images) ==="

if [[ -z "${TEACHER_DIRS}" ]]; then
  echo "TEACHER_DIRS not set — running single-checkpoint inference first to create a teacher dir..."
  TEACHER_SINGLE="${SMOKE_ROOT}/teacher_single"
  mkdir -p "${TEACHER_SINGLE}"
  python submit_jit.py \
    --input-dir "${SUBSET_ROOT}/Drop" \
    --checkpoint "${CKPT}" \
    --ckpt_type best \
    --output-root "${SMOKE_ROOT}/teacher" \
    --model-name "smoke_teacher" \
    --state-key auto \
    --use-scene \
    --scene-checkpoint "${SCENE_CKPT}" \
    --scene-output-json "${SMOKE_ROOT}/scene_smoke.json" \
    --steps 1 --stride 256 --tile-batch-size 2 \
    --device "${DEVICE}" \
    --tta-hflip

  TEACHER_DIR=$(find "${SMOKE_ROOT}/teacher" -maxdepth 1 -type d -name "smoke_teacher_*" | sort | tail -n 1)
  if [[ ! -d "${TEACHER_DIR}" ]]; then
    echo "FATAL: Could not create teacher predictions" >&2; exit 2
  fi
  # Create a second "pseudo" teacher dir by copying and adding noise (simulates ensemble diversity).
  TEACHER_DIR2="${SMOKE_ROOT}/teacher/smoke_teacher_2"
  mkdir -p "${TEACHER_DIR2}"
  for f in "${TEACHER_DIR}"/*.png; do
    python -c "
from PIL import Image
import numpy as np
img = np.asarray(Image.open('${f}').convert('RGB'), dtype=np.float32)
# tiny perturbation for ensemble diversity
noise = np.random.randn(*img.shape).astype(np.float32) * 0.5
Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8)).save('${TEACHER_DIR2}/$(basename ${f})')
"
  done
  TEACHER_DIRS="${TEACHER_DIR} ${TEACHER_DIR2}"
fi

# shellcheck disable=SC2086
python tools/pseudo_label.py generate \
  --input-dir "${SUBSET_ROOT}/Drop" \
  --pred-dirs ${TEACHER_DIRS} \
  --output-dir "${PSEUDO_ROOT}/PseudoGT" \
  --mask-dir "${PSEUDO_ROOT}/masks" \
  --fusion trimmed_mean \
  --tta-variance-csv "${PSEUDO_ROOT}/tta_variance.csv"

echo "Pseudo images:"
ls -la "${PSEUDO_ROOT}/PseudoGT/" | head -10
echo "Masks:"
ls -la "${PSEUDO_ROOT}/masks/" | head -10

# ---- validate pseudo vs GT --------------------------------------------
echo ""
echo "=== [3/7] Validating pseudo strategies ==="
# shellcheck disable=SC2086
python tools/pseudo_label.py validate \
  --input-dir "${SUBSET_ROOT}/Drop" \
  --pred-dirs ${TEACHER_DIRS} \
  --gt-clear-dir "${SUBSET_ROOT}/Clear" \
  --output-dir "${SMOKE_ROOT}/validate" \
  --fusion trimmed_mean

echo ""
echo "Validation results:"
cat "${SMOKE_ROOT}/validate/pseudo_validation.csv" 2>/dev/null || echo "(CSV not found — check logs above)"

# ---- filter pseudo samples --------------------------------------------
echo ""
echo "=== [4/7] Filtering pseudo samples ==="
python tools/pseudo_label.py filter \
  --tta-variance-csv "${PSEUDO_ROOT}/tta_variance.csv" \
  --output-dir "${PSEUDO_ROOT}" \
  --pseudo-dir "${PSEUDO_ROOT}/PseudoGT" \
  --mask-dir "${PSEUDO_ROOT}/masks" \
  --filter-top-pct 10.0

# ---- V1 smoke training (refiner only) ---------------------------------
echo ""
echo "=== [5/7] V1 smoke training (1 epoch, 5 steps) ==="

V1_OUT="${SMOKE_ROOT}/train_v1"
SMOKE=1 VERSION=v1 \
  GPU="${GPU}" \
  DATA_PATH="${DATA_PATH}" \
  VAL_DATA_PATH="${SUBSET_ROOT}" \
  CKPT="${CKPT}" \
  SCENE_FOCUS_2_PATH="${SUBSET_ROOT}/Drop_focus_2scene.json" \
  PSEUDO_DATA_PATH="${PSEUDO_ROOT}/PseudoGT" \
  PSEUDO_MASK_DIR="${PSEUDO_ROOT}/masks" \
  PSEUDO_BATCH=2 \
  OUT_ROOT="${V1_OUT}" \
  MASTER_PORT_BASE=31920 \
  bash scripts/train_pseudo.sh

echo "V1 smoke test passed.  Checkpoint: ${V1_OUT}/16/checkpoint-last.pth"
ls -la "${V1_OUT}/16/" | grep checkpoint

# ---- V2 smoke training (filtered refiner only) ------------------------
echo ""
echo "=== [6/7] V2 smoke training (1 epoch, 5 steps) ==="

V2_OUT="${SMOKE_ROOT}/train_v2"
SMOKE=1 VERSION=v2 \
  GPU="${GPU}" \
  DATA_PATH="${DATA_PATH}" \
  VAL_DATA_PATH="${SUBSET_ROOT}" \
  CKPT="${CKPT}" \
  SCENE_FOCUS_2_PATH="${SUBSET_ROOT}/Drop_focus_2scene.json" \
  PSEUDO_DATA_PATH="${PSEUDO_ROOT}/PseudoGT" \
  PSEUDO_MASK_DIR="${PSEUDO_ROOT}/masks" \
  PSEUDO_DATA_FILTERED="${PSEUDO_ROOT}/PseudoGT_filtered" \
  PSEUDO_MASK_FILTERED="${PSEUDO_ROOT}/masks_filtered" \
  PSEUDO_BATCH=2 \
  OUT_ROOT="${V2_OUT}" \
  MASTER_PORT_BASE=31922 \
  bash scripts/train_pseudo.sh

echo "V2 smoke test passed."

# ---- V3 smoke training (joint unfreeze) -------------------------------
if [[ "${UNFREEZE_BLOCKS}" -ge 1 ]]; then
  echo ""
  echo "=== [7/7] V3 smoke training (1 epoch, 5 steps, unfreeze ${UNFREEZE_BLOCKS} blocks) ==="

  V3_OUT="${SMOKE_ROOT}/train_v3"
  SMOKE=1 VERSION=v3 \
    GPU="${GPU}" \
    DATA_PATH="${DATA_PATH}" \
    VAL_DATA_PATH="${SUBSET_ROOT}" \
    CKPT="${CKPT}" \
    SCENE_FOCUS_2_PATH="${SUBSET_ROOT}/Drop_focus_2scene.json" \
    PSEUDO_DATA_PATH="${PSEUDO_ROOT}/PseudoGT" \
    PSEUDO_MASK_DIR="${PSEUDO_ROOT}/masks" \
    PSEUDO_BATCH=2 \
    UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS}" \
    OUT_ROOT="${V3_OUT}" \
    MASTER_PORT_BASE=31924 \
    bash scripts/train_pseudo.sh

  echo "V3 smoke test passed."
else
  echo ""
  echo "=== [7/7] V3 smoke test SKIPPED (UNFREEZE_BLOCKS=0) ==="
fi

# ---- single-image inference smoke test --------------------------------
echo ""
echo "=== Inference smoke test (1 image) ==="
# Pick one checkpoint from V1
V1_CKPT="${V1_OUT}/16/checkpoint-last.pth"
if [[ -f "${V1_CKPT}" ]]; then
  # Create a single-image test dir
  SINGLE_DIR="${SMOKE_ROOT}/single_test"
  mkdir -p "${SINGLE_DIR}/Drop"
  first_img=$(ls "${SUBSET_ROOT}/Drop/" | head -1)
  cp "${SUBSET_ROOT}/Drop/${first_img}" "${SINGLE_DIR}/Drop/"

  python submit_jit.py \
    --input-dir "${SINGLE_DIR}/Drop" \
    --checkpoint "${V1_CKPT}" \
    --ckpt_type last \
    --output-root "${SMOKE_ROOT}/infer" \
    --model-name "smoke_infer" \
    --state-key model \
    --use-scene \
    --scene-json "${SUBSET_ROOT}/Drop_focus_2scene.json" \
    --steps 1 --stride 256 --tile-batch-size 1 \
    --device "${DEVICE}" \
    --tta-hflip

  INFER_DIR=$(find "${SMOKE_ROOT}/infer" -maxdepth 1 -type d -name "smoke_infer_*" | sort | tail -n 1)
  if [[ -d "${INFER_DIR}" ]]; then
    echo "Inference output: $(ls "${INFER_DIR}" | wc -l) PNG files"
  fi
else
  echo "V1 checkpoint not found at ${V1_CKPT}, skipping inference test."
fi

# ---- fusion smoke test ------------------------------------------------
echo ""
echo "=== Fusion smoke test ==="
if [[ -d "${INFER_DIR:-}" && -d "${TEACHER_DIR:-}" ]]; then
  FUSION_DIR="${SMOKE_ROOT}/fusion"
  mkdir -p "${FUSION_DIR}"
  python tools/ensemble_submission_dirs.py \
    --input-dirs "${TEACHER_DIR}" "${INFER_DIR}" \
    --weights 0.70,0.30 \
    --output-dir "${FUSION_DIR}" \
    --archive-path "${SMOKE_ROOT}/fusion_smoke.zip" \
    --model-name "smoke_fusion"
  echo "Fusion ZIP: ${SMOKE_ROOT}/fusion_smoke.zip"
else
  echo "Skipping fusion test (missing inference or teacher dirs)."
fi

# ---- done -------------------------------------------------------------
echo ""
echo "============================================================"
echo "SMOKE TEST COMPLETE"
echo "============================================================"
echo ""
echo "All artifacts under: ${SMOKE_ROOT}"
echo ""
echo "Next: run real pseudo generation + V1 training:"
echo "  TEACHER_DIRS='<your ensemble dirs>' bash scripts/generate_and_validate_pseudo.sh"
echo "  VERSION=v1 bash scripts/train_pseudo.sh"
echo "============================================================"
