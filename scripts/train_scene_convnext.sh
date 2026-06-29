#!/usr/bin/env bash
set -euo pipefail

# Train a ConvNeXt scene classifier from a JiT-compatible filename-to-class JSON.
#
# Examples:
#   DATA_PATH=/data/RainDrop_Train bash scripts/train_scene_convnext.sh
#   LABELS_JSON=/data/RainDrop_Train/Drop_dn_2scene.json NUM_CLASSES=2 CLASS_NAMES=night,day bash scripts/train_scene_convnext.sh

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
IMAGE_DIR=${IMAGE_DIR:-${DATA_PATH}/Drop}
LABELS_JSON=${LABELS_JSON:-${DATA_PATH}/Drop_scen_pred.json}
OUTPUT_DIR=${OUTPUT_DIR:-run/scene_convnext}

MODEL=${MODEL:-convnext_tiny}
NUM_CLASSES=${NUM_CLASSES:-0}
CLASS_NAMES=${CLASS_NAMES:-}
IMAGE_SIZE=${IMAGE_SIZE:-448}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
VAL_FRACTION=${VAL_FRACTION:-0.1}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
AMP_DTYPE=${AMP_DTYPE:-auto}
RESUME=${RESUME:-}
PRETRAINED=${PRETRAINED:-1}
CLASS_WEIGHTS=${CLASS_WEIGHTS:-1}

extra_args=()
if [[ -n "${CLASS_NAMES}" ]]; then
  extra_args+=(--class-names "${CLASS_NAMES}")
fi
if [[ -n "${RESUME}" ]]; then
  extra_args+=(--resume "${RESUME}")
fi
if [[ "${PRETRAINED}" != "1" ]]; then
  extra_args+=(--no-pretrained)
fi
if [[ "${CLASS_WEIGHTS}" != "1" ]]; then
  extra_args+=(--no-class-weights)
fi

python scene_tools/train_scene_convnext.py \
  --data-root "${DATA_PATH}" \
  --image-dir "${IMAGE_DIR}" \
  --labels-json "${LABELS_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --num-classes "${NUM_CLASSES}" \
  --image-size "${IMAGE_SIZE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --val-fraction "${VAL_FRACTION}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  "${extra_args[@]}"
