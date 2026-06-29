#!/usr/bin/env bash

set -euo pipefail

# DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
# DATA_ROOT="/root/huilin/data/eccv_dn"
# DATA_ROOT="/scrinvme/huilin/tp/eccv_dn"
DATA_ROOT="D:\zhl\data\eccv_dn"


SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}
SAMPLE_ROOT=${SAMPLE_ROOT:-${DATA_ROOT}/sample_check}
SAMPLE_SCENE_CSV=${SAMPLE_SCENE_CSV:-${DATA_ROOT}/sample_check.csv}


python tools/data_tools.py drop-scene-samples \
  --data-root "${DATA_ROOT}" \
  --dest-root "${SAMPLE_ROOT}" \
  --scene-csv "${SAMPLE_SCENE_CSV}"


# python tools/data_tools.py copy \
#   --day-root "${DATA_ROOT}/DayRainDrop_Train" \
#   --night-root "${DATA_ROOT}/NightRainDrop_Train" \
#   --dst-root "${DATA_ROOT}/RainDrop_Train"

# python tools/data_tools.py pseudo-scene \
#   --data-root "${DATA_ROOT}/RainDrop_Train" \
#   --num-workers "${SCENE_NUM_WORKERS}"

# python tools/data_tools.py pseudo-day-night-scene \
#   --data-root "${DATA_ROOT}/RainDrop_Train" \
#   --output-json "${DATA_ROOT}/RainDrop_Train/Drop_dn_2scene.json" \
#   --num-workers "${SCENE_NUM_WORKERS}"

# python tools/data_tools.py pseudo-focus-scene \
#   --data-root "${DATA_ROOT}/RainDrop_Train" \
#   --scene-count 2 \
#   --output-json "${DATA_ROOT}/RainDrop_Train/Drop_focus_2scene.json" \
#   --num-workers "${SCENE_NUM_WORKERS}"

# python tools/data_tools.py pseudo-focus-scene \
#   --data-root "${DATA_ROOT}/RainDrop_Train" \
#   --scene-count 4 \
#   --output-json "${DATA_ROOT}/RainDrop_Train/Drop_focus_4scene.json" \
#   --num-workers "${SCENE_NUM_WORKERS}"
