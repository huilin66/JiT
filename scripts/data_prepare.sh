#!/usr/bin/env bash

set -euo pipefail

# DATA_ROOT="/root/huilin/data/eccv_dn"
# DATA_ROOT="/scrinvme/huilin/tp/eccv_dn"
# DATA_ROOT="D:\zhl\data\eccv_dn"
SCENE_NUM_WORKERS=${SCENE_NUM_WORKERS:-8}
EXTRACT_SAMPLES=${EXTRACT_SAMPLES:-0}
SAMPLE_ROOT=${SAMPLE_ROOT:-${DATA_ROOT}/sample_check}
SAMPLE_SCENE_CSV=${SAMPLE_SCENE_CSV:-${DATA_ROOT}/sample_check.csv}

extract_first_images() {
  local source_dir="$1"
  local dest_dir="$2"
  local csv_mode="$3"

  if [[ ! -d "${source_dir}" ]]; then
    echo "[samples] skip missing directory: ${source_dir}"
    return
  fi

  python tools/data_tools.py samples \
    --source-dir "${source_dir}" \
    --dest-dir "${dest_dir}" \
    --recursive \
    --keep-tree \
    --scene-csv "${SAMPLE_SCENE_CSV}" \
    --scene-csv-mode "${csv_mode}"
}

if [[ "${EXTRACT_SAMPLES}" == "1" ]]; then
  extract_first_images "${DATA_ROOT}/DayRainDrop_Train/Drop" "${SAMPLE_ROOT}/day/drop" "write"
  extract_first_images "${DATA_ROOT}/NightRainDrop_Train/Drop" "${SAMPLE_ROOT}/night/drop" "append"
fi

python tools/data_tools.py copy \
  --day-root "${DATA_ROOT}/DayRainDrop_Train" \
  --night-root "${DATA_ROOT}/NightRainDrop_Train" \
  --dst-root "${DATA_ROOT}/RainDrop_Train"

python tools/data_tools.py pseudo-scene \
  --data-root "${DATA_ROOT}/RainDrop_Train" \
  --num-workers "${SCENE_NUM_WORKERS}"

python tools/data_tools.py pseudo-day-night-scene \
  --data-root "${DATA_ROOT}/RainDrop_Train" \
  --output-json "${DATA_ROOT}/RainDrop_Train/Drop_dn_2scene.json" \
  --num-workers "${SCENE_NUM_WORKERS}"

python tools/data_tools.py pseudo-focus-scene \
  --data-root "${DATA_ROOT}/RainDrop_Train" \
  --scene-count 2 \
  --output-json "${DATA_ROOT}/RainDrop_Train/Drop_focus_2scene.json" \
  --num-workers "${SCENE_NUM_WORKERS}"

python tools/data_tools.py pseudo-focus-scene \
  --data-root "${DATA_ROOT}/RainDrop_Train" \
  --scene-count 4 \
  --output-json "${DATA_ROOT}/RainDrop_Train/Drop_focus_4scene.json" \
  --num-workers "${SCENE_NUM_WORKERS}"
