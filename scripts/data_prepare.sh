#!/usr/bin/env bash

set -euo pipefail

# DATA_ROOT="/root/huilin/data/eccv_dn"
# DATA_ROOT="/scrinvme/huilin/tp/eccv_dn"
DATA_ROOT="D:\zhl\data\eccv_dn"

python tools/data_tools.py copy \
  --day-root "${DATA_ROOT}/DayRainDrop_Train" \
  --night-root "${DATA_ROOT}/NightRainDrop_Train" \
  --dst-root "${DATA_ROOT}/RainDrop_Train"

python tools/data_tools.py pseudo-scene \
  --data-root "${DATA_ROOT}/RainDrop_Train"