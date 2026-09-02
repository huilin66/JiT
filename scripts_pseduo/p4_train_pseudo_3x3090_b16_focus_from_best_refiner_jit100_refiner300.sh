#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/p4_train_pseudo_single_gpu_b16_focus_from_best_refiner_jit100_refiner300.sh" "$@"
