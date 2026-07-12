#!/usr/bin/env bash
set -euo pipefail

# Shell entry point for aligned multi-frame scene consensus post-processing.
# The NumPy/Pillow implementation remains in tools/scene_consensus_postprocess.py.
#
# Single-model example:
#   bash scripts/deadline2d_post/scene_consensus_postprocess.sh \
#     --pred-dirs D:/predictions/ft_edge_freq \
#     --output-dir D:/outputs/ft_edge_freq_scene_median \
#     --archive-path D:/outputs/ft_edge_freq_scene_median.zip \
#     --expected-count 592 --scene-weight 1.0
#
# Multi-model example:
#   bash scripts/deadline2d_post/scene_consensus_postprocess.sh \
#     --pred-dirs D:/predictions/jit_b D:/predictions/ft_edge_freq \
#     --model-weights 0.5,0.5 \
#     --output-dir D:/outputs/jit_b_msdt_scene_median \
#     --archive-path D:/outputs/jit_b_msdt_scene_median.zip \
#     --expected-count 592 --scene-weight 1.0

JIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
ENGINE="${JIT_ROOT}/tools/scene_consensus_postprocess.py"

if [[ ! -f "${ENGINE}" ]]; then
  echo "Missing scene-consensus engine: ${ENGINE}" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  "${PYTHON_BIN}" "${ENGINE}" --help
  exit 0
fi

exec "${PYTHON_BIN}" "${ENGINE}" "$@"
