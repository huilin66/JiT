#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Unified pseudo-label fine-tuning script (V1 / V2 / V3).
#
# V1 (conservative, low risk):
#   Refiner-only.  Real:pseudo=70:30, pseudo_loss_weight=0.25, epochs=60.
#   LPIPS from epoch 0.  Recommended FIRST run.
#
# V2 (medium risk):
#   Refiner-only with filtered pseudo (top 10% high-variance dropped).
#   Real:pseudo=60:40, pseudo_loss_weight=0.35, epochs=50.
#   Higher edge/freq loss.  Run only if V1 shows improvement.
#
# V3 (high risk):
#   Joint JiT (last N blocks unfrozen) + Refiner.
#   Real:pseudo=60:40, pseudo_loss_weight=0.25, epochs=30.
#   Very low LR, reduced batch.  Run only if V1/V2 show improvement.
#
# Usage:
#   VERSION=v1 DATA_ROOT=... CKPT=... PSEUDO_DATA_PATH=... bash scripts/train_pseudo.sh
#   VERSION=v2 UNFREEZE_BLOCKS=4  bash scripts/train_pseudo.sh
#   VERSION=all                   bash scripts/train_pseudo.sh   # run V1→V2→V3 in sequence
#
# Smoke test:
#   SMOKE=1 VERSION=v1 bash scripts/train_pseudo.sh
# ============================================================================

# ---- Quick-reference presets ------------------------------------------
# These can be overridden via environment variables.

VERSION=${VERSION:-v1}
SMOKE=${SMOKE:-0}

# Shared paths ----------------------------------------------------------
export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPU=${GPU:-0}
NPROC=${NPROC:-1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30920}

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}

CKPT=${CKPT:-run/train/focus_2scene_msdt_refiner_h_1xA100_48g/h16_refiner_c1/16}
RESUME_STATE_KEY=${RESUME_STATE_KEY:-model_ema2}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

PSEUDO_DATA_PATH=${PSEUDO_DATA_PATH:-${DATA_PATH}/pseudo_labels/PseudoGT}
PSEUDO_MASK_DIR=${PSEUDO_MASK_DIR:-${DATA_PATH}/pseudo_labels/masks}
PSEUDO_DATA_FILTERED=${PSEUDO_DATA_FILTERED:-${DATA_PATH}/pseudo_labels/PseudoGT_filtered}
PSEUDO_MASK_FILTERED=${PSEUDO_MASK_FILTERED:-${DATA_PATH}/pseudo_labels/masks_filtered}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
REFINER_BASE_DIM=${REFINER_BASE_DIM:-32}
REFINER_NUM_BLOCKS=${REFINER_NUM_BLOCKS:-2}
REFINER_USE_FREQUENCY=${REFINER_USE_FREQUENCY:-1}

# ---- Version-specific defaults ----------------------------------------
# Each version block below can be tuned individually.  Override any value
# by exporting it before calling the script.

declare -A CFG

# --- V1: conservative --------------------------------------------------
CFG[v1,desc]="V1 — Refiner-only, 70:30, pseudo_w=0.25, 60ep (conservative)"
CFG[v1,out_root]="${OUT_ROOT:-run/train/v1_pseudo_refiner}"
CFG[v1,pseudo_path]="${PSEUDO_DATA_PATH}"
CFG[v1,pseudo_mask]="${PSEUDO_MASK_DIR}"
CFG[v1,pseudo_ratio]="${PSEUDO_RATIO:-0.30}"
CFG[v1,pseudo_lw]="${PSEUDO_LOSS_WEIGHT:-0.25}"
CFG[v1,freeze_jit]="1"
CFG[v1,unfreeze_blocks]="0"
CFG[v1,batch]="${PSEUDO_BATCH:-16}"
CFG[v1,lr]="${PSEUDO_LR:-5e-6}"
CFG[v1,epochs]="${PSEUDO_EPOCHS:-60}"
CFG[v1,warmup]="${PSEUDO_WARMUP:-3}"
CFG[v1,save_epoch_freq]="${SAVE_EPOCH_FREQ:-20}"
CFG[v1,lpips_w]="${LPIPS_WEIGHT:-1.0}"
CFG[v1,edge_w]="${EDGE_WEIGHT:-0.10}"
CFG[v1,freq_w]="${FREQ_WEIGHT:-0.10}"
CFG[v1,lpips_warmup]="${LPIPS_WARMUP:-0.0}"
CFG[v1,refiner_max_res]="${REFINER_MAX_RESIDUAL:-0.35}"

# --- V2: medium risk (filtered pseudo) ---------------------------------
CFG[v2,desc]="V2 — Refiner-only FILTERED, 60:40, pseudo_w=0.35, 50ep (medium risk)"
CFG[v2,out_root]="${OUT_ROOT:-run/train/v2_pseudo_refiner_filtered}"
CFG[v2,pseudo_path]="${PSEUDO_DATA_FILTERED}"
CFG[v2,pseudo_mask]="${PSEUDO_MASK_FILTERED}"
CFG[v2,pseudo_ratio]="${PSEUDO_RATIO:-0.40}"
CFG[v2,pseudo_lw]="${PSEUDO_LOSS_WEIGHT:-0.35}"
CFG[v2,freeze_jit]="1"
CFG[v2,unfreeze_blocks]="0"
CFG[v2,batch]="${PSEUDO_BATCH:-16}"
CFG[v2,lr]="${PSEUDO_LR:-5e-6}"
CFG[v2,epochs]="${PSEUDO_EPOCHS:-50}"
CFG[v2,warmup]="${PSEUDO_WARMUP:-3}"
CFG[v2,save_epoch_freq]="${SAVE_EPOCH_FREQ:-10}"
CFG[v2,lpips_w]="${LPIPS_WEIGHT:-1.0}"
CFG[v2,edge_w]="${EDGE_WEIGHT:-0.15}"
CFG[v2,freq_w]="${FREQ_WEIGHT:-0.15}"
CFG[v2,lpips_warmup]="${LPIPS_WARMUP:-0.0}"
CFG[v2,refiner_max_res]="${REFINER_MAX_RESIDUAL:-0.35}"

# --- V3: high risk (joint unfreeze) ------------------------------------
CFG[v3,desc]="V3 — Joint JiT (last ${UNFREEZE_BLOCKS:-4} blk) + Refiner, 60:40, pseudo_w=0.25, 30ep (HIGH RISK)"
CFG[v3,out_root]="${OUT_ROOT:-run/train/v3_pseudo_joint}"
CFG[v3,pseudo_path]="${PSEUDO_DATA_PATH}"
CFG[v3,pseudo_mask]="${PSEUDO_MASK_DIR}"
CFG[v3,pseudo_ratio]="${PSEUDO_RATIO:-0.40}"
CFG[v3,pseudo_lw]="${PSEUDO_LOSS_WEIGHT:-0.25}"
CFG[v3,freeze_jit]="0"
CFG[v3,unfreeze_blocks]="${UNFREEZE_BLOCKS:-4}"
CFG[v3,batch]="${PSEUDO_BATCH:-8}"
CFG[v3,lr]="${PSEUDO_LR:-5e-6}"
CFG[v3,lr_jit]="${LR_JIT_LAST_BLOCKS:-1e-6}"
CFG[v3,epochs]="${PSEUDO_EPOCHS:-30}"
CFG[v3,warmup]="${PSEUDO_WARMUP:-2}"
CFG[v3,save_epoch_freq]="${SAVE_EPOCH_FREQ:-10}"
CFG[v3,lpips_w]="${LPIPS_WEIGHT:-0.5}"
CFG[v3,edge_w]="${EDGE_WEIGHT:-0.10}"
CFG[v3,freq_w]="${FREQ_WEIGHT:-0.10}"
CFG[v3,lpips_warmup]="${LPIPS_WARMUP:-0.0}"
CFG[v3,refiner_max_res]="${REFINER_MAX_RESIDUAL:-0.25}"

# ========================================================================
# Smoke-test overrides (short run to verify pipeline is not broken)
# ========================================================================
if [[ "${SMOKE}" == "1" ]]; then
  for ver in v1 v2 v3; do
    CFG[${ver},epochs]="1"
    CFG[${ver},warmup]="0"
    CFG[${ver},out_root]="${CFG[${ver},out_root]}_smoke"
  done
  SMOKE_MAX_STEPS="--max_train_steps 5"
  SMOKE_NO_EVAL=""
  echo "=== SMOKE TEST MODE: 1 epoch, 5 steps ==="
else
  SMOKE_MAX_STEPS=""
  SMOKE_NO_EVAL=""
fi

# ========================================================================
# Helper
# ========================================================================
run_one() {
  local ver="$1"
  local port="$2"

  local desc="${CFG[${ver},desc]}"
  local out_root="${CFG[${ver},out_root]}"
  local pseudo_path="${CFG[${ver},pseudo_path]}"
  local pseudo_mask="${CFG[${ver},pseudo_mask]}"
  local pseudo_ratio="${CFG[${ver},pseudo_ratio]}"
  local pseudo_lw="${CFG[${ver},pseudo_lw]}"
  local freeze_jit="${CFG[${ver},freeze_jit]}"
  local unfreeze_blocks="${CFG[${ver},unfreeze_blocks]}"
  local batch="${CFG[${ver},batch]}"
  local lr="${CFG[${ver},lr]}"
  local epochs="${CFG[${ver},epochs]}"
  local warmup="${CFG[${ver},warmup]}"
  local save_epoch_freq="${CFG[${ver},save_epoch_freq]}"
  local lr_jit="${CFG[${ver},lr_jit]:-0}"
  local lpips_w="${CFG[${ver},lpips_w]}"
  local edge_w="${CFG[${ver},edge_w]}"
  local freq_w="${CFG[${ver},freq_w]}"
  local lpips_warmup="${CFG[${ver},lpips_warmup]}"
  local refiner_max_res="${CFG[${ver},refiner_max_res]}"
  local output_dir="${out_root}/16"

  echo ""
  echo "============================================================"
  echo "[${ver^^}] ${desc}"
  echo "============================================================"
  echo "Checkpoint:       ${CKPT}  (state: ${RESUME_STATE_KEY})"
  echo "Pseudo data:      ${pseudo_path}"
  echo "Pseudo masks:     ${pseudo_mask}"
  echo "Real:pseudo =     $(python -c "r=${pseudo_ratio}; print(f'{1-r:.0%}:{r:.0%}')")"
  echo "Pseudo loss w:    ${pseudo_lw}"
  echo "Freeze JiT:       ${freeze_jit}  |  Unfreeze blocks: ${unfreeze_blocks}"
  echo "LR: ${lr}  |  Batch: ${batch}  |  Epochs: ${epochs}  |  Warmup: ${warmup}"
  if [[ "${unfreeze_blocks}" -gt 0 ]]; then
    echo "JiT LR: ${lr_jit}  |  Numbered checkpoint freq: ${save_epoch_freq}"
  else
    echo "Numbered checkpoint freq: ${save_epoch_freq}"
  fi
  echo "LPIPS: ${lpips_w} (warmup_ratio=${lpips_warmup})"
  echo "Edge: ${edge_w}  |  Freq: ${freq_w}"
  echo "Refiner max_res:  ${refiner_max_res}"
  echo "Output:           ${output_dir}"
  echo "============================================================"

  # Validate paths
  if [[ ! -f "${CKPT}" && ! -d "${CKPT}" ]]; then
    echo "FATAL: checkpoint not found: ${CKPT}" >&2; return 1
  fi
  if [[ ! -d "${pseudo_path}" ]]; then
    echo "FATAL: pseudo data not found: ${pseudo_path}" >&2
    echo "Run scripts/generate_and_validate_pseudo.sh first." >&2
    return 1
  fi

  # Build extra args
  local extra_args=()
  if [[ "${freeze_jit}" == "0" && "${unfreeze_blocks}" -gt 0 ]]; then
    extra_args+=(--freeze_jit 0 --unfreeze_jit_last_blocks "${unfreeze_blocks}")
  fi
  if [[ -n "${SMOKE_MAX_STEPS}" ]]; then
    extra_args+=(${SMOKE_MAX_STEPS})
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    # Smoke test: skip online eval to save time
    extra_args+=(--eval_epoch 1000)
  else
    extra_args+=(--online_eval --eval_epoch 5)
  fi

  # https://pytorch.org/docs/stable/elastic/run.html
  CUDA_VISIBLE_DEVICES="${GPU}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    main_jit.py \
    --model "${MODEL}" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${batch}" \
    --lr "${lr}" \
    --lr_jit_last_blocks "${lr_jit}" \
    --lr_schedule cosine \
    --epochs "${epochs}" \
    --warmup_epochs "${warmup}" \
    --eval_num_images 100 \
    --num_sampling_steps 1 \
    --cfg 1.0 \
    --num_workers 8 \
    --save_last_freq 10 \
    --save_epoch_freq "${save_epoch_freq}" \
    --log_freq 50 \
    --output_dir "${output_dir}" \
    --use_bg_subnet 0 \
    --use_detail_refiner 1 \
    --freeze_jit "${freeze_jit}" \
    --refiner_base_dim "${REFINER_BASE_DIM}" \
    --refiner_num_blocks "${REFINER_NUM_BLOCKS}" \
    --refiner_use_frequency "${REFINER_USE_FREQUENCY}" \
    --refiner_max_residual "${refiner_max_res}" \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}" \
    --resume "${CKPT}" \
    --resume_state_key "${RESUME_STATE_KEY}" \
    --resume_optimizer 0 \
    --pseudo_data_path "${pseudo_path}" \
    --pseudo_mask_dir "${pseudo_mask}" \
    --pseudo_ratio "${pseudo_ratio}" \
    --pseudo_loss_weight "${pseudo_lw}" \
    --loss_lpips_weight "${lpips_w}" \
    --loss_edge_weight "${edge_w}" \
    --loss_freq_weight "${freq_w}" \
    --loss_lpips_warmup_ratio "${lpips_warmup}" \
    "${extra_args[@]}"

  echo ""
  echo "${ver^^} training finished.  Checkpoints: ${output_dir}/"
  return 0
}

# ========================================================================
# Main
# ========================================================================

CUDA_VISIBLE_DEVICES="${GPU}" python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; \
   print(f'GPU: {torch.cuda.get_device_name(0)}')"

VERSIONS=()
case "${VERSION}" in
  v1)   VERSIONS=(v1) ;;
  v2)   VERSIONS=(v2) ;;
  v3)   VERSIONS=(v3) ;;
  all)  VERSIONS=(v1 v2 v3) ;;
  *)
    echo "Unknown VERSION=${VERSION}.  Use v1 | v2 | v3 | all." >&2
    exit 2
    ;;
esac

for idx in "${!VERSIONS[@]}"; do
  ver="${VERSIONS[$idx]}"
  port=$((MASTER_PORT_BASE + idx))
  run_one "${ver}" "${port}" || exit $?
done

echo ""
echo "All requested pseudo-finetuning runs complete: ${VERSIONS[*]}"
if [[ "${SMOKE}" == "1" ]]; then
  echo "(smoke test — verify logs above for errors before submitting a real run)"
fi
