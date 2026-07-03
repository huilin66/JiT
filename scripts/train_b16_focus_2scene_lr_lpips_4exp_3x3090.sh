#!/usr/bin/env bash
set -euo pipefail

# Four focused JiT-B/16 experiments on 3 x RTX 3090:
#   1) lower LR
#   2) higher LR
#   3) lower LPIPS loss weight
#   4) higher LPIPS loss weight
#
# All runs keep focus-2scene, no head, no aug, no edge/frequency loss.

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30420}

DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train/b16_focus_2scene_lr_lpips_4exp_no_head_3x3090}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-600}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_EPOCH=${EVAL_EPOCH:-5}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
SAVE_LAST_FREQ=${SAVE_LAST_FREQ:-5}
LOG_FREQ=${LOG_FREQ:-50}
ONLINE_EVAL=${ONLINE_EVAL:-1}
RESUME_OPTIMIZER=${RESUME_OPTIMIZER:-0}

LR_BASE=${LR_BASE:-5e-5}
LR_LOW=${LR_LOW:-3e-5}
LR_HIGH=${LR_HIGH:-7e-5}
LPIPS_BASE=${LPIPS_BASE:-0.5}
LPIPS_LOW=${LPIPS_LOW:-0.3}
LPIPS_HIGH=${LPIPS_HIGH:-0.7}

online_eval_args=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
  online_eval_args+=(--online_eval)
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

run_exp() {
  local exp_name="$1"
  local lr="$2"
  local lpips_weight="$3"
  local port="$4"
  local output_dir="${OUT_ROOT}/${exp_name}/16"

  echo "============================================================"
  echo "[JiT-B focus 2scene LR/LPIPS] ${exp_name}"
  echo "GPUS=${GPUS}, nproc=${NPROC}, batch=${BATCH_SIZE}"
  echo "lr=${lr}, loss_lpips_weight=${lpips_weight}"
  echo "scene_path=${SCENE_FOCUS_2_PATH}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    main_jit.py \
    --model "${MODEL}" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${lr}" \
    --lr_schedule cosine \
    --epochs "${EPOCHS}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --eval_epoch "${EVAL_EPOCH}" \
    --eval_num_images "${EVAL_NUM_IMAGES}" \
    --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
    --cfg 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --save_last_freq "${SAVE_LAST_FREQ}" \
    --log_freq "${LOG_FREQ}" \
    --output_dir "${output_dir}" \
    --use_bg_subnet 0 \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${CKPT}" \
    --resume_optimizer "${RESUME_OPTIMIZER}" \
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}" \
    --loss_lpips_weight "${lpips_weight}" \
    "${online_eval_args[@]}"
}

run_exp "b16_focus_2scene_lr_low_lpips_base_no_head" "${LR_LOW}" "${LPIPS_BASE}" "$((MASTER_PORT_BASE + 0))"
run_exp "b16_focus_2scene_lr_high_lpips_base_no_head" "${LR_HIGH}" "${LPIPS_BASE}" "$((MASTER_PORT_BASE + 1))"
run_exp "b16_focus_2scene_lr_base_lpips_low_no_head" "${LR_BASE}" "${LPIPS_LOW}" "$((MASTER_PORT_BASE + 2))"
run_exp "b16_focus_2scene_lr_base_lpips_high_no_head" "${LR_BASE}" "${LPIPS_HIGH}" "$((MASTER_PORT_BASE + 3))"

echo "JiT-B focus_2scene LR/LPIPS 4-experiment 3x3090 training finished."
