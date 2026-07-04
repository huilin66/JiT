#!/usr/bin/env bash
set -euo pipefail

# Smoke test JiT-H/16 focus-2scene no-head run on 1 x A100.
# Each run uses 2 epochs with 1 train step per epoch.
# Epoch 1 activates LPIPS and tests its backward path.

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPUS=${GPUS:-0}
NPROC=${NPROC:-1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30520}

DATA_PATH=${DATA_PATH:-/data/huilin/scrinvme/huilin/tp/eccv_dn/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
SCENE_FOCUS_2_PATH=${SCENE_FOCUS_2_PATH:-${DATA_PATH}/Drop_focus_2scene.json}

CKPT_L=${CKPT_L:-ckpt/jit-l-16}
CKPT_H=${CKPT_H:-ckpt/jit-h-16}
OUT_ROOT=${OUT_ROOT:-run/train_smoke/focus_2scene_h_no_head_1xA100_48g}

IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE_L=${BATCH_SIZE_L:-8}
BATCH_SIZE_H=${BATCH_SIZE_H:-4}
LR_L=${LR_L:-3e-5}
LR_H=${LR_H:-2e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}

CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPUs:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"
VISIBLE_GPU_COUNT=$(CUDA_VISIBLE_DEVICES="${GPUS}" python -c "import torch; print(torch.cuda.device_count())")
if (( NPROC > VISIBLE_GPU_COUNT )); then
  echo "NPROC=${NPROC} is larger than visible GPU count ${VISIBLE_GPU_COUNT} from GPUS=${GPUS}" >&2
  exit 2
fi

run_exp() {
  local exp_name="$1"
  local model="$2"
  local ckpt="$3"
  local batch_size="$4"
  local lr="$5"
  local port="$6"
  local output_dir="${OUT_ROOT}/${exp_name}/16"

  echo "============================================================"
  echo "[Smoke A100 focus 2scene] ${exp_name}"
  echo "model=${model}"
  echo "ckpt=${ckpt}"
  echo "GPUS=${GPUS}, nproc=${NPROC}, batch=${batch_size}, lr=${lr}"
  echo "Epoch 0 tests normal loss; epoch 1 tests LPIPS backward."
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    main_jit.py \
    --model "${model}" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${batch_size}" \
    --lr "${lr}" \
    --lr_schedule cosine \
    --epochs 2 \
    --warmup_epochs 1 \
    --eval_epoch 1 \
    --eval_num_images "${EVAL_NUM_IMAGES}" \
    --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --cfg 1.0 \
    --num_workers "${NUM_WORKERS}" \
    --save_last_freq 1 \
    --log_freq 1 \
    --output_dir "${output_dir}" \
    --use_bg_subnet 0 \
    --use_scene_dataset 1 \
    --data_path "${DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --resume "${ckpt}" \
    --resume_optimizer 0 \
    --online_eval \
    --scene_train_path "${SCENE_FOCUS_2_PATH}" \
    --scene_val_path "${SCENE_FOCUS_2_PATH}"

  if [[ ! -f "${output_dir}/checkpoint-last.pth" ]]; then
    echo "Smoke test failed: checkpoint-last.pth was not created for ${exp_name}" >&2
    exit 1
  fi
}

run_exp "jit_h16_focus_2scene_no_head" "JiT-H/16" "${CKPT_H}" "${BATCH_SIZE_H}" "${LR_H}" "$((MASTER_PORT_BASE + 1))"

echo "JiT/H focus_2scene 1xA100 smoke tests passed."
