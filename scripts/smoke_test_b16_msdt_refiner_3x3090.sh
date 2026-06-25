#!/usr/bin/env bash
set -euo pipefail

# Two-epoch smoke test. Each epoch runs one train iteration.
DATA_ROOT=${DATA_ROOT:-/root/huilin/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUTPUT_DIR=${OUTPUT_DIR:-run/train_smoke/jit_b16_msdt_refiner/16}

if [[ -z "${CKPT}" ]]; then
  echo "CKPT must point to a trained JiT checkpoint file or directory." >&2
  exit 2
fi
if [[ ! -f "${CKPT}" && ! -f "${CKPT}/checkpoint-last.pth" ]]; then
  echo "Cannot find checkpoint file: ${CKPT} (or ${CKPT}/checkpoint-last.pth)" >&2
  exit 2
fi

GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-29730}

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  main_jit.py \
  --model JiT-B/16 \
  --img_size 256 \
  --batch_size 4 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --epochs 2 \
  --warmup_epochs 1 \
  --eval_epoch 1 \
  --eval_num_images 1 \
  --num_sampling_steps 3 \
  --max_train_steps 1 \
  --num_workers 2 \
  --save_last_freq 1 \
  --log_freq 1 \
  --output_dir "${OUTPUT_DIR}" \
  --use_bg_subnet 0 \
  --use_detail_refiner 1 \
  --freeze_jit 1 \
  --refiner_base_dim 16 \
  --refiner_num_blocks 1 \
  --refiner_use_frequency 1 \
  --refiner_max_residual 0.25 \
  --loss_edge_weight 0.05 \
  --loss_freq_weight 0.05 \
  --use_scene_dataset 0 \
  --data_path "${DATA_PATH}" \
  --val_data_path "${VAL_DATA_PATH}" \
  --resume "${CKPT}" \
  --resume_state_key model_ema1 \
  --resume_optimizer 0 \
  --online_eval
