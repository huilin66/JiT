#!/usr/bin/env bash
set -euo pipefail

# Smoke test for train_b16_dn_blur_4scene_aug_1x5090.sh.
# Each setting runs two epochs with one train iteration per epoch.
# Epoch 0 has zero LPIPS weight; epoch 1 activates LPIPS backward.

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

GPU=${GPU:-0}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30040}

DATA_ROOT=${DATA_ROOT:-D:/zhl/data/eccv_dn}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/RainDrop_Train}
VAL_DATA_PATH=${VAL_DATA_PATH:-${DATA_PATH}}
CKPT=${CKPT:-ckpt/jit-b-16}
OUT_ROOT=${OUT_ROOT:-run/train_smoke/b16_dn_blur_4scene_aug_1x5090}

SCENE_DN_BLUR_4_PATH=${SCENE_DN_BLUR_4_PATH:-${DATA_PATH}/Drop_dn_blur_4scene.json}

MODEL=${MODEL:-JiT-B/16}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-2.5e-5}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-1}
MORE_AUG=${MORE_AUG:-1}
LOSS_EDGE_WEIGHT=${LOSS_EDGE_WEIGHT:-0.03}
LOSS_FREQ_WEIGHT=${LOSS_FREQ_WEIGHT:-0.01}

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

run_smoke() {
  local scene_name="$1"
  local scene_path="$2"
  local port="$3"
  local edge_weight="$4"
  local freq_weight="$5"
  local output_dir="${OUT_ROOT}/b16_${scene_name}_aug_no_head/16"

  echo "============================================================"
  echo "[Smoke manual blur ${scene_name}] b16_${scene_name}_aug_no_head"
  echo "Epoch 0 tests normal loss; epoch 1 tests LPIPS backward."
  echo "scene_path=${scene_path}"
  echo "more_aug=${MORE_AUG}"
  echo "loss_edge_weight=${edge_weight}"
  echo "loss_freq_weight=${freq_weight}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python main_jit.py \
    --model "${MODEL}" \
    --proj_dropout 0.0 \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
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
    --resume "${CKPT}" \
    --resume_optimizer 0 \
    --online_eval \
    --scene_train_path "${scene_path}" \
    --scene_val_path "${scene_path}" \
    --more_aug "${MORE_AUG}" \
    --loss_edge_weight "${edge_weight}" \
    --loss_freq_weight "${freq_weight}"

  test -f "${output_dir}/checkpoint-last.pth"
}

run_smoke "dn_blur_4scene" "${SCENE_DN_BLUR_4_PATH:-${DATA_PATH}/Drop_dn_blur_4scene.json}" "$((MASTER_PORT_BASE + 0))" 0.0 0.0
run_smoke "dn_blur_4scene_edge_freq" "${SCENE_DN_BLUR_4_PATH:-${DATA_PATH}/Drop_dn_blur_4scene.json}" "$((MASTER_PORT_BASE + 1))" "${LOSS_EDGE_WEIGHT}" "${LOSS_FREQ_WEIGHT}"

echo "dn_blur_4scene augmented smoke tests passed, including LPIPS backward."
