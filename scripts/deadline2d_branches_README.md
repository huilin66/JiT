# Deadline2d parallel branches

All commands assume the current directory is the JiT repository root.

## P1: prediction-only scene consensus

```bash
NAME=jit_b_scene OUTPUT_ROOT=/path/to/output \
bash scripts/deadline2d_p1_post/run_scene_postprocess.sh /path/to/jit_b_png_dir

NAME=jit_b_msdt MODEL_WEIGHTS=0.5,0.5 \
bash scripts/deadline2d_p1_post/run_scene_postprocess.sh \
  /path/to/jit_b_png_dir /path/to/msdt_png_dir
```

## Selected JiT-B initialization

P2 and P3 both initialize the MSDT detail refiner from the best JiT-B row in
`submission_history_jit.csv` (development score 34.4282):

```text
run/train/b16_focus_2scene_msdt_refiner_plan_c12_01_c1_higher_than_c_1x5090/16/checkpoint-last.pth
state_key=model_ema1
```

Override `JIT_B_CKPT` and `JIT_STATE_KEY` if the server uses another path.

Every training run saves both `model_best_score.pth` (highest
`PSNR_Y + 10*SSIM_Y - 5*LPIPS`) and `model_best_psnr.pth` (highest PSNR-Y).
Inference defaults to score-best; set `BEST_KIND=psnr` to use PSNR-best.

## P2: fast JiT-MSDT Blur->Clear continuation on 3x3090

The default formal schedule is 100 epochs (roughly 3.5--4 hours when one epoch
including validation takes about two minutes).

Smoke test first:

```bash
DATA_ROOT=/path/to/RainDrop_Train OUT_ROOT=/path/to/p2_smoke \
EPOCHS=1 MAX_TRAIN_STEPS=2 MAX_VAL_BATCHES=1 \
bash scripts/deadline2d_p2_fast_bc-refiner/train_3x3090.sh
```

Formal training and inference:

```bash
DATA_ROOT=/path/to/RainDrop_Train OUT_ROOT=/path/to/p2_run \
bash scripts/deadline2d_p2_fast_bc-refiner/train_3x3090.sh

INPUT_DIR=/path/to/existing_prediction_pngs CKPT_ROOT=/path/to/p2_run \
OUTPUT_ROOT=/path/to/p2_predictions \
bash scripts/deadline2d_p2_fast_bc-refiner/infer_1x5090.sh
```

Omit `INPUT_DIR` to generate the matching JiT-B base automatically from the
selected checkpoint before refinement. Set `SCENE_JSON` to reuse a verified
two-class JSON; otherwise JiT's scene classifier runs automatically.

## P3: frequency DetailRefiner on 1xA100

The default formal schedule is 120 epochs. Score-best and PSNR-best are saved
independently throughout the run.

```bash
DATA_ROOT=/path/to/RainDrop_Train OUT_ROOT=/path/to/p3_smoke \
EPOCHS=1 MAX_TRAIN_STEPS=2 MAX_VAL_BATCHES=1 \
bash scripts/deadline2d_p3_detail-refiner/train_1xA100.sh

DATA_ROOT=/path/to/RainDrop_Train OUT_ROOT=/path/to/p3_run \
bash scripts/deadline2d_p3_detail-refiner/train_1xA100.sh

INPUT_DIR=/path/to/existing_prediction_pngs CHECKPOINT=/path/to/p3_run/model_best.pth \
OUTPUT_ROOT=/path/to/p3_predictions \
bash scripts/deadline2d_p3_detail-refiner/infer_1x5090.sh
```

As with P2, omit `INPUT_DIR` to generate the matching JiT-B base automatically.

Training consumes `Blur/` as input and `Clear/` as target. Both branches load
the existing `detail_refiner.*` weights from JiT-B before Blur->Clear training.
Inference consumes a flat directory of already restored JiT PNGs or generates
that base itself. Every inference output has a manifest containing the source
JiT checkpoint hash, refiner checkpoint hash, and archive hash.
