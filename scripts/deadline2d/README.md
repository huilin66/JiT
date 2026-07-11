# Two-day deadline scripts

These scripts keep the deadline experiment matrix intentionally small.

## 1. Three parallel RTX 3090 fine-tunes

Run from the JiT repository. The MSDT repository is expected at `../MSDT` unless
`MSDT_ROOT` is set. If `MSDT/splits/raindrop_split.json` is missing, the launcher
creates it deterministically in one process before starting the three GPUs. If it
already exists, its group/sample fingerprints are validated and it is reused.

```bash
MSDT_ROOT=/path/to/MSDT \
DATA_PATH=/path/to/RainDrop_Train \
SOURCE_CKPT=/path/to/MSDT/checkpoints/msdt_1x5090/no_scene/model_best.pth \
GPUS="0 1 2" EPOCHS=8 BATCH_SIZE=4 \
bash scripts/deadline2d/train_msdt_3x3090_finetune.sh
```

The launcher resets Adam and the cosine scheduler to `lr=5e-6`, runs one loss
variant per GPU, and writes separate logs/metrics/checkpoints below
`MSDT/checkpoints/deadline2d_3x3090` by default. Select the run using validation
`Score`, with PSNR/SSIM/LPIPS checked individually for oversmoothing.

Before the full run, execute the fast three-GPU smoke test. Each GPU performs
one train step and validates one image; it also checks BF16, finite gradients,
an actual parameter update, metrics, and checkpoint creation:

```bash
MSDT_ROOT=/path/to/MSDT \
DATA_PATH=/path/to/RainDrop_Train \
SOURCE_CKPT=/path/to/MSDT/checkpoints/msdt_1x5090/no_scene/model_best.pth \
GPUS="0 1 2" \
bash scripts/deadline2d/smoke_test_msdt_3x3090_finetune.sh
```

Only start the full eight-epoch run after the script prints
`SMOKE_TEST_PASSED`.

## 2. One RTX 5090 inference

Run JiT and MSDT separately when checkpoint paths need adjustment:

```bash
GPU=0 DATA_ROOT=/path/to/eccv_dn \
JIT_B_CKPT=/path/to/jit_b_checkpoint_dir \
JIT_L_CKPT=/path/to/jit_l_checkpoint_dir \
bash scripts/deadline2d/infer_jit_1x5090.sh

GPU=0 MSDT_ROOT=/path/to/MSDT DATA_ROOT=/path/to/eccv_dn \
FT_ROOT=/path/to/JiT/run/deadline2d_3x3090 \
bash scripts/deadline2d/infer_msdt_1x5090.sh
```

JiT inference predicts a fresh two-class scene JSON by default with
`run/scene_convnext_focus_2scene_v1/checkpoint-best.pth`, validates exact input
coverage and labels, then reuses it for JiT-B and JiT-L. Set
`SCENE_CKPT=/path/to/checkpoint-best.pth` to select another classifier, or set
`SCENE_JSON=/path/to/existing.json` to deliberately skip scene prediction.

Or run the full sequence, including the small fusion candidate set:

```bash
GPU=0 DATA_ROOT=/path/to/eccv_dn MSDT_ROOT=/path/to/MSDT \
bash scripts/deadline2d/run_all_1x5090.sh
```

`build_fusion_candidates.sh` creates two fixed fusions, one disagreement-adaptive
fusion, and beta 0.95/0.90 input-backblend variants. Every postprocessed candidate
checks the expected PNG count, dimensions, produces a flat ZIP, and records SHA-256
and arguments in `manifest.json`.

Do not blindly submit every candidate. Validate the same operations on the 406-image
development set first, then carry only the most stable one to the final set.
