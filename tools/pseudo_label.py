#!/usr/bin/env python3
"""Pseudo-label generation and validation for JiT raindrop removal.

Modes:
  generate  — Build mask-blended pseudo labels from ensemble predictions.
  validate  — Compare pure / TTA-median / mask-blended pseudo strategies vs GT.

Generate mode:
  1. Loads multiple prediction directories (ensemble) and optional TTA variants.
  2. Fuses all predictions per image with trimmed-mean or median.
  3. Computes a restoration mask from input-teacher diff.
  4. Blends pseudo = mask * fused + (1-mask) * (0.85*input + 0.15*fused).
  5. Saves pseudo PNGs and mask PNGs.
  6. Computes per-sample TTA variance for downstream filtering.

Validate mode:
  1. For each image, builds several pseudo candidates (pure, TTA-median, mask-blended).
  2. Evaluates each against the real GT using PSNR(Y) / SSIM(Y) / LPIPS / score.
  3. Saves a comparison CSV and optional side-by-side visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def list_pngs(directory):
    d = Path(directory)
    return sorted(p.name for p in d.glob("*.png"))


def load_rgb(path):
    """Load image as RGB float32 numpy [H, W, 3] in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(array, path):
    """Save float32 [H, W, 3] in [0, 1] as PNG."""
    out = np.rint(np.clip(array * 255.0, 0, 255)).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path, format="PNG", optimize=True)


def save_gray(array, path):
    """Save float32 [H, W] or [H, W, 1] in [0, 1] as 8-bit grayscale PNG."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    out = np.rint(np.clip(arr * 255.0, 0, 255)).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path, format="PNG", optimize=True)


def _diff_mask(diff_gray, t1=0.02, t2=0.08, blur_sigma=3.0):
    """Soft mask from per-pixel difference.

    mask = clip((diff - t1) / (t2 - t1), 0, 1)
    Then Gaussian blur for smooth boundaries.
    """
    mask = np.clip((diff_gray - t1) / max(t2 - t1, 0.001), 0.0, 1.0)
    if blur_sigma > 0:
        mask = gaussian_filter(mask, sigma=blur_sigma)
    return mask.astype(np.float32)


def load_scene_labels(path):
    if not path:
        return {}
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = {}
    for row in rows:
        if "filename" in row and "class_id" in row:
            labels[row["filename"]] = row["class_id"]
        elif {"type", "folder_name", "class_id"}.issubset(row):
            labels[f"{row['type'].strip().upper()}_{str(row['folder_name']).zfill(5)}"] = row["class_id"]
    return labels


def scene_key_for_name(name, scene_labels):
    if not scene_labels:
        return None
    if name in scene_labels:
        return str(scene_labels[name])
    parts = name.split("_")
    if len(parts) >= 2:
        time_prefix = "D" if parts[0].lower() == "day" else "N"
        key = f"{time_prefix}_{parts[1]}"
        if key in scene_labels:
            return str(scene_labels[key])
    return None


def fuse_prediction_arrays(preds, fusion, trim_low, trim_high):
    if len(preds) == 1:
        return preds[0]
    if fusion == "median":
        return np.median(np.stack(preds, axis=0), axis=0).astype(np.float32)
    if fusion == "trimmed_mean":
        return trimmed_mean_fusion(preds, trim_low, trim_high)
    if fusion == "mean":
        return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)
    raise ValueError(f"Unknown fusion: {fusion}")


# ---------------------------------------------------------------------------
# generate mode
# ---------------------------------------------------------------------------

def trimmed_mean_fusion(arrays, trim_low=5.0, trim_high=95.0):
    """Pixel-wise trimmed mean across the stack [N, H, W, C]."""
    stacked = np.stack(arrays, axis=0)  # [N, H, W, C]
    lo = np.percentile(stacked, trim_low, axis=0, keepdims=True)
    hi = np.percentile(stacked, trim_high, axis=0, keepdims=True)
    mask = (stacked >= lo) & (stacked <= hi)
    masked = np.where(mask, stacked, 0.0)
    count = mask.sum(axis=0, keepdims=True).clip(min=1)
    return (masked.sum(axis=0) / count).astype(np.float32)


def compute_tta_variance(arrays):
    """Per-pixel variance across N predictions, averaged to scalar."""
    stacked = np.stack(arrays, axis=0)  # [N, H, W, C]
    var = np.var(stacked, axis=0)  # [H, W, C]
    return float(np.mean(var))


def generate_pseudo(
    input_dir,
    pred_dirs,
    output_dir,
    mask_dir="",
    fusion="trimmed_mean",
    trim_low=5.0,
    trim_high=95.0,
    mask_t1=0.02,
    mask_t2=0.08,
    blur_sigma=3.0,
    blend_input=0.85,
    tta_variance_csv="",
    scene_json="",
    scene_median_weight=0.0,
    scene_median_aligned=False,
):
    """Generate mask-blended pseudo labels."""
    input_dir = Path(input_dir)
    pred_dirs = [Path(d) for d in pred_dirs]
    output_dir = Path(output_dir)
    mask_dir = Path(mask_dir) if mask_dir else output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Find common images across all prediction dirs.
    names = list_pngs(pred_dirs[0])
    for d in pred_dirs[1:]:
        cur = set(list_pngs(d))
        names = [n for n in names if n in cur]
    # Also require rainy input for each name.
    input_names = set(list_pngs(input_dir))
    names = [n for n in names if n in input_names]

    if not names:
        raise RuntimeError("No common images found across input and prediction dirs")

    scene_labels = load_scene_labels(scene_json)
    scene_medians = {}
    fused_cache = {}
    if scene_median_weight > 0:
        if not scene_median_aligned:
            raise ValueError(
                "--scene-median-weight requires --scene-median-aligned. "
                "Scene median is unsafe for unaligned frames."
            )
        scene_groups = defaultdict(list)
        for name in tqdm(names, desc="Precompute scene medians"):
            preds = [load_rgb(d / name) for d in pred_dirs]
            fused = np.clip(
                fuse_prediction_arrays(preds, fusion, trim_low, trim_high), 0.0, 1.0
            )
            fused_cache[name] = fused
            scene_key = scene_key_for_name(name, scene_labels)
            if scene_key is not None:
                scene_groups[scene_key].append(name)
        for scene_key, group_names in scene_groups.items():
            shapes = {fused_cache[n].shape for n in group_names}
            if len(group_names) > 1 and len(shapes) == 1:
                scene_medians[scene_key] = np.median(
                    np.stack([fused_cache[n] for n in group_names], axis=0),
                    axis=0,
                ).astype(np.float32)

    tta_variances = {}
    mask_stats = defaultdict(list)

    print(f"Generating pseudo labels: {len(names)} images")
    print(f"  Fusion: {fusion} (trim [{trim_low}, {trim_high}])")
    print(f"  Mask:   t1={mask_t1}, t2={mask_t2}, blur_sigma={blur_sigma}")
    print(f"  Blend:  {blend_input:.2f}*input + {1-blend_input:.2f}*fused")
    print(f"  Output: {output_dir}")

    for name in tqdm(names, desc="Pseudo generate"):
        # Load all teacher predictions.
        preds = [load_rgb(d / name) for d in pred_dirs]
        x = load_rgb(input_dir / name)

        # Fusion.
        fused = fused_cache.get(name)
        if fused is None:
            fused = fuse_prediction_arrays(preds, fusion, trim_low, trim_high)
        scene_key = scene_key_for_name(name, scene_labels)
        if scene_median_weight > 0 and scene_key in scene_medians:
            fused = (
                (1.0 - scene_median_weight) * fused
                + scene_median_weight * scene_medians[scene_key]
            ).astype(np.float32)

        # TTA variance.
        if len(preds) > 1:
            tta_variances[name] = compute_tta_variance(preds)

        # Restoration mask.
        diff = np.mean(np.abs(x - fused), axis=2)  # [H, W]
        mask = _diff_mask(diff, t1=mask_t1, t2=mask_t2, blur_sigma=blur_sigma)

        # Mask-blended pseudo.
        mask_3c = np.stack([mask] * 3, axis=-1)
        pseudo = mask_3c * fused + (1.0 - mask_3c) * (blend_input * x + (1.0 - blend_input) * fused)
        pseudo = np.clip(pseudo, 0.0, 1.0)

        # Save.
        save_rgb(pseudo, output_dir / name)
        save_gray(mask, mask_dir / f"{Path(name).stem}_mask.png")

        # Stats.
        mask_stats["mean"].append(float(np.mean(mask)))
        mask_stats["min"].append(float(np.min(mask)))
        mask_stats["max"].append(float(np.max(mask)))

    # Summary.
    print("\n[Mask stats]")
    for key in ("mean", "min", "max"):
        vals = mask_stats[key]
        print(f"  mask_{key}: avg={np.mean(vals):.4f}, "
              f"p10={np.percentile(vals, 10):.4f}, p90={np.percentile(vals, 90):.4f}")

    if tta_variances:
        var_vals = list(tta_variances.values())
        print(f"\n[TTA variance] {len(var_vals)} samples")
        print(f"  avg={np.mean(var_vals):.6f}, "
              f"p50={np.percentile(var_vals, 50):.6f}, "
              f"p90={np.percentile(var_vals, 90):.6f}, "
              f"max={np.max(var_vals):.6f}")

    if tta_variance_csv:
        csv_path = Path(tta_variance_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "tta_variance", "mask_mean"])
            writer.writeheader()
            for name in sorted(names):
                writer.writerow({
                    "filename": name,
                    "tta_variance": round(tta_variances.get(name, 0.0), 8),
                    "mask_mean": round(mask_stats["mean"][names.index(name)] if name in names else 0.0, 6),
                })
        print(f"TTA variance CSV: {csv_path}")

    print(f"\nPseudo labels: {output_dir} ({len(names)} images)")
    print(f"Masks:         {mask_dir}")


# ---------------------------------------------------------------------------
# validate mode
# ---------------------------------------------------------------------------

def evaluate_prediction(pred, gt):
    """Compute PSNR(Y), SSIM(Y), LPIPS, composite score.

    Returns dict with float values. LPIPS requires pip install lpips.
    """
    def _rgb_to_y_np(arr):
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    def _ssim_np(a, b):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
        sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
        sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
        ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
            (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + 1e-12
        )
        return float(np.mean(ssim_map))

    try:
        import lpips
        import torch
        from piq import ssim

        _lpips_vgg = lpips.LPIPS(net="vgg")
        _lpips_vgg.eval()

        def _rgb_to_y(t):
            return 0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]
    except ImportError:
        pred_y = _rgb_to_y_np(np.clip(pred, 0.0, 1.0))
        gt_y = _rgb_to_y_np(np.clip(gt, 0.0, 1.0))
        mse = float(np.mean((pred_y - gt_y) ** 2))
        psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        ssim_val = _ssim_np(pred_y, gt_y)
        return {
            "psnr_y": psnr,
            "ssim_y": ssim_val,
            "lpips": 0.0,
            "score": psnr + 10.0 * ssim_val,
        }

    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)

    pred_y = _rgb_to_y(pred_t)
    gt_y = _rgb_to_y(gt_t)

    mse = torch.mean((pred_y - gt_y) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    psnr = float((10.0 * torch.log10(1.0 / mse)).mean())

    try:
        from piq import ssim as piq_ssim
        ssim_val = float(piq_ssim(pred_y, gt_y, data_range=1.0))
    except ImportError:
        ssim_val = 0.0

    try:
        import lpips
        _lpips_vgg = lpips.LPIPS(net="vgg").eval()
        lpips_val = float(_lpips_vgg(pred_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).mean())
        score = psnr + 10.0 * ssim_val - 5.0 * lpips_val
    except ImportError:
        lpips_val = 0.0
        score = psnr + 10.0 * ssim_val

    return {"psnr_y": psnr, "ssim_y": ssim_val, "lpips": lpips_val, "score": score}


def validate_pseudo(
    input_dir,
    pred_dirs,
    gt_clear_dir,
    output_dir,
    fusion="trimmed_mean",
    trim_low=5.0,
    trim_high=95.0,
    mask_t1=0.02,
    mask_t2=0.08,
    blur_sigma=3.0,
    blend_input=0.85,
    save_viz=False,
):
    """Compare pseudo strategies against ground truth."""
    input_dir = Path(input_dir)
    pred_dirs = [Path(d) for d in pred_dirs]
    gt_clear_dir = Path(gt_clear_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_names = set(list_pngs(gt_clear_dir))
    input_names = set(list_pngs(input_dir))
    pred_names = set(list_pngs(pred_dirs[0]))
    for d in pred_dirs[1:]:
        pred_names &= set(list_pngs(d))
    names = sorted(gt_names & input_names & pred_names)

    if not names:
        raise RuntimeError("No common images found")

    print(f"Validating pseudo strategies on {len(names)} images against GT")

    results = []
    for name in tqdm(names, desc="Pseudo validate"):
        preds = [load_rgb(d / name) for d in pred_dirs]
        x = load_rgb(input_dir / name)
        gt = load_rgb(gt_clear_dir / name)

        # Strategy A: pure teacher (first prediction)
        pure = preds[0]
        metrics_pure = evaluate_prediction(pure, gt)

        # Strategy B: TTA median (or trimmed mean)
        if len(preds) == 1:
            tta_fused = preds[0]
        elif fusion == "median":
            tta_fused = np.median(np.stack(preds, axis=0), axis=0).astype(np.float32)
        elif fusion == "trimmed_mean":
            tta_fused = trimmed_mean_fusion(preds, trim_low, trim_high)
        else:
            tta_fused = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)
        tta_fused = np.clip(tta_fused, 0.0, 1.0)
        metrics_tta = evaluate_prediction(tta_fused, gt)

        # Strategy C: mask-blended pseudo
        diff = np.mean(np.abs(x - tta_fused), axis=2)
        mask = _diff_mask(diff, t1=mask_t1, t2=mask_t2, blur_sigma=blur_sigma)
        mask_3c = np.stack([mask] * 3, axis=-1)
        pseudo = mask_3c * tta_fused + (1.0 - mask_3c) * (blend_input * x + (1.0 - blend_input) * tta_fused)
        pseudo = np.clip(pseudo, 0.0, 1.0)
        metrics_pseudo = evaluate_prediction(pseudo, gt)

        results.append({
            "filename": name,
            "mask_mean": round(float(np.mean(mask)), 6),
            **{f"pure_{k}": round(v, 6) for k, v in metrics_pure.items()},
            **{f"tta_{k}": round(v, 6) for k, v in metrics_tta.items()},
            **{f"pseudo_{k}": round(v, 6) for k, v in metrics_pseudo.items()},
        })

        if save_viz:
            viz_dir = output_dir / "viz" / Path(name).stem
            viz_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(x, viz_dir / "input.png")
            save_rgb(gt, viz_dir / "gt.png")
            save_rgb(pure, viz_dir / "pure_teacher.png")
            save_rgb(tta_fused, viz_dir / f"tta_{fusion}.png")
            save_rgb(pseudo, viz_dir / "mask_blended.png")
            save_gray(mask, viz_dir / "mask.png")

    # Summary.
    keys = ["psnr_y", "ssim_y", "lpips", "score"]
    strategies = ["pure", "tta", "pseudo"]
    print("\n" + "=" * 70)
    print(f"{'Strategy':<16} {'PSNR(Y)':>10} {'SSIM(Y)':>10} {'LPIPS':>10} {'Score':>10}")
    print("-" * 70)
    for strat in strategies:
        avg = {}
        for k in keys:
            col = f"{strat}_{k}"
            avg[k] = np.mean([r[col] for r in results])
        print(f"{strat:<16} {avg['psnr_y']:>10.4f} {avg['ssim_y']:>10.4f} "
              f"{avg['lpips']:>10.4f} {avg['score']:>10.4f}")
    print("=" * 70)

    # Save CSV.
    csv_path = output_dir / "pseudo_validation.csv"
    fieldnames = ["filename", "mask_mean"]
    for strat in strategies:
        for k in keys:
            fieldnames.append(f"{strat}_{k}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nValidation CSV: {csv_path}")

    # Recommendation.
    pseudo_scores = [r["pseudo_score"] for r in results]
    tta_scores = [r["tta_score"] for r in results]
    pure_scores = [r["pure_score"] for r in results]
    print(f"\nScore comparison vs GT (higher = closer to real GT):")
    print(f"  Pure teacher:     {np.mean(pure_scores):.4f}")
    print(f"  TTA {fusion}:     {np.mean(tta_scores):.4f}")
    print(f"  Mask-blended:     {np.mean(pseudo_scores):.4f}")
    if np.mean(pseudo_scores) > np.mean(tta_scores):
        print("  => Mask-blended pseudo is BETTER than plain fusion for pseudo-GT.")
    else:
        print("  => Note: mask-blended pseudo scores lower vs real GT (expected — it "
              "preserves input texture which differs from GT). This is OK for training.")


# ---------------------------------------------------------------------------
# filter mode
# ---------------------------------------------------------------------------

def filter_pseudo(
    tta_variance_csv,
    output_dir,
    pseudo_dir,
    mask_dir,
    filter_top_pct=10.0,
    filter_threshold=0.0,
):
    """Filter out high-variance pseudo samples. Outputs a filtered file list."""
    with open(tta_variance_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Empty TTA variance CSV")

    variances = np.array([float(r["tta_variance"]) for r in rows])
    if filter_threshold > 0:
        threshold = float(filter_threshold)
    elif filter_top_pct > 0:
        threshold = np.percentile(variances, 100.0 - filter_top_pct)
    else:
        threshold = float("inf")

    kept, dropped = [], []
    for row, var in zip(rows, variances):
        (kept if var <= threshold else dropped).append(row["filename"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered file lists.
    with open(output_dir / "filtered_kept.txt", "w") as f:
        f.write("\n".join(sorted(kept)) + "\n")
    with open(output_dir / "filtered_dropped.txt", "w") as f:
        f.write("\n".join(sorted(dropped)) + "\n")

    # Optionally copy kept samples to filtered dirs.
    if pseudo_dir:
        filtered_pseudo = output_dir / "PseudoGT_filtered"
        filtered_pseudo.mkdir(parents=True, exist_ok=True)
        for name in kept:
            src = Path(pseudo_dir) / name
            if src.exists():
                import shutil
                shutil.copy2(src, filtered_pseudo / name)

    if mask_dir:
        filtered_mask = output_dir / "masks_filtered"
        filtered_mask.mkdir(parents=True, exist_ok=True)
        for name in kept:
            stem = Path(name).stem
            src = Path(mask_dir) / f"{stem}_mask.png"
            if src.exists():
                import shutil
                shutil.copy2(src, filtered_mask / f"{stem}_mask.png")

    print(f"Filtered pseudo samples: {len(kept)} kept, {len(dropped)} dropped "
          f"({100*len(dropped)/len(rows):.1f}% @ threshold={threshold:.6f})")
    print(f"Kept list:   {output_dir / 'filtered_kept.txt'}")
    print(f"Dropped list: {output_dir / 'filtered_dropped.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate mask-blended pseudo labels")
    gen.add_argument("--input-dir", required=True, help="Rainy Drop image directory")
    gen.add_argument("--pred-dirs", nargs="+", required=True,
                     help="One or more prediction directories (ensemble/TTA outputs)")
    gen.add_argument("--output-dir", required=True, help="Output directory for pseudo PNGs")
    gen.add_argument("--mask-dir", default="", help="Output directory for mask PNGs")
    gen.add_argument("--fusion", default="trimmed_mean",
                     choices=["trimmed_mean", "median", "mean"])
    gen.add_argument("--trim-low", type=float, default=5.0)
    gen.add_argument("--trim-high", type=float, default=95.0)
    gen.add_argument("--mask-t1", type=float, default=0.02,
                     help="Lower threshold for soft mask (fraction of max diff)")
    gen.add_argument("--mask-t2", type=float, default=0.08,
                     help="Upper threshold for soft mask")
    gen.add_argument("--blur-sigma", type=float, default=3.0,
                     help="Gaussian blur sigma for mask smoothing")
    gen.add_argument("--blend-input", type=float, default=0.85,
                     help="Input weight in non-mask regions: (1-blend_input)*fused + blend_input*input")
    gen.add_argument("--tta-variance-csv", default="",
                     help="Save per-sample TTA variance for filtering")
    gen.add_argument("--scene-json", default="",
                     help="Optional scene-label JSON/CSV for explicit aligned scene median.")
    gen.add_argument("--scene-median-weight", type=float, default=0.0,
                     help="Optional weight mixed from per-scene median teacher output. Default off.")
    gen.add_argument("--scene-median-aligned", action="store_true",
                     help="Required with --scene-median-weight; asserts same-scene frames are aligned.")

    # --- validate ---
    val = sub.add_parser("validate", help="Compare pseudo strategies against real GT")
    val.add_argument("--input-dir", required=True)
    val.add_argument("--pred-dirs", nargs="+", required=True)
    val.add_argument("--gt-clear-dir", required=True, help="Real GT Clear directory")
    val.add_argument("--output-dir", required=True)
    val.add_argument("--fusion", default="trimmed_mean",
                     choices=["trimmed_mean", "median", "mean"])
    val.add_argument("--trim-low", type=float, default=5.0)
    val.add_argument("--trim-high", type=float, default=95.0)
    val.add_argument("--mask-t1", type=float, default=0.02)
    val.add_argument("--mask-t2", type=float, default=0.08)
    val.add_argument("--blur-sigma", type=float, default=3.0)
    val.add_argument("--blend-input", type=float, default=0.85)
    val.add_argument("--save-viz", action="store_true",
                     help="Save side-by-side visualization images")

    # --- filter ---
    filt = sub.add_parser("filter", help="Filter pseudo samples by TTA variance")
    filt.add_argument("--tta-variance-csv", required=True)
    filt.add_argument("--output-dir", required=True)
    filt.add_argument("--pseudo-dir", default="")
    filt.add_argument("--mask-dir", default="")
    filt.add_argument("--filter-top-pct", type=float, default=10.0,
                       help="Drop top N%% highest-variance samples (default 10)")
    filt.add_argument("--filter-threshold", type=float, default=0.0,
                       help="Absolute variance threshold (overrides --filter-top-pct if > 0)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "generate":
        generate_pseudo(
            input_dir=args.input_dir,
            pred_dirs=args.pred_dirs,
            output_dir=args.output_dir,
            mask_dir=args.mask_dir,
            fusion=args.fusion,
            trim_low=args.trim_low,
            trim_high=args.trim_high,
            mask_t1=args.mask_t1,
            mask_t2=args.mask_t2,
            blur_sigma=args.blur_sigma,
            blend_input=args.blend_input,
            tta_variance_csv=args.tta_variance_csv,
            scene_json=args.scene_json,
            scene_median_weight=args.scene_median_weight,
            scene_median_aligned=args.scene_median_aligned,
        )
    elif args.mode == "validate":
        validate_pseudo(
            input_dir=args.input_dir,
            pred_dirs=args.pred_dirs,
            gt_clear_dir=args.gt_clear_dir,
            output_dir=args.output_dir,
            fusion=args.fusion,
            trim_low=args.trim_low,
            trim_high=args.trim_high,
            mask_t1=args.mask_t1,
            mask_t2=args.mask_t2,
            blur_sigma=args.blur_sigma,
            blend_input=args.blend_input,
            save_viz=args.save_viz,
        )
    elif args.mode == "filter":
        filter_pseudo(
            tta_variance_csv=args.tta_variance_csv,
            output_dir=args.output_dir,
            pseudo_dir=args.pseudo_dir,
            mask_dir=args.mask_dir,
            filter_top_pct=args.filter_top_pct,
            filter_threshold=args.filter_threshold,
        )


if __name__ == "__main__":
    main()
