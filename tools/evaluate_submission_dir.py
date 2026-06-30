#!/usr/bin/env python3
"""Evaluate a PNG prediction directory against Clear images."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from piq import ssim
from tqdm import tqdm


CSV_FIELDS = [
    "timestamp",
    "model_name",
    "prediction_dir",
    "checkpoint",
    "ckpt_type",
    "state_key",
    "steps",
    "stride",
    "tile_batch_size",
    "scene_json",
    "num_images",
    "runtime_seconds",
    "psnr_y",
    "ssim_y",
    "lpips",
    "score",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--clear-dir", required=True)
    parser.add_argument("--csv", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--ckpt-type", default="")
    parser.add_argument("--state-key", default="")
    parser.add_argument("--steps", default="")
    parser.add_argument("--stride", default="")
    parser.add_argument("--tile-batch-size", default="")
    parser.add_argument("--scene-json", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def append_csv(path: str, row: dict):
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != CSV_FIELDS:
            raise RuntimeError(f"Existing CSV has incompatible header: {csv_path}")
    with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    clear_dir = Path(args.clear_dir)
    pred_files = sorted(prediction_dir.glob("*.png"))
    if not pred_files:
        raise RuntimeError(f"No PNG predictions found in {prediction_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    totals = {"psnr_y": 0.0, "ssim_y": 0.0, "lpips": 0.0, "score": 0.0}
    started = time.perf_counter()
    with torch.inference_mode():
        for pred_path in tqdm(pred_files, desc="Evaluating local predictions"):
            clear_path = clear_dir / pred_path.name
            if not clear_path.is_file():
                raise FileNotFoundError(f"Missing GT image: {clear_path}")
            pred = load_rgb(pred_path).to(device)
            target = load_rgb(clear_path).to(device)
            if pred.shape != target.shape:
                raise RuntimeError(f"Shape mismatch for {pred_path.name}: {tuple(pred.shape)} vs {tuple(target.shape)}")

            pred_y = rgb_to_y(pred)
            target_y = rgb_to_y(target)
            mse = torch.mean((pred_y - target_y) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
            psnr_y = float((10.0 * torch.log10(1.0 / mse)).mean().cpu())
            ssim_y = float(ssim(pred_y, target_y, data_range=1.0).cpu())
            lpips_value = float(lpips_vgg(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().cpu())
            score = psnr_y + 10.0 * ssim_y - 5.0 * lpips_value

            totals["psnr_y"] += psnr_y
            totals["ssim_y"] += ssim_y
            totals["lpips"] += lpips_value
            totals["score"] += score

    count = len(pred_files)
    runtime = time.perf_counter() - started
    metrics = {key: value / count for key, value in totals.items()}
    row = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model_name": args.model_name or prediction_dir.name,
        "prediction_dir": str(prediction_dir.resolve()),
        "checkpoint": args.checkpoint,
        "ckpt_type": args.ckpt_type,
        "state_key": args.state_key,
        "steps": args.steps,
        "stride": args.stride,
        "tile_batch_size": args.tile_batch_size,
        "scene_json": args.scene_json,
        "num_images": count,
        "runtime_seconds": round(runtime, 3),
        "psnr_y": round(metrics["psnr_y"], 6),
        "ssim_y": round(metrics["ssim_y"], 6),
        "lpips": round(metrics["lpips"], 6),
        "score": round(metrics["score"], 6),
        "notes": args.notes,
    }
    append_csv(args.csv, row)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
