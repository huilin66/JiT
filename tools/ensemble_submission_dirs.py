#!/usr/bin/env python3
"""Average multiple PNG submission directories and package a ZIP."""

from __future__ import annotations

from tqdm import tqdm
import argparse
import csv
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


HISTORY_FIELDS = [
    "timestamp",
    "model_name",
    "archive_name",
    "checkpoint",
    "state_key",
    "architecture",
    "use_scene",
    "use_bg_subnet",
    "steps",
    "patch_size",
    "stride",
    "tile_batch_size",
    "input_dir",
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
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--weights", default="", help="Comma-separated weights. Default: uniform.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--history-csv", default="")
    parser.add_argument("--model-name", default="jit_ensemble")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--remove-images-after-zip", action="store_true")
    return parser.parse_args()


def parse_weights(raw, count):
    if not raw:
        return np.full(count, 1.0 / count, dtype=np.float64)
    weights = np.asarray([float(item) for item in raw.replace(";", ",").split(",")], dtype=np.float64)
    if weights.size != count:
        raise ValueError(f"Expected {count} weights, got {weights.size}")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return weights / total


def list_pngs(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    files = sorted(path.name for path in directory.glob("*.png"))
    if not files:
        raise RuntimeError(f"No PNG files found in {directory}")
    return files


def create_archive(image_dir, archive_path):
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in sorted(Path(image_dir).glob("*.png")):
            archive.write(image_path, arcname=image_path.name)


def append_history(path, row):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if path.exists():
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if header != HISTORY_FIELDS:
            raise RuntimeError(f"Existing history CSV has an incompatible header: {path}")
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    input_dirs = [Path(path) for path in args.input_dirs]
    weights = parse_weights(args.weights, len(input_dirs))
    names = list_pngs(input_dirs[0])
    for directory in input_dirs[1:]:
        current = list_pngs(directory)
        if current != names:
            raise RuntimeError(f"PNG file list mismatch: {directory}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for name in tqdm(names):
        accum = None
        for weight, directory in zip(weights, input_dirs):
            image = Image.open(directory / name).convert("RGB")
            array = np.asarray(image, dtype=np.float64)
            accum = array * weight if accum is None else accum + array * weight
        out = np.rint(np.clip(accum, 0.0, 255.0)).astype(np.uint8)
        Image.fromarray(out, mode="RGB").save(output_dir / name, format="PNG", optimize=True)
    runtime = time.perf_counter() - started

    archive_path = Path(args.archive_path)
    create_archive(output_dir, archive_path)
    append_history(
        args.history_csv,
        {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "model_name": args.model_name,
            "archive_name": archive_path.name,
            "checkpoint": "|".join(str(path) for path in input_dirs),
            "state_key": "ensemble",
            "architecture": "ensemble",
            "use_scene": "",
            "use_bg_subnet": "",
            "steps": "ensemble",
            "patch_size": "",
            "stride": "ensemble",
            "tile_batch_size": "",
            "input_dir": str(Path(args.input_dir).resolve()) if args.input_dir else "",
            "num_images": len(names),
            "runtime_seconds": round(runtime, 3),
            "psnr_y": "",
            "ssim_y": "",
            "lpips": "",
            "score": "",
            "notes": args.notes,
        },
    )
    print(f"Ensemble submission: {archive_path}")
    print(f"Inputs: {len(input_dirs)} dirs; images: {len(names)}")
    if args.remove_images_after_zip:
        import shutil

        shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
