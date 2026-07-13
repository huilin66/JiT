#!/usr/bin/env python3
"""Build pseudo labels directly from original same-scene rainy inputs.

This implements the transductive idea used by several NTIRE raindrop teams:
frames from the same scene share a stable background while raindrops move or
change between observations, so a robust per-pixel scene consensus can suppress
inconsistent raindrops and serve as pseudo ground truth.

Typical usage:
  python tools/original_scene_pseudo.py \
    --input-dir /path/to/Drop \
    --output-dir /path/to/pseudo_scene_median/PseudoGT \
    --mask-dir /path/to/pseudo_scene_median/masks \
    --group-regex "(?i)^(day|night)[_-](\\d+)" \
    --method median
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_GROUP_REGEX = r"(?i)^(day|night)(?:raindrop)?(?:__|[_-])(\d+)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=r"//158.132.186.40/isds/huilin/tp/eccv_dn/RainDrop_Train/Drop", help="Flat directory of original rainy images.")
    parser.add_argument("--output-dir", default=r'demo/pseduo_gt', help="Output pseudo-GT directory.")
    parser.add_argument("--mask-dir", default=r"demo/pseduo_masks", help="Optional output directory for soft rain masks.")
    parser.add_argument(
        "--group-regex",
        default=DEFAULT_GROUP_REGEX,
        help="Regex matched against filename stem; capture groups form scene id.",
    )
    parser.add_argument("--min-scene-size", type=int, default=2)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--method", choices=["median", "trimmed_mean", "mean"], default="median")
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument(
        "--target-mode",
        choices=["consensus", "mask_blend"],
        default="consensus",
        help=(
            "consensus: write the scene consensus as pseudo GT for every image. "
            "mask_blend: use consensus only in changed regions and keep more input elsewhere."
        ),
    )
    parser.add_argument("--mask-t1", type=float, default=0.02)
    parser.add_argument("--mask-t2", type=float, default=0.08)
    parser.add_argument("--mask-blur", type=float, default=3.0)
    parser.add_argument(
        "--input-keep",
        type=float,
        default=0.85,
        help="For target-mode=mask_blend: non-mask region is input_keep*input + (1-input_keep)*consensus.",
    )
    parser.add_argument(
        "--consensus-weight",
        type=float,
        default=1.0,
        help="Final pseudo = consensus_weight*pseudo + (1-consensus_weight)*input.",
    )
    parser.add_argument("--save-scene-consensus", default="", help="Optional directory for one consensus PNG per scene.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def scene_key(name: str, pattern: re.Pattern[str]) -> str:
    match = pattern.match(Path(name).stem)
    if match is None:
        raise ValueError(
            f"Cannot parse scene id from filename: {name!r}. "
            "Pass --group-regex for your naming convention."
        )
    if match.groups():
        return "_".join(str(part).lower() for part in match.groups())
    return match.group(0).lower()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(array: np.ndarray, path: Path) -> None:
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path, format="PNG", compress_level=6)


def save_gray(array: np.ndarray, path: Path) -> None:
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path, format="PNG", compress_level=6)


def reduce_stack(stack: np.ndarray, method: str, trim_fraction: float) -> np.ndarray:
    if method == "median":
        return np.median(stack, axis=0).astype(np.float32)
    if method == "mean":
        return np.mean(stack, axis=0, dtype=np.float32)

    count = stack.shape[0]
    trim = int(np.floor(count * trim_fraction))
    if trim == 0:
        return np.mean(stack, axis=0, dtype=np.float32)
    if 2 * trim >= count:
        raise ValueError(f"--trim-fraction={trim_fraction} removes all {count} scene images")
    ordered = np.sort(stack, axis=0)
    return np.mean(ordered[trim:count - trim], axis=0, dtype=np.float32)


def soft_diff_mask(image: np.ndarray, consensus: np.ndarray, t1: float, t2: float, blur: float) -> np.ndarray:
    diff = np.mean(np.abs(image - consensus), axis=2)
    mask = np.clip((diff - t1) / max(t2 - t1, 1e-6), 0.0, 1.0).astype(np.float32)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur)
    return np.clip(mask, 0.0, 1.0)


def build_groups(paths: list[Path], group_regex: str) -> dict[str, list[Path]]:
    pattern = re.compile(group_regex)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[scene_key(path.name, pattern)].append(path)
    return dict(sorted(groups.items()))


def audit_output_dir(path: Path, overwrite: bool) -> None:
    if path.is_dir() and any(path.glob("*.png")) and not overwrite:
        raise FileExistsError(f"Output directory already has PNG files: {path}; use --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.min_scene_size < 2:
        raise ValueError("--min-scene-size must be at least 2")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("--trim-fraction must be in [0, 0.5)")
    if not 0.0 <= args.input_keep <= 1.0:
        raise ValueError("--input-keep must be in [0, 1]")
    if not 0.0 <= args.consensus_weight <= 1.0:
        raise ValueError("--consensus-weight must be in [0, 1]")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    scene_dir = Path(args.save_scene_consensus) if args.save_scene_consensus else None

    paths = list_images(input_dir)
    if not paths:
        raise RuntimeError(f"No input images found in: {input_dir}")
    if args.expected_count and len(paths) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} images, found {len(paths)}")

    groups = build_groups(paths, args.group_regex)
    too_small = {key: len(value) for key, value in groups.items() if len(value) < args.min_scene_size}
    if too_small:
        preview = dict(list(too_small.items())[:10])
        raise ValueError(f"Scenes smaller than --min-scene-size: {preview}")

    audit_output_dir(output_dir, args.overwrite)
    if mask_dir:
        audit_output_dir(mask_dir, args.overwrite)
    if scene_dir:
        audit_output_dir(scene_dir, args.overwrite)

    diagnostics: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []

    for key, group_paths in tqdm(groups.items(), desc="Original scene pseudo", unit="scene"):
        frames = [load_rgb(path) for path in group_paths]
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(f"Shape mismatch within scene {key}: {sorted(shapes)}")

        stack = np.stack(frames, axis=0)
        consensus = reduce_stack(stack, args.method, args.trim_fraction)
        if scene_dir:
            save_rgb(consensus, scene_dir / f"{key}.png")

        scene_mae = float(np.mean(np.abs(stack - consensus)))
        mask_means: list[float] = []
        for path, image in zip(group_paths, frames):
            mask = soft_diff_mask(
                image,
                consensus,
                t1=args.mask_t1,
                t2=args.mask_t2,
                blur=args.mask_blur,
            )
            mask_means.append(float(mask.mean()))

            if args.target_mode == "consensus":
                pseudo = consensus
            else:
                mask_3c = mask[:, :, None]
                non_mask = args.input_keep * image + (1.0 - args.input_keep) * consensus
                pseudo = mask_3c * consensus + (1.0 - mask_3c) * non_mask

            if args.consensus_weight < 1.0:
                pseudo = args.consensus_weight * pseudo + (1.0 - args.consensus_weight) * image

            save_rgb(pseudo, output_dir / path.name)
            if mask_dir:
                save_gray(mask, mask_dir / f"{path.stem}_mask.png")

            per_image_rows.append({
                "filename": path.name,
                "scene": key,
                "mask_mean": float(mask.mean()),
                "input_consensus_mae": float(np.mean(np.abs(image - consensus))),
            })

        diagnostics.append({
            "scene": key,
            "images": len(group_paths),
            "width": int(stack.shape[2]),
            "height": int(stack.shape[1]),
            "input_consensus_mae": scene_mae,
            "mask_mean_avg": float(np.mean(mask_means)),
            "mask_mean_min": float(np.min(mask_means)),
            "mask_mean_max": float(np.max(mask_means)),
        })

    output_names = sorted(path.name for path in output_dir.glob("*.png"))
    expected_names = sorted(path.name for path in paths)
    if output_names != expected_names:
        raise RuntimeError("Output filename audit failed")

    diagnostics_path = output_dir / "scene_pseudo_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    image_csv_path = output_dir / "scene_pseudo_per_image.csv"
    with image_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)

    manifest = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()) if mask_dir else "",
        "scene_consensus_dir": str(scene_dir.resolve()) if scene_dir else "",
        "count": len(paths),
        "scenes": len(groups),
        "scene_size_min": min(len(value) for value in groups.values()),
        "scene_size_max": max(len(value) for value in groups.values()),
        "method": args.method,
        "trim_fraction": args.trim_fraction,
        "target_mode": args.target_mode,
        "group_regex": args.group_regex,
        "mask_t1": args.mask_t1,
        "mask_t2": args.mask_t2,
        "mask_blur": args.mask_blur,
        "input_keep": args.input_keep,
        "consensus_weight": args.consensus_weight,
        "diagnostics_csv": str(diagnostics_path.resolve()),
        "per_image_csv": str(image_csv_path.resolve()),
    }
    manifest_path = output_dir / "scene_pseudo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
