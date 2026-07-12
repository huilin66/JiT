#!/usr/bin/env python3
"""Fuse aligned predictions that share one clean image within each scene.

Each scene contains multiple rainy observations of the same clean background.
For every model, this tool first computes a robust across-observation consensus.
With multiple models, the per-model consensuses are then fused.  The resulting
scene image is written under every original filename in that scene so the
archive remains submission-compatible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_GROUP_REGEX = r"(?i)^(day|night)(?:raindrop)?(?:__|[_-])(\d+)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dirs", nargs="+", required=True,
                        help="One or more flat PNG prediction directories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX,
                        help="Regex matched against filename stem; capture groups form scene id")
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--min-scene-size", type=int, default=2)
    parser.add_argument("--scene-method", choices=["median", "trimmed_mean", "mean"], default="median")
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--model-method", choices=["weighted_mean", "median"], default="weighted_mean")
    parser.add_argument("--model-weights", default="",
                        help="Comma-separated nonnegative weights; default equal")
    parser.add_argument("--scene-weight", type=float, default=1.0,
                        help="Output=(1-w)*per-image model fusion+w*scene consensus")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def png_map(directory: str) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {root}")
    files = {path.name: path for path in root.glob("*.png")}
    if not files:
        raise ValueError(f"No flat PNG files in: {root}")
    return files


def aligned_maps(directories: list[str]) -> tuple[list[str], list[dict[str, Path]]]:
    maps = [png_map(directory) for directory in directories]
    names = sorted(maps[0])
    for directory, current in zip(directories[1:], maps[1:]):
        if set(current) != set(names):
            missing = sorted(set(names) - set(current))
            extra = sorted(set(current) - set(names))
            raise ValueError(
                f"PNG filename mismatch in {directory}: missing={len(missing)}, extra={len(extra)}"
            )
    return names, maps


def scene_key(name: str, pattern: re.Pattern[str]) -> str:
    match = pattern.match(Path(name).stem)
    if match is None:
        raise ValueError(f"Cannot parse scene id from filename: {name!r}")
    if match.groups():
        return "_".join(str(part).lower() for part in match.groups())
    return match.group(0).lower()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(array: np.ndarray, path: Path) -> None:
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path, format="PNG", compress_level=6)


def reduce_scene(stack: np.ndarray, method: str, trim_fraction: float) -> np.ndarray:
    if method == "median":
        return np.median(stack, axis=0).astype(np.float32)
    if method == "mean":
        return np.mean(stack, axis=0, dtype=np.float32)
    count = stack.shape[0]
    trim = int(np.floor(count * trim_fraction))
    if trim == 0:
        return np.mean(stack, axis=0, dtype=np.float32)
    if 2 * trim >= count:
        raise ValueError(f"trim_fraction={trim_fraction} removes all {count} scene observations")
    ordered = np.sort(stack, axis=0)
    return np.mean(ordered[trim:count - trim], axis=0, dtype=np.float32)


def parse_weights(text: str, count: int) -> np.ndarray:
    if text:
        values = np.asarray([float(part.strip()) for part in text.split(",")], dtype=np.float32)
        if len(values) != count:
            raise ValueError(f"Expected {count} model weights, got {len(values)}")
    else:
        values = np.ones(count, dtype=np.float32)
    if np.any(values < 0) or not float(values.sum()) > 0:
        raise ValueError("Model weights must be nonnegative and sum to a positive value")
    return values / values.sum()


def fuse_models(stack: np.ndarray, method: str, weights: np.ndarray) -> np.ndarray:
    if method == "median":
        return np.median(stack, axis=0).astype(np.float32)
    return np.tensordot(weights, stack, axes=(0, 0)).astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.scene_weight <= 1.0:
        raise ValueError("--scene-weight must be in [0, 1]")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("--trim-fraction must be in [0, 0.5)")
    if args.min_scene_size < 2:
        raise ValueError("--min-scene-size must be at least 2")

    names, maps = aligned_maps(args.pred_dirs)
    if args.expected_count and len(names) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} PNGs, found {len(names)}")
    pattern = re.compile(args.group_regex)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[scene_key(name, pattern)].append(name)
    too_small = {key: len(value) for key, value in groups.items() if len(value) < args.min_scene_size}
    if too_small:
        preview = dict(list(sorted(too_small.items()))[:10])
        raise ValueError(f"Scenes smaller than --min-scene-size: {preview}")

    output_dir = Path(args.output_dir)
    existing = list(output_dir.glob("*.png")) if output_dir.is_dir() else []
    if existing and not args.overwrite:
        raise FileExistsError(f"Output directory already contains PNGs: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.model_weights, len(maps))
    diagnostics: list[dict[str, object]] = []

    for key in tqdm(sorted(groups), desc="Scene consensus", unit="scene"):
        group_names = sorted(groups[key])
        model_frames: list[list[np.ndarray]] = []
        expected_shape: tuple[int, ...] | None = None
        for current in maps:
            frames = [load_rgb(current[name]) for name in group_names]
            shapes = {frame.shape for frame in frames}
            if len(shapes) != 1:
                raise ValueError(f"Shape mismatch within scene {key}: {sorted(shapes)}")
            if expected_shape is None:
                expected_shape = frames[0].shape
            elif frames[0].shape != expected_shape:
                raise ValueError(f"Shape mismatch between models for scene {key}")
            model_frames.append(frames)

        model_consensus = np.stack([
            reduce_scene(np.stack(frames, axis=0), args.scene_method, args.trim_fraction)
            for frames in model_frames
        ], axis=0)
        scene_consensus = fuse_models(model_consensus, args.model_method, weights)

        per_model_dispersion = [
            float(np.mean(np.abs(np.stack(frames, axis=0) - model_consensus[index])))
            for index, frames in enumerate(model_frames)
        ]
        model_disagreement = float(np.mean(np.abs(model_consensus - scene_consensus)))

        for image_index, name in enumerate(group_names):
            per_image_models = np.stack(
                [model_frames[model_index][image_index] for model_index in range(len(maps))], axis=0
            )
            per_image = fuse_models(per_image_models, args.model_method, weights)
            result = (1.0 - args.scene_weight) * per_image + args.scene_weight * scene_consensus
            save_rgb(result, output_dir / name)

        diagnostics.append({
            "scene": key,
            "images": len(group_names),
            "mean_within_model_mae": float(np.mean(per_model_dispersion)),
            "model_disagreement_mae": model_disagreement,
            "width": int(expected_shape[1]),
            "height": int(expected_shape[0]),
        })

    output_names = sorted(path.name for path in output_dir.glob("*.png"))
    if output_names != names:
        raise RuntimeError(f"Output filename audit failed: expected={len(names)}, output={len(output_names)}")

    archive_path = Path(args.archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(output_dir / name, arcname=name)

    diagnostics_path = output_dir / "scene_consensus_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    manifest = {
        "count": len(names),
        "scenes": len(groups),
        "scene_size_min": min(map(len, groups.values())),
        "scene_size_max": max(map(len, groups.values())),
        "pred_dirs": [str(Path(path).resolve()) for path in args.pred_dirs],
        "scene_method": args.scene_method,
        "model_method": args.model_method,
        "normalized_model_weights": weights.tolist(),
        "scene_weight": args.scene_weight,
        "group_regex": args.group_regex,
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha256_file(archive_path),
    }
    (output_dir / "scene_consensus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
