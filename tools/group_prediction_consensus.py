#!/usr/bin/env python3
"""Build a flat submission folder by replacing each test group with one consensus prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, help="Flat test prediction directory.")
    parser.add_argument(
        "--test-group-dir",
        required=True,
        help="Directory containing test group subfolders, e.g. group_001_x-y/*.png.",
    )
    parser.add_argument("--output-dir", required=True, help="Flat output prediction directory.")
    parser.add_argument("--archive-path", default="", help="Optional zip archive path.")
    parser.add_argument(
        "--method",
        choices=[
            "closest_to_median",
            "median",
            "trimmed_mean",
            "mean",
            "inverse_median_distance",
            "sharpest",
        ],
        default="closest_to_median",
        help="How to build the one group prediction.",
    )
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument(
        "--group-weight",
        type=float,
        default=1.0,
        help="Output=(1-w)*original_prediction+w*group_consensus. Use 1.0 to replace all group images.",
    )
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--copy-ungrouped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_files(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prediction_map(pred_dir: Path) -> dict[str, Path]:
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    files = image_files(pred_dir, recursive=False)
    if not files:
        raise RuntimeError(f"No prediction images found in: {pred_dir}")
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate prediction filenames in: {pred_dir}")
    return {path.name: path for path in files}


def load_groups(test_group_dir: Path) -> dict[str, list[str]]:
    if not test_group_dir.is_dir():
        raise FileNotFoundError(f"test_group directory not found: {test_group_dir}")

    groups: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for group_dir in sorted(path for path in test_group_dir.iterdir() if path.is_dir()):
        names = [path.name for path in image_files(group_dir, recursive=False)]
        if not names:
            continue
        for name in names:
            if name in owner:
                raise RuntimeError(f"Image {name} appears in both {owner[name]} and {group_dir.name}")
            owner[name] = group_dir.name
        groups[group_dir.name] = names

    if not groups:
        raise RuntimeError(f"No non-empty group folders found in: {test_group_dir}")
    return groups


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        existing = [path for path in output_dir.iterdir() if path.is_file()]
        if existing and not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
        if overwrite:
            for path in existing:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(array: np.ndarray, path: Path) -> None:
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path, format="PNG", compress_level=6)


def sharpness_score(array: np.ndarray) -> float:
    gray = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    gy, gx = np.gradient(gray)
    return float(np.mean(gx * gx + gy * gy))


def trimmed_mean(stack: np.ndarray, trim_fraction: float) -> np.ndarray:
    count = stack.shape[0]
    trim = int(np.floor(count * trim_fraction))
    if trim == 0:
        return np.mean(stack, axis=0, dtype=np.float32)
    if 2 * trim >= count:
        raise ValueError(f"trim_fraction={trim_fraction} removes all {count} group images")
    ordered = np.sort(stack, axis=0)
    return np.mean(ordered[trim:count - trim], axis=0, dtype=np.float32)


def build_consensus(
    names: list[str],
    arrays: list[np.ndarray],
    method: str,
    trim_fraction: float,
) -> tuple[np.ndarray, str, float, float]:
    stack = np.stack(arrays, axis=0)
    median = np.median(stack, axis=0).astype(np.float32)
    distances = np.asarray([float(np.mean(np.abs(array - median))) for array in arrays], dtype=np.float32)
    sharpness = np.asarray([sharpness_score(array) for array in arrays], dtype=np.float32)

    selected_name = ""
    selected_score = 0.0
    if method == "median":
        consensus = median
    elif method == "mean":
        consensus = np.mean(stack, axis=0, dtype=np.float32)
    elif method == "trimmed_mean":
        consensus = trimmed_mean(stack, trim_fraction)
    elif method == "inverse_median_distance":
        weights = 1.0 / np.maximum(distances, 1e-6)
        weights = weights / weights.sum()
        consensus = np.tensordot(weights, stack, axes=(0, 0)).astype(np.float32)
    elif method == "sharpest":
        selected_index = int(np.argmax(sharpness))
        consensus = arrays[selected_index]
        selected_name = names[selected_index]
        selected_score = float(sharpness[selected_index])
    else:
        selected_index = int(np.argmin(distances))
        consensus = arrays[selected_index]
        selected_name = names[selected_index]
        selected_score = float(distances[selected_index])

    return consensus, selected_name, selected_score, float(np.mean(distances))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_readme_path() -> Path:
    return Path(__file__).resolve().parents[1] / "readme.txt"


def write_archive(output_dir: Path, names: list[str], archive_path: Path, readme_path: Path) -> str:
    if not readme_path.is_file():
        raise FileNotFoundError(f"readme.txt not found: {readme_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(output_dir / name, arcname=name)
        archive.write(readme_path, arcname="readme.txt")
    return sha256_file(archive_path)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.group_weight <= 1.0:
        raise ValueError("--group-weight must be in [0, 1]")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("--trim-fraction must be in [0, 0.5)")
    if args.min_group_size < 1:
        raise ValueError("--min-group-size must be at least 1")

    pred_dir = Path(args.pred_dir)
    test_group_dir = Path(args.test_group_dir)
    output_dir = Path(args.output_dir)

    pred_paths = prediction_map(pred_dir)
    groups = load_groups(test_group_dir)
    if args.expected_count and len(pred_paths) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} predictions, found {len(pred_paths)}")

    grouped_names = {name for names in groups.values() for name in names}
    missing = sorted(name for name in grouped_names if name not in pred_paths)
    if missing:
        preview = "\n".join(f"  - {name}" for name in missing[:30])
        more = "" if len(missing) <= 30 else f"\n  ... and {len(missing) - 30} more"
        raise FileNotFoundError(f"Missing predictions for grouped images ({len(missing)}):\n{preview}{more}")
    if not args.copy_ungrouped and set(pred_paths) - grouped_names:
        raise RuntimeError("Predictions contain ungrouped files; use --copy-ungrouped or complete test_group.")

    prepare_output_dir(output_dir, args.overwrite)
    diagnostics: list[dict[str, object]] = []
    processed: set[str] = set()

    for group_name, names in tqdm(groups.items(), desc="Group consensus", unit="group"):
        if len(names) < args.min_group_size:
            for name in names:
                shutil.copy2(pred_paths[name], output_dir / name)
                processed.add(name)
            diagnostics.append({
                "group": group_name,
                "images": len(names),
                "method": "copied_small_group",
                "selected_file": "",
                "selected_score": 0.0,
                "mean_mae_to_group_median": 0.0,
                "width": 0,
                "height": 0,
            })
            continue

        arrays = [load_rgb(pred_paths[name]) for name in names]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(f"Prediction shape mismatch in {group_name}: {sorted(shapes)}")
        consensus, selected_name, selected_score, mean_distance = build_consensus(
            names, arrays, args.method, args.trim_fraction
        )

        for name, original in zip(names, arrays):
            if args.group_weight < 1.0:
                result = (1.0 - args.group_weight) * original + args.group_weight * consensus
            else:
                result = consensus
            save_rgb(result, output_dir / name)
            processed.add(name)

        shape = arrays[0].shape
        diagnostics.append({
            "group": group_name,
            "images": len(names),
            "method": args.method,
            "selected_file": selected_name,
            "selected_score": selected_score,
            "mean_mae_to_group_median": mean_distance,
            "width": int(shape[1]),
            "height": int(shape[0]),
        })

    ungrouped = sorted(set(pred_paths) - processed)
    if args.copy_ungrouped:
        for name in tqdm(ungrouped, desc="Copy ungrouped", unit="img"):
            shutil.copy2(pred_paths[name], output_dir / name)

    output_names = sorted(path.name for path in image_files(output_dir, recursive=False))
    expected_names = sorted(pred_paths)
    if output_names != expected_names:
        raise RuntimeError(
            f"Output filename audit failed: expected={len(expected_names)}, output={len(output_names)}"
        )

    diagnostics_path = output_dir / "group_consensus_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    archive_sha256 = ""
    archive = ""
    readme_path = repo_readme_path()
    if args.archive_path:
        archive_path = Path(args.archive_path)
        archive_sha256 = write_archive(output_dir, expected_names, archive_path, readme_path)
        archive = str(archive_path.resolve())

    manifest = {
        "pred_dir": str(pred_dir.resolve()),
        "test_group_dir": str(test_group_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "archive": archive,
        "archive_sha256": archive_sha256,
        "archive_readme": str(readme_path.resolve()) if args.archive_path else "",
        "method": args.method,
        "trim_fraction": args.trim_fraction,
        "group_weight": args.group_weight,
        "prediction_count": len(pred_paths),
        "group_count": len(groups),
        "grouped_prediction_count": len(grouped_names),
        "ungrouped_prediction_count": len(ungrouped),
        "min_group_size": min(len(names) for names in groups.values()),
        "max_group_size": max(len(names) for names in groups.values()),
    }
    (output_dir / "group_consensus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
