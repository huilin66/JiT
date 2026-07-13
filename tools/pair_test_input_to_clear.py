#!/usr/bin/env python3
"""Pair flat rainy test inputs with grouped training Drop folders by visual similarity.

The script first groups flat test-input images into same-scene sequences, then
selects one representative image from each test group and matches it to the most
similar group under DayRainDrop_Train/Drop.  When a Clear root is provided, the
corresponding Clear image from the matched group is also recorded for GT use.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_TEST_DIR = r"\\158.132.186.40\isds\huilin\tp\eccv_dn\test-input"
DEFAULT_MATCH_ROOTS = [
    r"\\158.132.186.40\isds\huilin\tp\eccv_dn\DayRainDrop_Train\Drop",
    r"\\158.132.186.40\isds\huilin\tp\eccv_dn\NightRainDrop_Train\Drop",
]
DEFAULT_CLEAR_ROOTS = [
    r"\\158.132.186.40\isds\huilin\tp\eccv_dn\DayRainDrop_Train\Clear",
    r"\\158.132.186.40\isds\huilin\tp\eccv_dn\NightRainDrop_Train\Clear",
]


@dataclass
class ImageFeature:
    low_rgb: np.ndarray
    gray: np.ndarray
    grad: np.ndarray
    hist: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help="Flat rainy test-input directory.")
    parser.add_argument(
        "--match-roots",
        nargs="+",
        default=DEFAULT_MATCH_ROOTS,
        help="Grouped training Drop roots used for matching. Defaults to Day and Night.",
    )
    parser.add_argument(
        "--clear-roots",
        nargs="*",
        default=DEFAULT_CLEAR_ROOTS,
        help="Optional grouped Clear roots for corresponding GT. Defaults to Day and Night.",
    )
    parser.add_argument("--match-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--clear-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="demo/test_drop_pairing", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--feature-size", type=int, default=48, help="Low-resolution feature size.")
    parser.add_argument(
        "--max-match-images",
        type=int,
        default=8,
        help="Max images sampled per matched Drop group to build its visual representative. 0 = all.",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.0,
        help="Adjacent test-image split threshold. 0 = estimate automatically.",
    )
    parser.add_argument("--min-group-size", type=int, default=2, help="Merge groups smaller than this into neighbors.")
    parser.add_argument(
        "--group-mode",
        choices=["visual", "numeric-gap", "none"],
        default="visual",
        help="How to group test-input before matching.",
    )
    parser.add_argument(
        "--test-groups-csv",
        default="",
        help="Optional CSV with columns test_file,test_group_id to reuse an existing grouping.",
    )
    parser.add_argument(
        "--numeric-gap",
        type=int,
        default=1,
        help="For --group-mode numeric-gap: start a new group when numeric filename gap is larger than this.",
    )
    parser.add_argument("--unique-match", action="store_true", help="Greedily avoid assigning the same Drop group twice.")
    parser.add_argument(
        "--copy-paired-clear-dir",
        default="",
        help="Optional flat output directory. Corresponding Clear representative is copied as each test filename.",
    )
    parser.add_argument(
        "--copy-paired-drop-dir",
        default="",
        help="Optional flat output directory. Matched Drop representative is copied as each test filename.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(directory: Path, recursive: bool = False) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def natural_key(path: Path) -> tuple[int, str]:
    number = numeric_stem(path)
    return (number if number is not None else 10**12, path.name)


def dataset_label(root: Path) -> str:
    parent = root.parent.name
    lower = parent.lower()
    if "night" in lower:
        return "night"
    if "day" in lower:
        return "day"
    return parent or root.name


def group_key_for_root(root: Path, group_name: str) -> str:
    return f"{dataset_label(root)}_{group_name}"


def sample_evenly(paths: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_count).round().astype(int)
    return [paths[int(index)] for index in sorted(set(indices))]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def color_histogram(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    channels = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        channels.append(hist)
    return np.concatenate(channels, axis=0)


def extract_feature(path: Path, size: int) -> ImageFeature:
    image = load_rgb(path)
    low = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    # Heavy blur suppresses high-frequency raindrops and keeps the scene layout.
    low = cv2.GaussianBlur(low, (0, 0), sigmaX=2.2, sigmaY=2.2)
    gray = (0.299 * low[:, :, 0] + 0.587 * low[:, :, 1] + 0.114 * low[:, :, 2]).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad /= max(float(np.percentile(grad, 99)), 1e-6)
    grad = np.clip(grad, 0.0, 1.0)
    return ImageFeature(
        low_rgb=low.astype(np.float32),
        gray=gray,
        grad=grad.astype(np.float32),
        hist=color_histogram(low),
    )


def median_feature(features: list[ImageFeature]) -> ImageFeature:
    return ImageFeature(
        low_rgb=np.median(np.stack([f.low_rgb for f in features], axis=0), axis=0).astype(np.float32),
        gray=np.median(np.stack([f.gray for f in features], axis=0), axis=0).astype(np.float32),
        grad=np.median(np.stack([f.grad for f in features], axis=0), axis=0).astype(np.float32),
        hist=np.median(np.stack([f.hist for f in features], axis=0), axis=0).astype(np.float32),
    )


def feature_distance(a: ImageFeature, b: ImageFeature) -> float:
    low = float(np.mean(np.abs(a.low_rgb - b.low_rgb)))
    gray = float(np.mean(np.abs(a.gray - b.gray)))
    grad = float(np.mean(np.abs(a.grad - b.grad)))
    hist = float(np.mean(np.abs(a.hist - b.hist)))
    return 0.62 * low + 0.18 * gray + 0.12 * grad + 0.08 * hist


def auto_threshold(distances: list[float]) -> float:
    if len(distances) < 3:
        return float("inf")
    values = np.asarray(distances, dtype=np.float32)
    c1, c2 = np.percentile(values, [30, 90]).astype(np.float32)
    for _ in range(20):
        d1 = np.abs(values - c1)
        d2 = np.abs(values - c2)
        left = values[d1 <= d2]
        right = values[d1 > d2]
        if len(left) == 0 or len(right) == 0:
            break
        new_c1 = float(left.mean())
        new_c2 = float(right.mean())
        if abs(new_c1 - c1) + abs(new_c2 - c2) < 1e-6:
            break
        c1, c2 = new_c1, new_c2
    lo, hi = sorted([float(c1), float(c2)])
    if hi < lo * 1.25:
        return float(np.percentile(values, 92))
    return (lo + hi) * 0.5


def merge_small_groups(groups: list[list[Path]], min_group_size: int) -> list[list[Path]]:
    if min_group_size <= 1 or len(groups) <= 1:
        return groups
    merged: list[list[Path]] = []
    for group in groups:
        if len(group) < min_group_size and merged:
            merged[-1].extend(group)
        else:
            merged.append(group)
    if len(merged) > 1 and len(merged[0]) < min_group_size:
        merged[1] = merged[0] + merged[1]
        merged = merged[1:]
    return merged


def group_by_visual(paths: list[Path], features: dict[str, ImageFeature], threshold: float, min_group_size: int):
    distances = [
        feature_distance(features[paths[i - 1].name], features[paths[i].name])
        for i in range(1, len(paths))
    ]
    split_threshold = threshold if threshold > 0 else auto_threshold(distances)
    groups: list[list[Path]] = [[paths[0]]]
    for path, distance in zip(paths[1:], distances):
        if distance > split_threshold:
            groups.append([path])
        else:
            groups[-1].append(path)
    return merge_small_groups(groups, min_group_size), split_threshold, distances


def group_by_numeric_gap(paths: list[Path], gap: int, min_group_size: int):
    groups: list[list[Path]] = [[paths[0]]]
    for prev, current in zip(paths[:-1], paths[1:]):
        prev_num, cur_num = numeric_stem(prev), numeric_stem(current)
        if prev_num is not None and cur_num is not None and cur_num - prev_num > gap:
            groups.append([current])
        else:
            groups[-1].append(current)
    return merge_small_groups(groups, min_group_size)


def build_test_groups(paths: list[Path], features: dict[str, ImageFeature], args: argparse.Namespace):
    if not paths:
        raise RuntimeError("No test images found")
    if args.test_groups_csv:
        by_name = {path.name: path for path in paths}
        groups: dict[int, list[Path]] = {}
        with Path(args.test_groups_csv).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = row["test_file"]
                if name not in by_name:
                    raise RuntimeError(f"Grouped file is missing from --test-dir: {name}")
                groups.setdefault(int(row["test_group_id"]), []).append(by_name[name])
        return [sorted(groups[key], key=natural_key) for key in sorted(groups)], 0.0, []
    if args.group_mode == "none":
        return [[path] for path in paths], 0.0, []
    if args.group_mode == "numeric-gap":
        return group_by_numeric_gap(paths, args.numeric_gap, args.min_group_size), 0.0, []
    return group_by_visual(paths, features, args.group_threshold, args.min_group_size)


def grouped_images(root: Path, label: str) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for directory in sorted([path for path in root.iterdir() if path.is_dir()], key=natural_key):
        images = list_images(directory, recursive=False)
        if images:
            key = group_key_for_root(root, directory.name)
            groups[key] = {
                "dataset": dataset_label(root),
                "group": directory.name,
                "root": root,
                "paths": sorted(images, key=natural_key),
            }
    if not groups:
        raise RuntimeError(f"No {label} groups found in: {root}")
    return groups


def grouped_images_from_roots(roots: list[Path], label: str) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for root in roots:
        current = grouped_images(root, label)
        overlap = set(merged) & set(current)
        if overlap:
            raise RuntimeError(f"Duplicate {label} group keys: {sorted(overlap)[:5]}")
        merged.update(current)
    return dict(sorted(merged.items()))


def clear_root_by_dataset(roots: list[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for root in roots:
        mapping[dataset_label(root)] = root
    return mapping


def representative_path(paths: list[Path], features: dict[str, ImageFeature]) -> Path:
    if len(paths) == 1:
        return paths[0]
    center = median_feature([features[path.name] for path in paths])
    return min(paths, key=lambda path: feature_distance(features[path.name], center))


def assign_unique(group_rows: list[dict]) -> list[dict]:
    used: set[str] = set()
    assigned: list[dict] = []
    for row in sorted(group_rows, key=lambda item: item["margin"], reverse=True):
        candidates = row["_candidates"]
        chosen = next((item for item in candidates if item[0] not in used), candidates[0])
        used.add(chosen[0])
        updated = dict(row)
        updated["match_group"] = chosen[0]
        updated["best_distance"] = chosen[1]
        second = next((item for item in candidates if item[0] != chosen[0]), chosen)
        updated["second_match_group"] = second[0]
        updated["second_distance"] = second[1]
        updated["margin"] = second[1] - chosen[1]
        updated.pop("_candidates", None)
        assigned.append(updated)
    return sorted(assigned, key=lambda item: item["test_group_id"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}; use --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    test_dir = Path(args.test_dir)
    match_roots = [Path(args.match_root)] if args.match_root else [Path(path) for path in args.match_roots]
    clear_roots = [Path(args.clear_root)] if args.clear_root else [Path(path) for path in args.clear_roots]
    clear_roots_by_dataset = clear_root_by_dataset(clear_roots) if clear_roots else {}
    output_dir = Path(args.output_dir)
    audit_output_dir(output_dir, args.overwrite)

    test_paths = sorted(list_images(test_dir, recursive=False), key=natural_key)
    if not test_paths:
        raise RuntimeError(f"No test images found in: {test_dir}")

    print(f"Test images: {len(test_paths)}")
    test_features = {
        path.name: extract_feature(path, args.feature_size)
        for path in tqdm(test_paths, desc="Test features", unit="img")
    }

    grouped_match = grouped_images_from_roots(match_roots, "match")
    grouped_clear = grouped_images_from_roots(clear_roots, "Clear") if clear_roots else {}
    print(f"Match Drop groups: {len(grouped_match)}")
    print(f"Clear groups: {len(grouped_clear)}")
    match_features: dict[str, ImageFeature] = {}
    match_rep_paths: dict[str, Path] = {}
    for group_name, info in tqdm(grouped_match.items(), desc="Drop group features", unit="group"):
        paths = info["paths"]
        sampled = sample_evenly(paths, args.max_match_images)
        local_features = {path.name: extract_feature(path, args.feature_size) for path in sampled}
        match_features[group_name] = median_feature(list(local_features.values()))
        match_rep_paths[group_name] = representative_path(sampled, local_features)

    test_groups, split_threshold, adjacent_distances = build_test_groups(test_paths, test_features, args)
    print(f"Test groups: {len(test_groups)}")
    if args.group_mode == "visual":
        print(f"Visual split threshold: {split_threshold:.6f}")

    test_group_rows: list[dict] = []
    group_match_rows: list[dict] = []
    for group_id, group_paths in enumerate(test_groups, start=1):
        rep = representative_path(group_paths, test_features)
        # Use the whole test group, not only one representative frame, for robust scene matching.
        group_feature = median_feature([test_features[path.name] for path in group_paths])
        candidates = sorted(
            ((match_group, feature_distance(group_feature, feature)) for match_group, feature in match_features.items()),
            key=lambda item: item[1],
        )
        best_group, best_distance = candidates[0]
        second_group, second_distance = candidates[1] if len(candidates) > 1 else candidates[0]
        match_rep = match_rep_paths[best_group]
        match_info = grouped_match[best_group]

        group_match_rows.append({
            "test_group_id": group_id,
            "test_group_size": len(group_paths),
            "test_start": group_paths[0].name,
            "test_end": group_paths[-1].name,
            "test_rep": rep.name,
            "match_group": best_group,
            "match_dataset": match_info["dataset"],
            "match_scene": match_info["group"],
            "match_drop_file": match_rep.name,
            "match_drop_path": str(match_rep),
            "best_distance": best_distance,
            "second_match_group": second_group,
            "second_distance": second_distance,
            "margin": second_distance - best_distance,
            "_candidates": candidates,
        })

        for path in group_paths:
            test_group_rows.append({
                "test_file": path.name,
                "test_group_id": group_id,
                "is_group_rep": int(path.name == rep.name),
            })

    if args.unique_match:
        group_match_rows = assign_unique(group_match_rows)
    else:
        for row in group_match_rows:
            row.pop("_candidates", None)
    for row in group_match_rows:
        match_rep = match_rep_paths[row["match_group"]]
        match_info = grouped_match[row["match_group"]]
        row["match_dataset"] = match_info["dataset"]
        row["match_scene"] = match_info["group"]
        row["match_drop_file"] = match_rep.name
        row["match_drop_path"] = str(match_rep)

    match_by_group = {row["test_group_id"]: row for row in group_match_rows}
    pair_rows: list[dict] = []
    for row in test_group_rows:
        match = match_by_group[row["test_group_id"]]
        match_drop_path = match_rep_paths[match["match_group"]]
        clear_root = clear_roots_by_dataset.get(match["match_dataset"])
        clear_path = clear_root / match["match_scene"] / match_drop_path.name if clear_root else Path("")
        clear_exists = bool(clear_path and clear_path.exists())
        pair_rows.append({
            "test_file": row["test_file"],
            "test_group_id": row["test_group_id"],
            "is_group_rep": row["is_group_rep"],
            "match_group": match["match_group"],
            "match_dataset": match["match_dataset"],
            "match_scene": match["match_scene"],
            "match_drop_file": match_drop_path.name,
            "match_drop_path": str(match_drop_path),
            "clear_dataset": match["match_dataset"] if clear_exists else "",
            "clear_group": match["match_scene"] if clear_exists else "",
            "clear_file": clear_path.name if clear_exists else "",
            "clear_path": str(clear_path) if clear_exists else "",
            "test_group_rep": match["test_rep"],
            "best_distance": match["best_distance"],
            "second_match_group": match["second_match_group"],
            "second_distance": match["second_distance"],
            "margin": match["margin"],
        })

    write_csv(output_dir / "test_groups.csv", test_group_rows)
    write_csv(output_dir / "group_matches.csv", group_match_rows)
    write_csv(output_dir / "pair_map.csv", pair_rows)

    if args.copy_paired_clear_dir:
        paired_dir = Path(args.copy_paired_clear_dir)
        audit_output_dir(paired_dir, args.overwrite)
        for row in tqdm(pair_rows, desc="Copy paired Clear", unit="img"):
            if not row["clear_path"]:
                raise RuntimeError("Cannot copy Clear files because no corresponding clear_path was found")
            shutil.copy2(row["clear_path"], paired_dir / row["test_file"])

    if args.copy_paired_drop_dir:
        paired_dir = Path(args.copy_paired_drop_dir)
        audit_output_dir(paired_dir, args.overwrite)
        for row in tqdm(pair_rows, desc="Copy paired Drop", unit="img"):
            shutil.copy2(row["match_drop_path"], paired_dir / row["test_file"])

    if adjacent_distances:
        distance_rows = [
            {
                "prev_file": test_paths[i - 1].name,
                "file": test_paths[i].name,
                "adjacent_distance": distance,
                "split": int(distance > split_threshold),
            }
            for i, distance in enumerate(adjacent_distances, start=1)
        ]
        write_csv(output_dir / "adjacent_distances.csv", distance_rows)

    manifest = {
        "test_dir": str(test_dir),
        "match_roots": [str(path) for path in match_roots],
        "clear_roots": [str(path) for path in clear_roots],
        "output_dir": str(output_dir.resolve()),
        "test_images": len(test_paths),
        "test_groups": len(test_groups),
        "match_groups": len(grouped_match),
        "clear_groups": len(grouped_clear),
        "group_mode": args.group_mode,
        "group_threshold": split_threshold,
        "feature_size": args.feature_size,
        "max_match_images": args.max_match_images,
        "unique_match": bool(args.unique_match),
        "copy_paired_clear_dir": args.copy_paired_clear_dir,
        "copy_paired_drop_dir": args.copy_paired_drop_dir,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
