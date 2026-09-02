#!/usr/bin/env python3
"""Build pseudo labels from manually paired test/drop or test/test groups."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_test_input_to_clear import (
    build_test_groups,
    extract_feature,
    list_images,
    natural_key,
    sample_evenly,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DROP_INPUT_DIR = r"\\158.132.186.40\isds\huilin\tp\eccv_dn\Drop"
DEFAULT_TEST_INPUT_DIR = r"\\158.132.186.40\isds\huilin\tp\eccv_dn\test-input"
DEFAULT_DROP_PRED_DIR = r"\\158.132.186.40\isds\huilin\tp\eccv_dn\jit_submit_best_model_ema1_s1_r16_hflip_rot90_best_20260708_101616"
DEFAULT_TEST_PRED_DIR = r"\\158.132.186.40\isds\huilin\tp\eccv_dn\jit_submit_best_model_ema1_s1_r16_hflip_rot90_best_20260710_151811"
DEFAULT_TEST_GROUPS_CSV = "demo/test_group_from_folders.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-input-dir", default=DEFAULT_DROP_INPUT_DIR)
    parser.add_argument("--test-input-dir", default=DEFAULT_TEST_INPUT_DIR)
    parser.add_argument("--drop-pred-dir", default=DEFAULT_DROP_PRED_DIR, help="Flat Drop prediction directory.")
    parser.add_argument("--test-pred-dir", default=DEFAULT_TEST_PRED_DIR, help="Flat test prediction directory.")
    parser.add_argument("--manual-pair-dir", default="demo/test_pari_manual")
    parser.add_argument(
        "--auto-filter-pairs",
        action="store_true",
        help="Filter pair candidate images into <output-root>/auto_filtered_pairs and use them as manual pairs.",
    )
    parser.add_argument(
        "--pair-candidate-dirs",
        nargs="*",
        default=[],
        help="Candidate pair image directories, e.g. demo/test_group_pair demo/test_self_group_pair.",
    )
    parser.add_argument(
        "--pair-candidate-csvs",
        nargs="*",
        default=[],
        help="Optional metadata CSVs for candidate pairs. Rows should contain pair_image plus distance/rank columns.",
    )
    parser.add_argument(
        "--calibrate-manual-pair-dir",
        default="",
        help="Optional manually accepted pair dir used to infer distance/margin/rank thresholds.",
    )
    parser.add_argument("--auto-pair-output-dir", default="", help="Default: <output-root>/auto_filtered_pairs.")
    parser.add_argument("--auto-filter-max-rank", type=int, default=1)
    parser.add_argument("--auto-filter-max-distance", type=float, default=0.0, help="0 = no fixed distance limit.")
    parser.add_argument("--auto-filter-min-margin", type=float, default=-1.0, help="<0 = no fixed margin limit.")
    parser.add_argument(
        "--auto-filter-same-day-night",
        action="store_true",
        help="Require source/target group to be both day or both night.",
    )
    parser.add_argument(
        "--auto-filter-manual-sources-only",
        action="store_true",
        help="When --calibrate-manual-pair-dir is set, only keep source groups present in that manual dir.",
    )
    parser.add_argument("--auto-filter-day-night-boundary", type=int, default=19)
    parser.add_argument(
        "--auto-filter-allow-datasets",
        default="drop,test",
        help="Comma-separated target datasets to keep: drop,test.",
    )
    parser.add_argument("--output-root", default="demo/manual_group_pseudo")
    parser.add_argument("--pseudo-input-dir", default="", help="Defaults to <output-root>/pseudo_inputs.")
    parser.add_argument(
        "--pseudo-label-image-dir",
        default="",
        help="Defaults to <output-root>/pseudo_labels_image. Stores source test images for pseudo labels.",
    )
    parser.add_argument("--pseudo-label-dir", default="", help="Defaults to <output-root>/pseudo_labels.")
    parser.add_argument(
        "--pseudo-label-compare-dir",
        default="",
        help="Defaults to <output-root>/pseudo_labels_image_compare. Stores left=input, right=pseudo-label previews.",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional original dataset root. When set, pseudo samples are copied into <dataset-root>/RainDrop_Train.",
    )
    parser.add_argument(
        "--rain-train-dir",
        default="",
        help="Optional RainDrop_Train directory. Overrides --dataset-root/RainDrop_Train.",
    )
    parser.add_argument("--train-copy-prefix", default="test_pseudo_")
    parser.add_argument("--focus2scene-json", default="", help="Original Drop_focus_2scene.json to extend.")
    parser.add_argument("--blur2scene-json", default="", help="Original Drop_blur_2scene.json to extend.")
    parser.add_argument("--dn-blur-4scene-json", default="", help="Original Drop_dn_blur_4scene.json to extend.")
    parser.add_argument(
        "--output-focus2scene-json",
        default="",
        help="Default: <RainDrop_Train>/Drop_focus_2scene_test_pseudo.json.",
    )
    parser.add_argument(
        "--output-blur2scene-json",
        default="",
        help="Default: <RainDrop_Train>/Drop_blur_2scene_test_pseudo.json.",
    )
    parser.add_argument(
        "--output-dn-blur-4scene-json",
        default="",
        help="Default: <RainDrop_Train>/Drop_dn_blur_4scene_test_pseudo.json.",
    )
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--drop-group-mode", choices=["visual", "numeric-gap", "none"], default="visual")
    parser.add_argument("--test-group-mode", choices=["visual", "numeric-gap", "none"], default="numeric-gap")
    parser.add_argument(
        "--drop-groups-csv",
        default="",
        help="Optional CSV with file/group ids to reuse an existing Drop grouping.",
    )
    parser.add_argument(
        "--test-groups-csv",
        default=DEFAULT_TEST_GROUPS_CSV,
        help="Optional CSV with file/group ids to reuse an existing test grouping.",
    )
    parser.add_argument("--drop-group-threshold", type=float, default=0.0)
    parser.add_argument("--test-group-threshold", type=float, default=0.0)
    parser.add_argument("--drop-min-group-size", type=int, default=2)
    parser.add_argument("--test-min-group-size", type=int, default=2)
    parser.add_argument("--numeric-gap", type=int, default=1)
    parser.add_argument("--max-unique-images", type=int, default=0, help="Images sampled for group unique selection. 0 = all.")
    parser.add_argument(
        "--no-include-target-test-groups",
        action="store_true",
        help="Do not also generate pseudo labels for right-side test groups used as label sources.",
    )
    parser.add_argument(
        "--unique-method",
        choices=["closest_to_median", "median"],
        default="closest_to_median",
        help="closest_to_median copies one real prediction; median saves a fused median image.",
    )
    parser.add_argument("--no-resize-pseudo", action="store_true", help="Do not resize pseudo labels to test input size.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {path}; use --overwrite")
        if overwrite:
            resolved = path.resolve()
            cwd = Path.cwd().resolve()
            if cwd not in resolved.parents and resolved != cwd:
                raise RuntimeError(f"Refuse to remove directory outside workspace: {resolved}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def raise_missing_paths(title: str, paths: list[Path], limit: int = 30) -> None:
    if not paths:
        return
    preview = "\n".join(f"  - {path}" for path in paths[:limit])
    more = "" if len(paths) <= limit else f"\n  ... and {len(paths) - limit} more"
    raise FileNotFoundError(f"{title} ({len(paths)}):\n{preview}{more}")


def group_key(group_id: int) -> str:
    return f"group_{group_id:03d}"


def namespace_for_group_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    groups_csv = getattr(args, f"{dataset}_groups_csv")
    if groups_csv:
        groups_csv_path = Path(groups_csv)
        if groups_csv_path.exists():
            groups_csv = str(groups_csv_path)
        elif groups_csv == DEFAULT_TEST_GROUPS_CSV:
            groups_csv = ""
        else:
            raise FileNotFoundError(f"{dataset} groups CSV not found: {groups_csv}")
    return argparse.Namespace(
        group_mode=getattr(args, f"{dataset}_group_mode"),
        group_threshold=getattr(args, f"{dataset}_group_threshold"),
        min_group_size=getattr(args, f"{dataset}_min_group_size"),
        numeric_gap=args.numeric_gap,
        test_groups_csv=groups_csv,
    )


def csv_group_ids(groups_csv: str) -> list[int]:
    if not groups_csv:
        return []
    ids = set()
    with Path(groups_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "test_group_id" not in reader.fieldnames:
            raise ValueError(f"Groups CSV must contain test_group_id: {groups_csv}")
        for row in reader:
            ids.add(int(row["test_group_id"]))
    return sorted(ids)


def make_groups(
    dataset: str,
    input_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    group_root = output_root / f"{dataset}_group"
    reset_dir(group_root, args.overwrite)

    paths = sorted(list_images(input_dir, recursive=False), key=natural_key)
    if not paths:
        raise RuntimeError(f"No images found in {dataset} input dir: {input_dir}")

    features = {
        path.name: extract_feature(path, args.feature_size)
        for path in tqdm(paths, desc=f"{dataset} input features", unit="img")
    }
    group_args = namespace_for_group_args(args, dataset)
    groups, split_threshold, adjacent_distances = build_test_groups(paths, features, group_args)
    group_ids = csv_group_ids(group_args.test_groups_csv)
    if group_ids and len(group_ids) != len(groups):
        raise RuntimeError(
            f"{dataset} groups CSV ids ({len(group_ids)}) do not match built groups ({len(groups)})"
        )

    mapping: dict[str, list[str]] = {}
    first_last: dict[str, str] = {}
    for index, group_paths in enumerate(groups, start=1):
        key = group_key(group_ids[index - 1] if group_ids else index)
        group_dir = group_root / key
        group_dir.mkdir(parents=True, exist_ok=True)
        mapping[key] = [path.name for path in group_paths]
        first_last[key] = f"{group_paths[0].stem}-{group_paths[-1].stem}"
        for path in group_paths:
            shutil.copy2(path, group_dir / path.name)

    write_json(output_root / f"{dataset}_group_image_mapping.json", mapping)
    metadata = {
        "dataset": dataset,
        "input_dir": str(input_dir),
        "group_root": str(group_root.resolve()),
        "images": len(paths),
        "groups": len(groups),
        "group_mode": group_args.group_mode,
        "group_threshold": split_threshold,
        "groups_csv": group_args.test_groups_csv,
        "feature_size": args.feature_size,
        "first_last": first_last,
    }
    if adjacent_distances:
        metadata["adjacent_distances"] = [
            {
                "prev_file": paths[i - 1].name,
                "file": paths[i].name,
                "adjacent_distance": float(distance),
                "split": bool(distance > split_threshold),
            }
            for i, distance in enumerate(adjacent_distances, start=1)
        ]
    write_json(output_root / f"{dataset}_group_metadata.json", metadata)
    return mapping, first_last


def copy_group_predictions(
    dataset: str,
    pred_dir: Path,
    mapping: dict[str, list[str]],
    output_root: Path,
    overwrite: bool,
) -> dict[str, list[str]]:
    group_root = output_root / f"{dataset}_group"
    pred_mapping: dict[str, list[str]] = {}
    missing: list[Path] = []
    for key, names in tqdm(mapping.items(), desc=f"{dataset} group predictions", unit="group"):
        pred_group_dir = group_root / f"{key}_pred"
        reset_dir(pred_group_dir, overwrite)
        pred_mapping[key] = []
        for name in names:
            src = pred_dir / name
            if not src.exists():
                missing.append(src)
                continue
            shutil.copy2(src, pred_group_dir / name)
            pred_mapping[key].append(name)
    raise_missing_paths(f"Missing {dataset} prediction files", missing)
    write_json(output_root / f"{dataset}_group_pred_mapping.json", pred_mapping)
    return pred_mapping


def load_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def save_rgb_array(path: Path, array: np.ndarray) -> None:
    image = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    image.save(path)


def build_unique_predictions(
    dataset: str,
    mapping: dict[str, list[str]],
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    group_root = output_root / f"{dataset}_group"
    unique_mapping: dict[str, dict[str, object]] = {}

    for key, names in tqdm(mapping.items(), desc=f"{dataset} unique predictions", unit="group"):
        pred_group_dir = group_root / f"{key}_pred"
        unique_dir = group_root / f"{key}_pred_unique"
        reset_dir(unique_dir, args.overwrite)

        pred_paths = [pred_group_dir / name for name in names]
        sampled = sample_evenly(pred_paths, args.max_unique_images)
        arrays: list[np.ndarray] = []
        base_size: tuple[int, int] | None = None
        for path in sampled:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if base_size is None:
                    base_size = image.size
                elif image.size != base_size:
                    image = image.resize(base_size, Image.Resampling.BICUBIC)
                arrays.append(np.asarray(image, dtype=np.float32))

        if not arrays:
            raise RuntimeError(f"No predictions found for {dataset} {key}")
        median = np.median(np.stack(arrays, axis=0), axis=0)

        if args.unique_method == "median":
            unique_name = f"{key}_median.png"
            unique_path = unique_dir / unique_name
            save_rgb_array(unique_path, median)
            source_file = ""
            score = 0.0
        else:
            scores = [float(np.mean(np.abs(array - median))) for array in arrays]
            best_index = int(np.argmin(scores))
            source_path = sampled[best_index]
            unique_name = source_path.name
            unique_path = unique_dir / unique_name
            shutil.copy2(source_path, unique_path)
            source_file = source_path.name
            score = scores[best_index]

        unique_mapping[key] = {
            "unique_file": unique_name,
            "unique_path": str(unique_path.resolve()),
            "source_file": source_file,
            "score_to_group_median": score,
            "group_pred_dir": str(pred_group_dir.resolve()),
        }

    write_json(output_root / f"{dataset}_group_pred_unique_mapping.json", unique_mapping)
    return unique_mapping


SOURCE_RE = re.compile(r"^group_(?P<source>\d+)_test_[^_]+__")
DROP_TARGET_RE = re.compile(r"__drop_.*?(?:group_?)?(?P<target>\d+)_")
TEST_TARGET_RE = re.compile(r"__match_(?P<target>\d+)_")
RANK_RE = re.compile(r"_rank(?P<rank>\d+)")


def safe_float(value: object, default: float = 0.0) -> float:
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(value: object, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def pair_key(source_group: str, target_dataset: str, target_group: str) -> tuple[str, str, str]:
    return source_group, target_dataset, target_group


def parse_manual_pair_name(path: Path) -> tuple[str, str, str]:
    source_match = SOURCE_RE.search(path.name)
    if not source_match:
        raise ValueError(f"Cannot parse source test group from manual pair name: {path.name}")
    source_group = f"group_{int(source_match.group('source')):03d}"

    drop_match = DROP_TARGET_RE.search(path.name)
    if drop_match:
        return source_group, "drop", f"group_{int(drop_match.group('target')):03d}"

    test_match = TEST_TARGET_RE.search(path.name)
    if test_match:
        return source_group, "test", f"group_{int(test_match.group('target')):03d}"

    raise ValueError(f"Cannot parse target group from manual pair name: {path.name}")


def infer_rank_from_pair_name(name: str) -> int:
    match = RANK_RE.search(name)
    return int(match.group("rank")) if match else 1


def infer_target_is_day(pair_name: str, target_dataset: str, target_group: str, boundary: int) -> bool | None:
    lower = pair_name.lower()
    if target_dataset == "drop":
        if "__drop_night" in lower or "__night" in lower:
            return False
        if "__drop_day" in lower or "__day" in lower:
            return True
        return None
    return manual_group_id(target_group) <= boundary


def read_candidate_metadata(csv_paths: list[str]) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for raw_path in csv_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Pair candidate CSV not found: {path}")
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pair_image = str(row.get("pair_image", "")).strip()
                if not pair_image:
                    continue
                distance = safe_float(
                    row.get("distance", row.get("best_distance", row.get("match_distance", ""))),
                    default=0.0,
                )
                margin = safe_float(
                    row.get("margin_to_second", row.get("margin", "")),
                    default=0.0,
                )
                rank = safe_int(row.get("match_rank", ""), default=infer_rank_from_pair_name(pair_image))
                metadata[pair_image] = {
                    "rank": rank,
                    "distance": distance,
                    "margin": margin,
                    "metadata_csv": str(path),
                }
    return metadata


def load_manual_positive_keys(manual_pair_dir: Path) -> set[tuple[str, str, str]]:
    if not manual_pair_dir.is_dir():
        raise FileNotFoundError(f"Calibration manual pair dir not found: {manual_pair_dir}")
    positives: set[tuple[str, str, str]] = set()
    for path in manual_pair_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            positives.add(pair_key(*parse_manual_pair_name(path)))
    if not positives:
        raise RuntimeError(f"No calibration pairs found in: {manual_pair_dir}")
    return positives


def collect_pair_candidates(
    candidate_dirs: list[str],
    candidate_csvs: list[str],
) -> list[dict[str, object]]:
    metadata = read_candidate_metadata(candidate_csvs)
    candidates: list[dict[str, object]] = []
    seen: set[Path] = set()
    for raw_dir in candidate_dirs:
        directory = Path(raw_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Pair candidate directory not found: {directory}")
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            source_group, target_dataset, target_group = parse_manual_pair_name(path)
            row = dict(metadata.get(path.name, {}))
            candidates.append({
                "pair_image": path.name,
                "pair_path": path,
                "source_group": source_group,
                "target_dataset": target_dataset,
                "target_group": target_group,
                "rank": int(row.get("rank", infer_rank_from_pair_name(path.name))),
                "distance": float(row.get("distance", 0.0)),
                "margin": float(row.get("margin", 0.0)),
                "metadata_csv": str(row.get("metadata_csv", "")),
            })
    if not candidates:
        raise RuntimeError(f"No pair candidate images found in: {candidate_dirs}")
    return candidates


def calibrated_thresholds(
    candidates: list[dict[str, object]],
    positives: set[tuple[str, str, str]],
    fixed_max_rank: int,
    fixed_max_distance: float,
    fixed_min_margin: float,
) -> dict[str, float | int]:
    positive_candidates = [
        candidate for candidate in candidates
        if pair_key(
            str(candidate["source_group"]),
            str(candidate["target_dataset"]),
            str(candidate["target_group"]),
        )
        in positives
    ]
    if not positive_candidates:
        return {
            "max_rank": fixed_max_rank,
            "max_distance": fixed_max_distance,
            "min_margin": fixed_min_margin,
            "positive_candidates": 0,
        }
    max_rank = max(int(candidate["rank"]) for candidate in positive_candidates)
    max_distance = max(float(candidate["distance"]) for candidate in positive_candidates)
    min_margin = min(float(candidate["margin"]) for candidate in positive_candidates)
    return {
        "max_rank": max(max_rank, fixed_max_rank),
        "max_distance": max(max_distance, fixed_max_distance) if fixed_max_distance > 0 else max_distance,
        "min_margin": min(min_margin, fixed_min_margin) if fixed_min_margin >= 0 else min_margin,
        "positive_candidates": len(positive_candidates),
    }


def filter_pair_candidates(args: argparse.Namespace, output_root: Path) -> Path:
    if not args.pair_candidate_dirs:
        raise ValueError("--auto-filter-pairs requires --pair-candidate-dirs")

    output_dir = Path(args.auto_pair_output_dir) if args.auto_pair_output_dir else output_root / "auto_filtered_pairs"
    reset_dir(output_dir, args.overwrite)

    candidates = collect_pair_candidates(args.pair_candidate_dirs, args.pair_candidate_csvs)
    allowed_datasets = {
        item.strip()
        for item in args.auto_filter_allow_datasets.split(",")
        if item.strip()
    }
    positives: set[tuple[str, str, str]] = set()
    if args.calibrate_manual_pair_dir:
        positives = load_manual_positive_keys(Path(args.calibrate_manual_pair_dir))
    positive_source_groups = {key[0] for key in positives}

    thresholds = calibrated_thresholds(
        candidates=candidates,
        positives=positives,
        fixed_max_rank=args.auto_filter_max_rank,
        fixed_max_distance=args.auto_filter_max_distance,
        fixed_min_margin=args.auto_filter_min_margin,
    )
    max_rank = int(thresholds["max_rank"])
    max_distance = float(thresholds["max_distance"])
    min_margin = float(thresholds["min_margin"])

    accepted_by_source: dict[str, dict[str, object]] = {}
    audit_rows: list[dict[str, object]] = []
    for candidate in candidates:
        source_group = str(candidate["source_group"])
        target_dataset = str(candidate["target_dataset"])
        target_group = str(candidate["target_group"])
        rank = int(candidate["rank"])
        distance = float(candidate["distance"])
        margin = float(candidate["margin"])
        reasons: list[str] = []

        if target_dataset not in allowed_datasets:
            reasons.append("target_dataset")
        if args.auto_filter_manual_sources_only and source_group not in positive_source_groups:
            reasons.append("manual_source")
        if rank > max_rank:
            reasons.append("rank")
        if max_distance > 0 and distance > max_distance + 1e-12:
            reasons.append("distance")
        if min_margin >= 0 and margin < min_margin - 1e-12:
            reasons.append("margin")
        if args.auto_filter_same_day_night:
            source_is_day = manual_group_id(source_group) <= args.auto_filter_day_night_boundary
            target_is_day = infer_target_is_day(
                str(candidate["pair_image"]),
                target_dataset,
                target_group,
                args.auto_filter_day_night_boundary,
            )
            if target_is_day is not None and source_is_day != target_is_day:
                reasons.append("day_night")

        accepted = not reasons
        if accepted:
            old = accepted_by_source.get(source_group)
            if old is None:
                accepted_by_source[source_group] = candidate
            else:
                old_key = (
                    int(old["rank"]),
                    float(old["distance"]),
                    -float(old["margin"]),
                    str(old["target_dataset"]),
                    str(old["target_group"]),
                )
                new_key = (
                    rank,
                    distance,
                    -margin,
                    target_dataset,
                    target_group,
                )
                if new_key < old_key:
                    accepted_by_source[source_group] = candidate

        audit_rows.append({
            "pair_image": candidate["pair_image"],
            "source_group": source_group,
            "target_dataset": target_dataset,
            "target_group": target_group,
            "rank": rank,
            "distance": distance,
            "margin": margin,
            "accepted_before_dedup": int(accepted),
            "reject_reasons": ",".join(reasons),
            "manual_positive": int(pair_key(source_group, target_dataset, target_group) in positives),
            "metadata_csv": candidate["metadata_csv"],
        })

    accepted_keys = {
        str(candidate["pair_image"])
        for candidate in accepted_by_source.values()
    }
    for row in audit_rows:
        row["accepted"] = int(str(row["pair_image"]) in accepted_keys)

    for candidate in sorted(
        accepted_by_source.values(),
        key=lambda item: manual_group_id(str(item["source_group"])),
    ):
        shutil.copy2(Path(candidate["pair_path"]), output_dir / str(candidate["pair_image"]))

    if not accepted_by_source:
        raise RuntimeError(
            "Auto pair filtering produced no pairs. "
            "Relax --auto-filter-max-rank/--auto-filter-max-distance/--auto-filter-min-margin."
        )

    with (output_dir / "filter_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    report = {
        "candidate_dirs": args.pair_candidate_dirs,
        "candidate_csvs": args.pair_candidate_csvs,
        "output_dir": str(output_dir.resolve()),
        "candidates": len(candidates),
        "accepted_pairs": len(accepted_by_source),
        "allowed_datasets": sorted(allowed_datasets),
        "same_day_night": bool(args.auto_filter_same_day_night),
        "manual_sources_only": bool(args.auto_filter_manual_sources_only),
        "day_night_boundary": args.auto_filter_day_night_boundary,
        "thresholds": thresholds,
        "calibrate_manual_pair_dir": args.calibrate_manual_pair_dir,
    }
    write_json(output_dir / "filter_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return output_dir


def build_manual_mapping(
    manual_pair_dir: Path,
    test_group_mapping: dict[str, list[str]],
    unique_mappings: dict[str, dict[str, dict[str, object]]],
    output_root: Path,
    include_target_test_groups: bool,
) -> dict[str, dict[str, object]]:
    manual_files = sorted(
        path for path in manual_pair_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not manual_files:
        raise RuntimeError(f"No manual pair images found in: {manual_pair_dir}")

    image_mapping: dict[str, dict[str, object]] = {}
    pair_rows: list[dict[str, str]] = []
    skipped_target_rows: list[dict[str, str]] = []
    for manual_file in manual_files:
        source_group, target_dataset, target_group = parse_manual_pair_name(manual_file)
        if source_group not in test_group_mapping:
            raise KeyError(f"Manual source group not found in test mapping: {source_group}")
        if target_group not in unique_mappings[target_dataset]:
            raise KeyError(f"Manual target group not found in {target_dataset} unique mapping: {target_group}")

        unique_info = unique_mappings[target_dataset][target_group]
        pair_rows.append({
            "manual_pair_file": manual_file.name,
            "source_group": source_group,
            "target_dataset": target_dataset,
            "target_group": target_group,
            "target_unique_file": str(unique_info["unique_file"]),
        })

        for test_name in test_group_mapping[source_group]:
            if test_name in image_mapping:
                old = image_mapping[test_name]
                raise RuntimeError(
                    f"Duplicate pseudo mapping for {test_name}: "
                    f"{old['manual_pair_file']} and {manual_file.name}"
                )
            image_mapping[test_name] = {
                "source_group": source_group,
                "target_dataset": target_dataset,
                "target_group": target_group,
                "target_unique_file": unique_info["unique_file"],
                "target_unique_path": unique_info["unique_path"],
                "target_unique_source_file": unique_info["source_file"],
                "manual_pair_file": manual_file.name,
            }

        if include_target_test_groups and target_dataset == "test":
            if target_group not in test_group_mapping:
                raise KeyError(f"Manual target test group not found in test mapping: {target_group}")
            for test_name in test_group_mapping[target_group]:
                if test_name in image_mapping:
                    skipped_target_rows.append({
                        "test_file": test_name,
                        "target_group": target_group,
                        "manual_pair_file": manual_file.name,
                        "existing_source_group": str(image_mapping[test_name]["source_group"]),
                        "existing_manual_pair_file": str(image_mapping[test_name]["manual_pair_file"]),
                    })
                    continue
                image_mapping[test_name] = {
                    "source_group": target_group,
                    "target_dataset": "test",
                    "target_group": target_group,
                    "target_unique_file": unique_info["unique_file"],
                    "target_unique_path": unique_info["unique_path"],
                    "target_unique_source_file": unique_info["source_file"],
                    "manual_pair_file": manual_file.name,
                    "mapping_role": "target_test_group_self",
                }

    write_json(output_root / "manual_pair_group_mapping.json", pair_rows)
    write_json(output_root / "skipped_target_test_group_mapping.json", skipped_target_rows)
    write_json(output_root / "image_group_pred_unique_mapping.json", image_mapping)
    return image_mapping


def copy_or_resize_pseudo_labels(
    mapping: dict[str, dict[str, object]],
    test_input_dir: Path,
    pseudo_label_dir: Path,
    resize_to_test: bool,
    overwrite: bool,
) -> None:
    reset_dir(pseudo_label_dir, overwrite)
    for test_name, info in tqdm(mapping.items(), desc="Pseudo labels", unit="img"):
        src = Path(str(info["target_unique_path"]))
        dst = pseudo_label_dir / test_name
        if not resize_to_test:
            shutil.copy2(src, dst)
            continue

        test_path = test_input_dir / test_name
        with Image.open(src) as pred_image:
            pred_image = pred_image.convert("RGB")
            with Image.open(test_path) as test_image:
                target_size = test_image.size
            if pred_image.size != target_size:
                pred_image = pred_image.resize(target_size, Image.Resampling.BICUBIC)
            pred_image.save(dst)


def copy_pseudo_inputs(
    mapping: dict[str, dict[str, object]],
    test_input_dir: Path,
    pseudo_input_dir: Path,
    overwrite: bool,
) -> None:
    reset_dir(pseudo_input_dir, overwrite)
    missing: list[Path] = []
    for test_name in tqdm(sorted(mapping), desc="Pseudo inputs", unit="img"):
        src = test_input_dir / test_name
        if not src.exists():
            missing.append(src)
            continue
        shutil.copy2(src, pseudo_input_dir / test_name)
    raise_missing_paths("Missing test input files for pseudo_inputs", missing)


def build_pseudo_label_compares(
    mapping: dict[str, dict[str, object]],
    image_dir: Path,
    pseudo_label_dir: Path,
    compare_dir: Path,
    overwrite: bool,
) -> None:
    reset_dir(compare_dir, overwrite)
    missing: list[Path] = []
    for test_name in tqdm(sorted(mapping), desc="Pseudo input/label compares", unit="img"):
        image_path = image_dir / test_name
        label_path = pseudo_label_dir / test_name
        if not image_path.exists():
            missing.append(image_path)
        if not label_path.exists():
            missing.append(label_path)
        if not image_path.exists() or not label_path.exists():
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            with Image.open(label_path) as label:
                label = label.convert("RGB")
                if label.size != image.size:
                    label = label.resize(image.size, Image.Resampling.BICUBIC)
                canvas = Image.new("RGB", (image.width + label.width, image.height))
                canvas.paste(image, (0, 0))
                canvas.paste(label, (image.width, 0))
                canvas.save(compare_dir / test_name)
    raise_missing_paths("Missing files needed for pseudo label compare images", missing)


def resolve_rain_train_dir(args: argparse.Namespace) -> Path | None:
    if args.rain_train_dir:
        return Path(args.rain_train_dir)
    if args.dataset_root:
        return Path(args.dataset_root) / "RainDrop_Train"
    return None


def precheck_args(args: argparse.Namespace) -> None:
    missing: list[Path] = []
    for raw in (
        args.drop_input_dir,
        args.test_input_dir,
        args.drop_pred_dir,
        args.test_pred_dir,
    ):
        path = Path(raw)
        if not path.is_dir():
            missing.append(path)
    if args.auto_filter_pairs:
        for raw in args.pair_candidate_dirs:
            path = Path(raw)
            if not path.is_dir():
                missing.append(path)
        for raw in args.pair_candidate_csvs:
            path = Path(raw)
            if not path.is_file():
                missing.append(path)
        if args.calibrate_manual_pair_dir:
            path = Path(args.calibrate_manual_pair_dir)
            if not path.is_dir():
                missing.append(path)
    else:
        manual_pair_dir = Path(args.manual_pair_dir)
        if not manual_pair_dir.is_dir():
            missing.append(manual_pair_dir)

    drop_groups_csv = Path(args.drop_groups_csv) if args.drop_groups_csv else None
    if drop_groups_csv is not None and not drop_groups_csv.is_file():
        missing.append(drop_groups_csv)
    test_groups_csv = Path(args.test_groups_csv) if args.test_groups_csv else None
    if (
        test_groups_csv is not None
        and args.test_groups_csv != DEFAULT_TEST_GROUPS_CSV
        and not test_groups_csv.is_file()
    ):
        missing.append(test_groups_csv)

    rain_train_dir = resolve_rain_train_dir(args)
    if rain_train_dir is not None:
        for path in (rain_train_dir, rain_train_dir / "Drop", rain_train_dir / "Clear"):
            if not path.is_dir():
                missing.append(path)
        focus_json = Path(args.focus2scene_json) if args.focus2scene_json else rain_train_dir / "Drop_focus_2scene.json"
        blur_json = Path(args.blur2scene_json) if args.blur2scene_json else rain_train_dir / "Drop_blur_2scene.json"
        dn_blur_json = Path(args.dn_blur_4scene_json) if args.dn_blur_4scene_json else rain_train_dir / "Drop_dn_blur_4scene.json"
        for path in (focus_json, blur_json, dn_blur_json):
            if not path.is_file():
                missing.append(path)

    raise_missing_paths("Missing required input paths", missing)


def copied_pseudo_name(original_name: str, prefix: str) -> str:
    return f"{prefix}{Path(original_name).name}"


def copy_pseudo_to_train(
    image_mapping: dict[str, dict[str, object]],
    test_input_dir: Path,
    pseudo_label_dir: Path,
    rain_train_dir: Path,
    prefix: str,
    overwrite: bool,
) -> dict[str, dict[str, str]]:
    drop_dir = rain_train_dir / "Drop"
    clear_dir = rain_train_dir / "Clear"
    drop_dir.mkdir(parents=True, exist_ok=True)
    clear_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, dict[str, str]] = {}
    missing: list[Path] = []
    for test_name in tqdm(sorted(image_mapping), desc="Copy pseudo to RainDrop_Train", unit="img"):
        new_name = copied_pseudo_name(test_name, prefix)
        src_drop = test_input_dir / test_name
        src_clear = pseudo_label_dir / test_name
        dst_drop = drop_dir / new_name
        dst_clear = clear_dir / new_name
        if not src_drop.exists():
            missing.append(src_drop)
        if not src_clear.exists():
            missing.append(src_clear)
        if missing and (not src_drop.exists() or not src_clear.exists()):
            continue
        if not overwrite and (dst_drop.exists() or dst_clear.exists()):
            raise FileExistsError(f"Pseudo train sample already exists: {new_name}; use --overwrite")
        shutil.copy2(src_drop, dst_drop)
        shutil.copy2(src_clear, dst_clear)
        copied[test_name] = {
            "train_name": new_name,
            "drop_path": str(dst_drop.resolve()),
            "clear_path": str(dst_clear.resolve()),
        }
    raise_missing_paths("Missing files needed to copy pseudo samples into RainDrop_Train", missing)
    return copied


def load_json_dict(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): int(value) for key, value in payload.items()}


def manual_group_id(group_name: str) -> int:
    match = re.search(r"(\d+)$", group_name)
    if not match:
        raise ValueError(f"Cannot parse group id from: {group_name}")
    return int(match.group(1))


def pseudo_scene_flags(info: dict[str, object]) -> tuple[int, int, int]:
    """Return (focus2_label, dn_blur4_label, group_id) for one copied pseudo sample.

    focus2: 0=background focus, 1=raindrop focus.
    dn_blur4: 0=night_not_blur, 1=night_blur, 2=day_not_blur, 3=day_blur.
    Manual rule: left-side test groups are raindrop-focus/background-blur;
    right-side test groups are background-focus/raindrop-blur.
    Groups <=19 are day, groups >=20 are night.
    """
    is_target_test_self = info.get("mapping_role") == "target_test_group_self"
    group_name = str(info["target_group"] if is_target_test_self else info["source_group"])
    gid = manual_group_id(group_name)
    is_day = gid <= 19
    is_blur = not is_target_test_self
    focus2 = 0 if is_target_test_self else 1
    dn_blur4 = (2 if is_day else 0) + int(is_blur)
    return focus2, dn_blur4, gid


def update_scene_jsons(
    image_mapping: dict[str, dict[str, object]],
    copied_mapping: dict[str, dict[str, str]],
    focus_json: Path,
    blur_json: Path,
    dn_blur_json: Path,
    output_focus_json: Path,
    output_blur_json: Path,
    output_dn_blur_json: Path,
) -> dict[str, object]:
    focus_labels = load_json_dict(focus_json)
    blur_labels = load_json_dict(blur_json)
    dn_blur_labels = load_json_dict(dn_blur_json)
    added: dict[str, dict[str, int]] = {}

    for test_name, info in image_mapping.items():
        train_name = copied_mapping[test_name]["train_name"]
        focus2, dn_blur4, gid = pseudo_scene_flags(info)
        blur2 = dn_blur4 % 2
        focus_labels[train_name] = focus2
        blur_labels[train_name] = blur2
        dn_blur_labels[train_name] = dn_blur4
        added[train_name] = {
            "source_test_file": test_name,
            "group_id": gid,
            "focus2": focus2,
            "blur2": blur2,
            "dn_blur4": dn_blur4,
        }

    write_json(output_focus_json, focus_labels)
    write_json(output_blur_json, blur_labels)
    write_json(output_dn_blur_json, dn_blur_labels)
    return {
        "focus2_json": str(output_focus_json.resolve()),
        "blur2_json": str(output_blur_json.resolve()),
        "dn_blur4_json": str(output_dn_blur_json.resolve()),
        "added": added,
        "added_count": len(added),
    }


def main() -> None:
    args = parse_args()
    precheck_args(args)
    output_root = Path(args.output_root)
    reset_dir(output_root, args.overwrite)

    drop_mapping, _ = make_groups("drop", Path(args.drop_input_dir), output_root, args)
    test_mapping, _ = make_groups("test", Path(args.test_input_dir), output_root, args)

    copy_group_predictions("drop", Path(args.drop_pred_dir), drop_mapping, output_root, args.overwrite)
    copy_group_predictions("test", Path(args.test_pred_dir), test_mapping, output_root, args.overwrite)

    drop_unique = build_unique_predictions("drop", drop_mapping, output_root, args)
    test_unique = build_unique_predictions("test", test_mapping, output_root, args)

    pair_dir = filter_pair_candidates(args, output_root) if args.auto_filter_pairs else Path(args.manual_pair_dir)

    image_mapping = build_manual_mapping(
        pair_dir,
        test_mapping,
        {"drop": drop_unique, "test": test_unique},
        output_root,
        include_target_test_groups=not args.no_include_target_test_groups,
    )

    pseudo_input_dir = Path(args.pseudo_input_dir) if args.pseudo_input_dir else output_root / "pseudo_inputs"
    copy_pseudo_inputs(
        image_mapping,
        Path(args.test_input_dir),
        pseudo_input_dir,
        overwrite=args.overwrite,
    )

    pseudo_label_image_dir = (
        Path(args.pseudo_label_image_dir)
        if args.pseudo_label_image_dir
        else output_root / "pseudo_labels_image"
    )
    copy_pseudo_inputs(
        image_mapping,
        Path(args.test_input_dir),
        pseudo_label_image_dir,
        overwrite=args.overwrite,
    )

    pseudo_label_dir = Path(args.pseudo_label_dir) if args.pseudo_label_dir else output_root / "pseudo_labels"
    copy_or_resize_pseudo_labels(
        image_mapping,
        Path(args.test_input_dir),
        pseudo_label_dir,
        resize_to_test=not args.no_resize_pseudo,
        overwrite=args.overwrite,
    )

    pseudo_label_compare_dir = (
        Path(args.pseudo_label_compare_dir)
        if args.pseudo_label_compare_dir
        else output_root / "pseudo_labels_image_compare"
    )
    build_pseudo_label_compares(
        image_mapping,
        pseudo_label_image_dir,
        pseudo_label_dir,
        pseudo_label_compare_dir,
        overwrite=args.overwrite,
    )

    rain_train_dir = resolve_rain_train_dir(args)
    copied_train_mapping: dict[str, dict[str, str]] = {}
    scene_update: dict[str, object] = {}
    if rain_train_dir is not None:
        copied_train_mapping = copy_pseudo_to_train(
            image_mapping=image_mapping,
            test_input_dir=Path(args.test_input_dir),
            pseudo_label_dir=pseudo_label_dir,
            rain_train_dir=rain_train_dir,
            prefix=args.train_copy_prefix,
            overwrite=args.overwrite,
        )
        write_json(output_root / "test_pseudo_train_copy_mapping.json", copied_train_mapping)

        focus_json = Path(args.focus2scene_json) if args.focus2scene_json else rain_train_dir / "Drop_focus_2scene.json"
        blur_json = Path(args.blur2scene_json) if args.blur2scene_json else rain_train_dir / "Drop_blur_2scene.json"
        dn_blur_json = Path(args.dn_blur_4scene_json) if args.dn_blur_4scene_json else rain_train_dir / "Drop_dn_blur_4scene.json"
        output_focus_json = (
            Path(args.output_focus2scene_json)
            if args.output_focus2scene_json
            else rain_train_dir / "Drop_focus_2scene_test_pseudo.json"
        )
        output_blur_json = (
            Path(args.output_blur2scene_json)
            if args.output_blur2scene_json
            else rain_train_dir / "Drop_blur_2scene_test_pseudo.json"
        )
        output_dn_blur_json = (
            Path(args.output_dn_blur_4scene_json)
            if args.output_dn_blur_4scene_json
            else rain_train_dir / "Drop_dn_blur_4scene_test_pseudo.json"
        )
        missing_scene_jsons = [path for path in (focus_json, blur_json, dn_blur_json) if not path.exists()]
        raise_missing_paths("Missing scene JSON files", missing_scene_jsons)
        scene_update = update_scene_jsons(
            image_mapping=image_mapping,
            copied_mapping=copied_train_mapping,
            focus_json=focus_json,
            blur_json=blur_json,
            dn_blur_json=dn_blur_json,
            output_focus_json=output_focus_json,
            output_blur_json=output_blur_json,
            output_dn_blur_json=output_dn_blur_json,
        )
        write_json(output_root / "test_pseudo_scene_update_manifest.json", scene_update)

    manifest = {
        "drop_input_dir": args.drop_input_dir,
        "test_input_dir": args.test_input_dir,
        "drop_pred_dir": args.drop_pred_dir,
        "test_pred_dir": args.test_pred_dir,
        "manual_pair_dir": args.manual_pair_dir,
        "effective_pair_dir": str(pair_dir.resolve()),
        "auto_filter_pairs": bool(args.auto_filter_pairs),
        "output_root": str(output_root.resolve()),
        "pseudo_input_dir": str(pseudo_input_dir.resolve()),
        "pseudo_label_image_dir": str(pseudo_label_image_dir.resolve()),
        "pseudo_label_dir": str(pseudo_label_dir.resolve()),
        "pseudo_label_compare_dir": str(pseudo_label_compare_dir.resolve()),
        "drop_groups": len(drop_mapping),
        "test_groups": len(test_mapping),
        "mapped_test_images": len(image_mapping),
        "unique_method": args.unique_method,
        "include_target_test_groups": not args.no_include_target_test_groups,
        "rain_train_dir": str(rain_train_dir.resolve()) if rain_train_dir is not None else "",
        "copied_train_images": len(copied_train_mapping),
        "scene_update": scene_update,
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
