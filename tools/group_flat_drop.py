#!/usr/bin/env python3
"""Group a flat Drop image directory into scene folders."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from pair_test_input_to_clear import (
    build_test_groups,
    extract_feature,
    list_images,
    natural_key,
    representative_path,
    write_csv,
)


DEFAULT_DROP_DIR = r"\\158.132.186.40\isds\huilin\tp\eccv_dn\Drop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-dir", default=DEFAULT_DROP_DIR, help="Flat Drop directory to group.")
    parser.add_argument("--output-dir", default="demo/drop_group", help="Grouped image output directory.")
    parser.add_argument("--meta-dir", default="demo/drop_group_meta", help="CSV/JSON metadata output directory.")
    parser.add_argument("--feature-size", type=int, default=48, help="Low-resolution feature size.")
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.0,
        help="Adjacent split threshold. 0 = estimate automatically.",
    )
    parser.add_argument("--min-group-size", type=int, default=2, help="Merge groups smaller than this into neighbors.")
    parser.add_argument(
        "--group-mode",
        choices=["visual", "numeric-gap", "none"],
        default="visual",
        help="How to group flat Drop images.",
    )
    parser.add_argument(
        "--numeric-gap",
        type=int,
        default=1,
        help="For --group-mode numeric-gap: start a new group when numeric filename gap is larger than this.",
    )
    parser.set_defaults(test_groups_csv="")
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


def group_folder_name(group_id: int, paths: list[Path]) -> str:
    return f"group_{group_id:03d}_{paths[0].stem}-{paths[-1].stem}"


def main() -> None:
    args = parse_args()
    drop_dir = Path(args.drop_dir)
    output_dir = Path(args.output_dir)
    meta_dir = Path(args.meta_dir)

    reset_dir(output_dir, args.overwrite)
    reset_dir(meta_dir, args.overwrite)

    paths = sorted(list_images(drop_dir, recursive=False), key=natural_key)
    if not paths:
        raise RuntimeError(f"No images found in: {drop_dir}")

    features = {
        path.name: extract_feature(path, args.feature_size)
        for path in tqdm(paths, desc="Drop features", unit="img")
    }
    groups, split_threshold, adjacent_distances = build_test_groups(paths, features, args)

    group_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    for group_id, group_paths in enumerate(groups, start=1):
        folder_name = group_folder_name(group_id, group_paths)
        group_dir = output_dir / folder_name
        group_dir.mkdir(parents=True, exist_ok=True)
        rep = representative_path(group_paths, features)

        for path in group_paths:
            shutil.copy2(path, group_dir / path.name)
            image_rows.append({
                "file": path.name,
                "group_id": group_id,
                "group_folder": folder_name,
                "is_group_rep": int(path.name == rep.name),
            })

        group_rows.append({
            "group_id": group_id,
            "group_folder": folder_name,
            "count": len(group_paths),
            "start_file": group_paths[0].name,
            "end_file": group_paths[-1].name,
            "rep_file": rep.name,
        })

    write_csv(meta_dir / "groups.csv", group_rows)
    write_csv(meta_dir / "images.csv", image_rows)

    if adjacent_distances:
        distance_rows = [
            {
                "prev_file": paths[i - 1].name,
                "file": paths[i].name,
                "adjacent_distance": distance,
                "split": int(distance > split_threshold),
            }
            for i, distance in enumerate(adjacent_distances, start=1)
        ]
        write_csv(meta_dir / "adjacent_distances.csv", distance_rows)

    manifest = {
        "drop_dir": str(drop_dir),
        "output_dir": str(output_dir.resolve()),
        "meta_dir": str(meta_dir.resolve()),
        "images": len(paths),
        "groups": len(groups),
        "group_mode": args.group_mode,
        "group_threshold": split_threshold,
        "feature_size": args.feature_size,
        "min_group_size": args.min_group_size,
    }
    (meta_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
