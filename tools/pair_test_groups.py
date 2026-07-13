#!/usr/bin/env python3
"""Pair each grouped test scene with its nearest other test group."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pair_test_input_to_clear import (
    extract_feature,
    feature_distance,
    list_images,
    median_feature,
    natural_key,
    representative_path,
    sample_evenly,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", default="demo/test_group", help="Grouped test image directory.")
    parser.add_argument("--output-dir", default="demo/test_self_group_pair", help="Pair visualization directory.")
    parser.add_argument("--meta-dir", default="demo/test_self_group_pairing", help="CSV/JSON metadata directory.")
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--max-images", type=int, default=0, help="Images sampled per group. 0 = all.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of nearest groups exported per source group.")
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


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def concat_images(left_path: Path, right_path: Path) -> Image.Image:
    left = load_rgb(left_path)
    right = load_rgb(right_path)
    if right.size != left.size:
        right = right.resize(left.size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (left.width + right.width, left.height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def group_id_from_name(index: int, name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return f"{index:03d}"


def main() -> None:
    args = parse_args()
    group_dir = Path(args.group_dir)
    output_dir = Path(args.output_dir)
    meta_dir = Path(args.meta_dir)

    reset_dir(output_dir, args.overwrite)
    reset_dir(meta_dir, args.overwrite)

    group_dirs = sorted([path for path in group_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
    if len(group_dirs) < 2:
        raise RuntimeError(f"Need at least two groups in: {group_dir}")

    groups: list[dict[str, object]] = []
    for index, directory in enumerate(tqdm(group_dirs, desc="Group features", unit="group"), start=1):
        paths = sorted(list_images(directory, recursive=False), key=natural_key)
        if not paths:
            continue
        sampled = sample_evenly(paths, args.max_images)
        features = {path.name: extract_feature(path, args.feature_size) for path in sampled}
        groups.append({
            "group_id": group_id_from_name(index, directory.name),
            "group_folder": directory.name,
            "paths": paths,
            "sampled": sampled,
            "feature": median_feature(list(features.values())),
            "rep": representative_path(sampled, features),
        })

    if len(groups) < 2:
        raise RuntimeError(f"Need at least two non-empty groups in: {group_dir}")

    match_rows: list[dict[str, object]] = []
    for source in groups:
        candidates = sorted(
            (
                (target, feature_distance(source["feature"], target["feature"]))
                for target in groups
                if target["group_folder"] != source["group_folder"]
            ),
            key=lambda item: item[1],
        )
        for rank, (target, distance) in enumerate(candidates[: max(args.top_k, 1)], start=1):
            second_distance = candidates[1][1] if len(candidates) > 1 else distance
            pair_name = (
                f"group_{source['group_id']}_test_{Path(source['rep']).stem}"
                f"__match_{target['group_id']}_{Path(target['rep']).stem}_rank{rank}.png"
            )
            concat_images(Path(source["rep"]), Path(target["rep"])).save(output_dir / pair_name)
            match_rows.append({
                "source_group_id": source["group_id"],
                "source_group_folder": source["group_folder"],
                "source_count": len(source["paths"]),
                "source_rep": Path(source["rep"]).name,
                "match_rank": rank,
                "match_group_id": target["group_id"],
                "match_group_folder": target["group_folder"],
                "match_count": len(target["paths"]),
                "match_rep": Path(target["rep"]).name,
                "distance": distance,
                "margin_to_second": second_distance - distance if rank == 1 else "",
                "pair_image": pair_name,
            })

    write_csv(meta_dir / "group_matches.csv", match_rows)
    manifest = {
        "group_dir": str(group_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "meta_dir": str(meta_dir.resolve()),
        "groups": len(groups),
        "top_k": args.top_k,
        "feature_size": args.feature_size,
        "max_images": args.max_images,
        "matches": len(match_rows),
    }
    (meta_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
