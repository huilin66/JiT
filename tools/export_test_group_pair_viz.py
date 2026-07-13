#!/usr/bin/env python3
"""Export grouped test images and representative test/drop pair visualizations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_TEST_DIR = r"\\158.132.186.40\isds\huilin\tp\eccv_dn\test-input"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-map", default="demo/test_drop_pairing/pair_map.csv")
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--test-group-dir", default="demo/test_group")
    parser.add_argument("--test-group-pair-dir", default="demo/test_group_pair")
    parser.add_argument(
        "--right-image",
        choices=["drop", "clear"],
        default="drop",
        help="Right side of pair visualization. Use drop for checking matching; clear for GT preview.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def group_folder_name(group_id: int, rows: list[dict[str, str]]) -> str:
    names = [row["test_file"] for row in rows]
    start = Path(names[0]).stem
    end = Path(names[-1]).stem
    return f"group_{group_id:03d}_{start}-{end}"


def main() -> None:
    args = parse_args()
    pair_map = Path(args.pair_map)
    test_dir = Path(args.test_dir)
    test_group_dir = Path(args.test_group_dir)
    test_group_pair_dir = Path(args.test_group_pair_dir)

    rows = read_rows(pair_map)
    if not rows:
        raise RuntimeError(f"Empty pair map: {pair_map}")

    reset_dir(test_group_dir, args.overwrite)
    reset_dir(test_group_pair_dir, args.overwrite)

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[int(row["test_group_id"])].append(row)

    summary: list[dict[str, object]] = []
    for group_id in tqdm(sorted(groups), desc="Export group views", unit="group"):
        group_rows = sorted(groups[group_id], key=lambda item: item["test_file"])
        folder_name = group_folder_name(group_id, group_rows)
        group_dir = test_group_dir / folder_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for row in group_rows:
            src = test_dir / row["test_file"]
            shutil.copy2(src, group_dir / row["test_file"])

        rep_row = next((row for row in group_rows if row.get("is_group_rep") == "1"), group_rows[0])
        test_path = test_dir / rep_row["test_file"]
        if args.right_image == "drop":
            right_path = Path(rep_row["match_drop_path"])
            right_label = f"drop_{rep_row['match_group']}_{Path(rep_row['match_drop_file']).stem}"
        else:
            if not rep_row.get("clear_path"):
                raise RuntimeError("pair_map has no clear_path; cannot export --right-image clear")
            right_path = Path(rep_row["clear_path"])
            right_label = f"clear_{rep_row['clear_group']}_{Path(rep_row['clear_file']).stem}"
        pair_image = concat_images(test_path, right_path)
        pair_name = (
            f"group_{group_id:03d}_test_{Path(rep_row['test_file']).stem}"
            f"__{right_label}.png"
        )
        pair_image.save(test_group_pair_dir / pair_name)

        summary.append({
            "test_group_id": group_id,
            "test_group_folder": folder_name,
            "test_count": len(group_rows),
            "test_rep": rep_row["test_file"],
            "match_group": rep_row["match_group"],
            "match_drop_file": rep_row["match_drop_file"],
            "clear_group": rep_row.get("clear_group", ""),
            "clear_file": rep_row.get("clear_file", ""),
            "pair_image": pair_name,
            "best_distance": float(rep_row["best_distance"]),
            "margin": float(rep_row["margin"]),
        })

    summary_csv = test_group_pair_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    manifest = {
        "pair_map": str(pair_map.resolve()),
        "test_dir": str(test_dir),
        "test_group_dir": str(test_group_dir.resolve()),
        "test_group_pair_dir": str(test_group_pair_dir.resolve()),
        "right_image": args.right_image,
        "groups": len(groups),
        "test_images": len(rows),
        "summary_csv": str(summary_csv.resolve()),
    }
    (test_group_pair_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
