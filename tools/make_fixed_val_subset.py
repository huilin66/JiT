#!/usr/bin/env python3
"""Create a fixed paired Drop/Clear validation subset for local submit sweeps."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Dataset root with Drop and Clear folders.")
    parser.add_argument("--output-root", required=True, help="Subset root to create, with Drop and Clear folders.")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scene-json", default="", help="Optional source scene JSON keyed by filename.")
    parser.add_argument("--output-scene-json", default="", help="Default: output-root/<scene-json name>.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(directory: Path):
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    drop_dir = data_root / "Drop"
    clear_dir = data_root / "Clear"
    out_drop = output_root / "Drop"
    out_clear = output_root / "Clear"

    if out_drop.exists() and out_clear.exists() and not args.overwrite:
        existing = list_images(out_drop)
        if existing:
            print(f"Fixed subset already exists: {output_root} ({len(existing)} images)")
            return

    drop_images = list_images(drop_dir)
    clear_names = {path.name for path in list_images(clear_dir)}
    paired = [path for path in drop_images if path.name in clear_names]
    if len(paired) < args.num_samples:
        raise RuntimeError(f"Only {len(paired)} paired images found, need {args.num_samples}")

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    out_drop.mkdir(parents=True, exist_ok=True)
    out_clear.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(paired, args.num_samples), key=lambda path: path.name)
    for drop_path in selected:
        shutil.copy2(drop_path, out_drop / drop_path.name)
        shutil.copy2(clear_dir / drop_path.name, out_clear / drop_path.name)

    manifest = {
        "data_root": str(data_root.resolve()),
        "num_samples": len(selected),
        "seed": args.seed,
        "filenames": [path.name for path in selected],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if args.scene_json:
        scene_path = Path(args.scene_json)
        with scene_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
        subset_labels = {path.name: int(labels[path.name]) for path in selected}
        output_scene_json = Path(args.output_scene_json) if args.output_scene_json else output_root / scene_path.name
        output_scene_json.parent.mkdir(parents=True, exist_ok=True)
        with output_scene_json.open("w", encoding="utf-8") as handle:
            json.dump(subset_labels, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Scene subset JSON: {output_scene_json}")

    print(f"Created fixed subset: {output_root} ({len(selected)} images)")


if __name__ == "__main__":
    main()
