#!/usr/bin/env python3
"""Override predicted test scene labels with focus2scene pseudo labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene-json", required=True, help="Predicted scene JSON keyed by test filename.")
    parser.add_argument(
        "--focus2scene-pseudo-json",
        required=True,
        help="Drop_focus_2scene_test_pseudo.json containing test_pseudo_<filename> labels.",
    )
    parser.add_argument("--input-dir", required=True, help="Test input directory used by submission.")
    parser.add_argument("--output-json", required=True, help="Updated scene JSON path.")
    parser.add_argument("--manifest-json", default="", help="Optional update manifest path.")
    parser.add_argument("--prefix", default="test_pseudo_", help="Pseudo train-copy filename prefix.")
    parser.add_argument("--require-any", action="store_true", help="Fail if no pseudo labels match test images.")
    return parser.parse_args()


def load_json_dict(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def list_image_names(input_dir: Path) -> list[str]:
    names = sorted(
        path.name
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not names:
        raise ValueError(f"No input images found below {input_dir}")
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate input image names are not supported, e.g. {duplicates[:5]}")
    return names


def parse_focus2(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"Illegal focus2scene label for {key!r}: {value!r}; expected 0 or 1")
    return int(value)


def main() -> None:
    args = parse_args()
    scene_json = Path(args.scene_json)
    focus_json = Path(args.focus2scene_pseudo_json)
    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)
    manifest_json = Path(args.manifest_json) if args.manifest_json else output_json.with_suffix(".manifest.json")

    labels = {str(name): int(value) for name, value in load_json_dict(scene_json).items()}
    pseudo = load_json_dict(focus_json)
    image_names = list_image_names(input_dir)

    missing_scene = [name for name in image_names if name not in labels]
    if missing_scene:
        raise KeyError(f"Scene JSON misses {len(missing_scene)} input image(s), first: {missing_scene[0]}")

    updated = dict(labels)
    changes: dict[str, dict[str, int | str]] = {}
    missing_pseudo: list[str] = []
    for name in image_names:
        pseudo_key = args.prefix + name
        if pseudo_key not in pseudo:
            if name in pseudo:
                pseudo_key = name
            else:
                missing_pseudo.append(name)
                continue
        old_label = int(updated[name])
        new_label = parse_focus2(pseudo[pseudo_key], pseudo_key)
        updated[name] = new_label
        changes[name] = {
            "pseudo_key": pseudo_key,
            "old": old_label,
            "new": new_label,
        }

    if args.require_any and not changes:
        raise RuntimeError(
            "No test images matched focus2scene pseudo labels. "
            f"Expected keys like {args.prefix}<test filename> in {focus_json}."
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(updated, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "scene_json": str(scene_json),
        "focus2scene_pseudo_json": str(focus_json),
        "input_dir": str(input_dir),
        "output_json": str(output_json),
        "prefix": args.prefix,
        "images": len(image_names),
        "updated": len(changes),
        "missing_pseudo": len(missing_pseudo),
        "old_counts": dict(sorted(Counter(labels[name] for name in image_names).items())),
        "new_counts": dict(sorted(Counter(updated[name] for name in image_names).items())),
        "changes": changes,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    print(
        "Updated scene labels from focus2scene pseudo: "
        f"{len(changes)}/{len(image_names)} images; missing_pseudo={len(missing_pseudo)}"
    )
    print(f"Saved updated scene JSON: {output_json}")
    print(f"Saved update manifest: {manifest_json}")


if __name__ == "__main__":
    main()
