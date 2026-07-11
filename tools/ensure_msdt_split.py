#!/usr/bin/env python3
"""Create or validate MSDT's deterministic scene-group split before multi-GPU launch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msdt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/raindrop_no_scene.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    msdt_root = Path(args.msdt_root).expanduser().resolve()
    if not (msdt_root / "raindrop_engine.py").is_file():
        raise FileNotFoundError(f"Invalid MSDT root: {msdt_root}")

    sys.path.insert(0, str(msdt_root))
    os.chdir(msdt_root)
    from raindrop_engine import prepare_datasets  # pylint: disable=import-outside-toplevel
    from raindrop_utils import load_config  # pylint: disable=import-outside-toplevel

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = msdt_root / config_path
    config = load_config(str(config_path))
    train_dataset, val_dataset, info = prepare_datasets(
        config,
        data_root=args.data_root,
        scene_json=None,
    )
    manifest = Path(config["data"]["split_manifest"])
    if not manifest.is_absolute():
        manifest = msdt_root / manifest
    if not manifest.is_file():
        raise RuntimeError(f"Split creation returned without a manifest: {manifest}")
    if not len(train_dataset) or not len(val_dataset):
        raise RuntimeError("Generated split contains an empty train or validation dataset")
    print(
        json.dumps(
            {
                "split_manifest": str(manifest),
                "created_or_validated": True,
                **info,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
