#!/usr/bin/env python3
"""Build validated flat-PNG deadline fusion, adaptive fusion, or backblend ZIPs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def png_map(directory: str) -> dict[str, Path]:
    root = Path(directory)
    files = {path.name: path for path in root.glob("*.png")}
    if not files:
        raise ValueError(f"No flat PNG files in {root}")
    return files


def aligned_maps(*directories: str) -> tuple[list[str], list[dict[str, Path]]]:
    maps = [png_map(directory) for directory in directories]
    names = sorted(maps[0])
    for directory, current in zip(directories[1:], maps[1:]):
        if sorted(current) != names:
            raise ValueError(f"PNG filename set differs: {directory}")
    return names, maps


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(array: np.ndarray, path: Path) -> None:
    output = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(output, mode="RGB").save(path, format="PNG", compress_level=6)


def package(directory: Path, archive_path: Path, names: list[str]) -> str:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(directory / name, arcname=name)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return digest


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", required=True, help="Original rainy flat-PNG directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0, help="R = X + beta*(prediction-X)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backblend = subparsers.add_parser("backblend")
    common_parser(backblend)
    backblend.add_argument("--restored-dir", required=True)

    adaptive = subparsers.add_parser("adaptive")
    common_parser(adaptive)
    adaptive.add_argument("--jit-dir", required=True)
    adaptive.add_argument("--msdt-dir", required=True)
    adaptive.add_argument("--threshold", type=float, default=0.02)
    adaptive.add_argument("--low-jit-weight", type=float, default=0.50)
    adaptive.add_argument("--high-jit-weight", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be in [0, 1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, object]] = []

    if args.command == "backblend":
        names, maps = aligned_maps(args.input_dir, args.restored_dir)
        input_map, restored_map = maps
        for name in tqdm(names, desc="backblend"):
            rainy, restored = load_rgb(input_map[name]), load_rgb(restored_map[name])
            if rainy.shape != restored.shape:
                raise ValueError(f"Shape mismatch for {name}: {rainy.shape} vs {restored.shape}")
            result = rainy + args.beta * (restored - rainy)
            save_rgb(result, output_dir / name)
    else:
        if not 0 <= args.low_jit_weight <= 1 or not 0 <= args.high_jit_weight <= 1:
            raise ValueError("JiT weights must be in [0, 1]")
        names, maps = aligned_maps(args.input_dir, args.jit_dir, args.msdt_dir)
        input_map, jit_map, msdt_map = maps
        for name in tqdm(names, desc="adaptive fusion"):
            rainy, jit, msdt = load_rgb(input_map[name]), load_rgb(jit_map[name]), load_rgb(msdt_map[name])
            if rainy.shape != jit.shape or rainy.shape != msdt.shape:
                raise ValueError(f"Shape mismatch for {name}")
            disagreement = float(np.abs(jit - msdt).mean())
            jit_weight = args.high_jit_weight if disagreement > args.threshold else args.low_jit_weight
            restored = jit_weight * jit + (1.0 - jit_weight) * msdt
            result = rainy + args.beta * (restored - rainy)
            save_rgb(result, output_dir / name)
            diagnostics.append({"filename": name, "disagreement": disagreement, "jit_weight": jit_weight})

    if args.expected_count and len(names) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} PNGs, found {len(names)}")
    digest = package(output_dir, Path(args.archive_path), names)
    if diagnostics:
        with (output_dir / "adaptive_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "disagreement", "jit_weight"])
            writer.writeheader()
            writer.writerows(diagnostics)
    manifest = {
        "command": args.command,
        "count": len(names),
        "beta": args.beta,
        "archive": str(Path(args.archive_path).resolve()),
        "sha256": digest,
        "arguments": vars(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
