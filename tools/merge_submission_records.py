#!/usr/bin/env python3
"""Merge submission record CSV files without dropping source-specific fields."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_RECORD_DIR = Path("demo/records")
DEFAULT_OUTPUT = DEFAULT_RECORD_DIR / "submission_records_merged.csv"
METADATA_COLUMNS = [
    "record_source_file",
    "record_source_row",
    "record_stage",
    "record_scope",
    "record_family",
    "score_float",
    "score_rank_desc",
]
PREFERRED_COLUMNS = [
    "timestamp",
    "model_name",
    "archive_name",
    "score",
    "checkpoint",
    "state_key",
    "architecture",
    "config",
    "use_scene",
    "use_bg_subnet",
    "input_mode",
    "input_dir",
    "input_path",
    "data_root",
    "scene_json",
    "scene_id",
    "steps",
    "patch_size",
    "tile_size",
    "tile_overlap",
    "stride",
    "tile_batch_size",
    "scale",
    "vflip",
    "hflip",
    "rot90",
    "rot180",
    "rot270",
    "num_images",
    "runtime_seconds",
    "output_dir",
    "psnr_y",
    "ssim_y",
    "lpips",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", default=str(DEFAULT_RECORD_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Optional explicit CSV files. Defaults to all CSVs in --record-dir except merged outputs.",
    )
    return parser.parse_args()


def score_value(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def infer_scope(row: dict[str, str], source_file: str) -> str:
    text = " ".join([
        source_file.lower(),
        row.get("input_dir", "").lower(),
        row.get("input_path", "").lower(),
        row.get("archive_name", "").lower(),
        row.get("checkpoint", "").lower(),
        row.get("notes", "").lower(),
    ])
    if row.get("num_images", "") == "592" or "test-input" in text or "_test" in source_file.lower():
        return "test"
    if row.get("num_images", "") == "406" or "drop" in text:
        return "drop"
    return ""


def timestamp_date(row: dict[str, str]) -> int | None:
    timestamp = row.get("timestamp", "").strip()
    if len(timestamp) < 8 or not timestamp[:8].isdigit():
        return None
    return int(timestamp[:8])


def infer_stage(row: dict[str, str], source_file: str) -> str:
    score = score_value(row.get("score", ""))
    text = " ".join([
        source_file.lower(),
        row.get("input_dir", "").lower(),
        row.get("input_path", "").lower(),
        row.get("archive_name", "").lower(),
        row.get("checkpoint", "").lower(),
        row.get("notes", "").lower(),
    ])
    if score is not None and score >= 34.0:
        return "preliminary"
    if row.get("num_images", "") == "406" or "drop" in text:
        return "preliminary"
    if row.get("num_images", "") == "592" or "test-input" in text or "_test" in source_file.lower():
        return "final"
    date = timestamp_date(row)
    if date is not None and date <= 20260710:
        return "preliminary"
    return ""


def infer_family(row: dict[str, str], source_file: str) -> str:
    text = " ".join([
        source_file.lower(),
        row.get("model_name", "").lower(),
        row.get("archive_name", "").lower(),
        row.get("checkpoint", "").lower(),
        row.get("notes", "").lower(),
    ])
    if "ensemble" in text:
        return "ensemble"
    if "msdt" in text:
        return "msdt"
    if "jit" in text:
        return "jit"
    return ""


def input_paths(record_dir: Path, output: Path, explicit_inputs: list[str]) -> list[Path]:
    if explicit_inputs:
        return [Path(path) for path in explicit_inputs]
    output_name = output.name.lower()
    return sorted(
        path for path in record_dir.glob("*.csv")
        if path.name.lower() != output_name and not path.name.lower().startswith("submission_records_merged")
    )


def sort_key(row: dict[str, str]) -> tuple[int, float, str, str, int]:
    score = score_value(row.get("score", ""))
    has_score = 0 if score is not None else 1
    return (
        has_score,
        -(score if score is not None else -1.0),
        row.get("record_scope", ""),
        row.get("timestamp", ""),
        int(row.get("record_source_row", "0") or "0"),
    )


def main() -> None:
    args = parse_args()
    record_dir = Path(args.record_dir)
    output = Path(args.output)
    paths = input_paths(record_dir, output, args.inputs)
    if not paths:
        raise FileNotFoundError(f"No input CSV files found in: {record_dir}")

    rows: list[dict[str, str]] = []
    source_fields: list[str] = []
    seen_fields: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for field in reader.fieldnames:
                if field not in seen_fields:
                    seen_fields.add(field)
                    source_fields.append(field)
            for index, row in enumerate(reader, start=2):
                normalized = {key: (value if value is not None else "") for key, value in row.items()}
                normalized["record_source_file"] = path.name
                normalized["record_source_row"] = str(index)
                normalized["record_stage"] = infer_stage(normalized, path.name)
                normalized["record_scope"] = infer_scope(normalized, path.name)
                normalized["record_family"] = infer_family(normalized, path.name)
                score = score_value(normalized.get("score", ""))
                normalized["score_float"] = "" if score is None else f"{score:.6f}"
                normalized["score_rank_desc"] = ""
                rows.append(normalized)

    scored = sorted(
        (row for row in rows if score_value(row.get("score", "")) is not None),
        key=lambda row: score_value(row.get("score", "")),
        reverse=True,
    )
    for rank, row in enumerate(scored, start=1):
        row["score_rank_desc"] = str(rank)

    fields: list[str] = []
    for field in METADATA_COLUMNS + PREFERRED_COLUMNS + source_fields:
        if field not in fields:
            fields.append(field)

    rows = sorted(rows, key=sort_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    scored_count = sum(1 for row in rows if row.get("score_float"))
    print(f"Wrote {len(rows)} rows from {len(paths)} files to {output}")
    print(f"Rows with score: {scored_count}; rows without score: {len(rows) - scored_count}")


if __name__ == "__main__":
    main()
