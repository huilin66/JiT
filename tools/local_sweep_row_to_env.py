#!/usr/bin/env python3
"""Print shell assignments from one local sweep CSV row."""

from __future__ import annotations

import argparse
import csv
import shlex


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--row", type=int, default=0, help="1-based data row index.")
    parser.add_argument("--model-name", default="", help="Use the last matching model_name row.")
    return parser.parse_args()


def emit(name: str, value: str):
    print(f"{name}={shlex.quote(str(value))}")


def main():
    args = parse_args()
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in CSV: {args.csv}")

    if args.model_name:
        matches = [row for row in rows if row.get("model_name") == args.model_name]
        if not matches:
            raise RuntimeError(f"No row with model_name={args.model_name!r}")
        row = matches[-1]
    else:
        if args.row <= 0 or args.row > len(rows):
            raise RuntimeError(f"--row must be in [1, {len(rows)}], got {args.row}")
        row = rows[args.row - 1]

    emit("JIT_CKPT", row.get("checkpoint", ""))
    emit("JIT_CKPT_TYPES", row.get("ckpt_type", "best"))
    emit("STATE_KEYS", row.get("state_key", "model_ema1"))
    emit("STEPS_LIST", row.get("steps", "1"))
    emit("STRIDES", row.get("stride", "128"))
    emit("TILE_BATCH_SIZE", row.get("tile_batch_size", "32"))
    emit("SCENE_JSON", row.get("scene_json", ""))
    emit("CONFIG_SOURCE_MODEL_NAME", row.get("model_name", ""))
    emit("NOTES", f"replay from {args.csv}")


if __name__ == "__main__":
    main()
