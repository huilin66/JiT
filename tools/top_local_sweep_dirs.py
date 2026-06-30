#!/usr/bin/env python3
"""Print top prediction directories from a local sweep CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = []
    for row in rows:
        try:
            score = float(row.get("score", ""))
        except ValueError:
            continue
        path = Path(row.get("prediction_dir", ""))
        if not path.is_dir():
            continue
        if args.min_score is not None and score < args.min_score:
            continue
        scored.append((score, path))
    scored.sort(key=lambda item: item[0], reverse=True)
    for _, path in scored[: args.top_k]:
        print(path)


if __name__ == "__main__":
    main()
