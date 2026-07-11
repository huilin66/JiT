#!/usr/bin/env python3
"""Reset an MSDT training checkpoint for a short, low-LR fine-tune.

The model weights are preserved. Adam moments, the best score, RNG state, and
the cosine scheduler are reset so train_raindrop.py can safely use --resume
without silently restoring the old learning rate.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source MSDT training checkpoint")
    parser.add_argument("--output", required=True, help="Prepared checkpoint path")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("Require 0 <= min_lr <= lr")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "scaler"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Source checkpoint misses training state: {sorted(missing)}")

    prepared = copy.deepcopy(checkpoint)
    optimizer = prepared["optimizer"]
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = float(args.lr)
        group["initial_lr"] = float(args.lr)

    scheduler = prepared["scheduler"]
    scheduler["T_max"] = int(args.epochs)
    scheduler["eta_min"] = float(args.min_lr)
    scheduler["base_lrs"] = [float(args.lr)] * len(optimizer["param_groups"])
    scheduler["last_epoch"] = 0
    scheduler["_step_count"] = 1
    scheduler["_last_lr"] = [float(args.lr)] * len(optimizer["param_groups"])

    prepared["epoch"] = 0
    prepared["best_score"] = float("-inf")
    prepared.pop("rng_state", None)
    embedded = prepared.get("config")
    if isinstance(embedded, dict):
        embedded.setdefault("training", {})["epochs"] = int(args.epochs)
        embedded.setdefault("optimizer", {})["lr"] = float(args.lr)
        embedded["optimizer"]["min_lr"] = float(args.min_lr)
        embedded["finetune_source"] = str(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared, destination)
    print(
        f"Prepared {destination}\n"
        f"source={source}\nmodel_preserved=True\noptimizer_state=reset\n"
        f"lr={args.lr}\nmin_lr={args.min_lr}\nepochs={args.epochs}"
    )


if __name__ == "__main__":
    main()
