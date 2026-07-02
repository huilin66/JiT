#!/usr/bin/env python3
"""Compare JiT-only and MSDT-refiner predictions on a paired local set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from piq import ssim
from tqdm import tqdm


CSV_FIELDS = [
    "filename",
    "jit_psnr_y",
    "jit_ssim_y",
    "jit_lpips",
    "jit_score",
    "refiner_psnr_y",
    "refiner_ssim_y",
    "refiner_lpips",
    "refiner_score",
    "delta_psnr_y",
    "delta_ssim_y",
    "delta_lpips",
    "delta_score",
    "drop_clear_l1",
    "jit_error_l1",
    "refiner_error_l1",
    "jit_residual_ratio",
    "refiner_change_l1",
    "refiner_change_ratio",
    "refiner_win",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-dir", default="", help="Original rainy Drop directory. Optional but recommended.")
    parser.add_argument("--clear-dir", required=True)
    parser.add_argument("--jit-dir", required=True, help="JiT-only prediction directory.")
    parser.add_argument("--refiner-dir", required=True, help="JiT+MSDT-refiner prediction directory.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def score_image(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> dict[str, float]:
    pred_y = rgb_to_y(pred)
    target_y = rgb_to_y(target)
    mse = torch.mean((pred_y - target_y) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    psnr_y = float((10.0 * torch.log10(1.0 / mse)).mean().cpu())
    ssim_y = float(ssim(pred_y, target_y, data_range=1.0).cpu())
    lpips_value = float(lpips_model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().cpu())
    return {
        "psnr_y": psnr_y,
        "ssim_y": ssim_y,
        "lpips": lpips_value,
        "score": psnr_y + 10.0 * ssim_y - 5.0 * lpips_value,
    }


def corr(rows: list[dict], x_key: str, y_key: str) -> float | None:
    x = np.asarray([row[x_key] for row in rows if row[x_key] != ""], dtype=np.float64)
    y = np.asarray([row[y_key] for row in rows if row[x_key] != ""], dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] != ""]
    if not values:
        return None
    return float(np.mean(values))


def rounded(value):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return round(float(value), 6)


def build_summary(rows: list[dict]) -> dict:
    improved = [row for row in rows if row["delta_score"] > 0.0]
    worsened = [row for row in rows if row["delta_score"] <= 0.0]
    summary = {
        "num_images": len(rows),
        "win_rate": len(improved) / max(len(rows), 1),
        "avg_jit_score": mean(rows, "jit_score"),
        "avg_refiner_score": mean(rows, "refiner_score"),
        "avg_delta_score": mean(rows, "delta_score"),
        "avg_delta_psnr_y": mean(rows, "delta_psnr_y"),
        "avg_delta_ssim_y": mean(rows, "delta_ssim_y"),
        "avg_delta_lpips": mean(rows, "delta_lpips"),
        "avg_jit_residual_ratio": mean(rows, "jit_residual_ratio"),
        "avg_refiner_change_l1": mean(rows, "refiner_change_l1"),
        "corr_jit_score_vs_delta_score": corr(rows, "jit_score", "delta_score"),
        "corr_jit_residual_ratio_vs_delta_score": corr(rows, "jit_residual_ratio", "delta_score"),
        "corr_refiner_change_l1_vs_delta_score": corr(rows, "refiner_change_l1", "delta_score"),
        "avg_delta_score_when_refiner_wins": mean(improved, "delta_score"),
        "avg_delta_score_when_refiner_loses": mean(worsened, "delta_score"),
        "worst_10_by_delta_score": [
            {
                "filename": row["filename"],
                "delta_score": row["delta_score"],
                "jit_score": row["jit_score"],
                "refiner_score": row["refiner_score"],
                "jit_residual_ratio": row["jit_residual_ratio"],
                "refiner_change_l1": row["refiner_change_l1"],
            }
            for row in sorted(rows, key=lambda item: item["delta_score"])[:10]
        ],
        "best_10_by_delta_score": [
            {
                "filename": row["filename"],
                "delta_score": row["delta_score"],
                "jit_score": row["jit_score"],
                "refiner_score": row["refiner_score"],
                "jit_residual_ratio": row["jit_residual_ratio"],
                "refiner_change_l1": row["refiner_change_l1"],
            }
            for row in sorted(rows, key=lambda item: item["delta_score"], reverse=True)[:10]
        ],
    }
    return {key: rounded(value) if isinstance(value, float) else value for key, value in summary.items()}


def main():
    args = parse_args()
    drop_dir = Path(args.drop_dir) if args.drop_dir else None
    clear_dir = Path(args.clear_dir)
    jit_dir = Path(args.jit_dir)
    refiner_dir = Path(args.refiner_dir)
    csv_path = Path(args.csv)

    jit_files = sorted(jit_dir.glob("*.png"))
    if not jit_files:
        raise RuntimeError(f"No JiT PNG files found in {jit_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    rows = []

    with torch.inference_mode():
        for jit_path in tqdm(jit_files, desc="Comparing JiT vs refiner"):
            clear_path = clear_dir / jit_path.name
            refiner_path = refiner_dir / jit_path.name
            if not clear_path.is_file():
                raise FileNotFoundError(f"Missing Clear image: {clear_path}")
            if not refiner_path.is_file():
                raise FileNotFoundError(f"Missing refiner prediction: {refiner_path}")

            clear = load_rgb(clear_path).to(device)
            jit = load_rgb(jit_path).to(device)
            refiner = load_rgb(refiner_path).to(device)
            if jit.shape != clear.shape or refiner.shape != clear.shape:
                raise RuntimeError(f"Shape mismatch for {jit_path.name}")

            jit_metrics = score_image(jit, clear, lpips_model)
            refiner_metrics = score_image(refiner, clear, lpips_model)
            jit_error_l1 = float(torch.mean(torch.abs(jit - clear)).cpu())
            refiner_error_l1 = float(torch.mean(torch.abs(refiner - clear)).cpu())
            refiner_change_l1 = float(torch.mean(torch.abs(refiner - jit)).cpu())

            drop_clear_l1 = ""
            jit_residual_ratio = ""
            if drop_dir is not None:
                drop_path = drop_dir / jit_path.name
                if not drop_path.is_file():
                    raise FileNotFoundError(f"Missing Drop image: {drop_path}")
                drop = load_rgb(drop_path).to(device)
                if drop.shape != clear.shape:
                    raise RuntimeError(f"Shape mismatch for Drop image: {drop_path.name}")
                drop_clear_l1 = float(torch.mean(torch.abs(drop - clear)).cpu())
                jit_residual_ratio = jit_error_l1 / max(drop_clear_l1, 1e-8)

            refiner_change_ratio = refiner_change_l1 / max(jit_error_l1, 1e-8)
            delta_score = refiner_metrics["score"] - jit_metrics["score"]
            rows.append(
                {
                    "filename": jit_path.name,
                    "jit_psnr_y": jit_metrics["psnr_y"],
                    "jit_ssim_y": jit_metrics["ssim_y"],
                    "jit_lpips": jit_metrics["lpips"],
                    "jit_score": jit_metrics["score"],
                    "refiner_psnr_y": refiner_metrics["psnr_y"],
                    "refiner_ssim_y": refiner_metrics["ssim_y"],
                    "refiner_lpips": refiner_metrics["lpips"],
                    "refiner_score": refiner_metrics["score"],
                    "delta_psnr_y": refiner_metrics["psnr_y"] - jit_metrics["psnr_y"],
                    "delta_ssim_y": refiner_metrics["ssim_y"] - jit_metrics["ssim_y"],
                    "delta_lpips": refiner_metrics["lpips"] - jit_metrics["lpips"],
                    "delta_score": delta_score,
                    "drop_clear_l1": drop_clear_l1,
                    "jit_error_l1": jit_error_l1,
                    "refiner_error_l1": refiner_error_l1,
                    "jit_residual_ratio": jit_residual_ratio,
                    "refiner_change_l1": refiner_change_l1,
                    "refiner_change_ratio": refiner_change_ratio,
                    "refiner_win": int(delta_score > 0.0),
                }
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: rounded(row[key]) for key in CSV_FIELDS})

    summary = build_summary(rows)
    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = csv_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
