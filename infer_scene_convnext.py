import argparse
import csv
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scene_convnext import CLASS_NAMES, SceneInferenceDataset, build_transforms


def parse_args():
    parser = argparse.ArgumentParser("Infer ConvNeXt scene labels")
    parser.add_argument("--input-dir", required=True, help="Test Drop directory")
    parser.add_argument("--checkpoint", required=True, help="checkpoint-best.pth")
    parser.add_argument("--output-json", required=True, help="JiT-compatible filename-to-class JSON")
    parser.add_argument("--output-csv", default="", help="Optional confidence/probability details")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--no-recursive", action="store_true")
    return parser.parse_args()


def resolve_amp(device, requested):
    if device.type != "cuda" or requested == "fp32":
        return None
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16 if requested == "bf16" else torch.float16


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(device, args.amp_dtype)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError("Expected a scene checkpoint containing the 'model' state dict")

    model_name = checkpoint.get("model_name", "convnext_tiny")
    image_size = int(checkpoint.get("image_size", 224))
    num_classes = int(checkpoint.get("num_classes", len(CLASS_NAMES)))
    if num_classes != len(CLASS_NAMES):
        raise ValueError(f"Expected 4 scene classes, checkpoint has {num_classes}")

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    _, eval_transform = build_transforms(image_size)
    dataset = SceneInferenceDataset(args.input_dir, eval_transform, recursive=not args.no_recursive)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    predictions = {}
    with torch.inference_mode():
        for images, names in tqdm(loader, desc="ConvNeXt scene inference"):
            images = images.to(device, non_blocking=True)
            context = torch.autocast("cuda", dtype=amp_dtype) if device.type == "cuda" and amp_dtype else nullcontext()
            with context:
                logits = model(images)
            probabilities = logits.float().softmax(dim=1).cpu()
            labels = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values
            for index, name in enumerate(names):
                class_id = int(labels[index])
                predictions[name] = class_id
                row = {
                    "filename": name,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": float(confidences[index]),
                }
                for prob_id in CLASS_NAMES:
                    row[f"prob_{prob_id}"] = float(probabilities[index, prob_id])
                rows.append(row)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2, sort_keys=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["filename", "class_id", "class_name", "confidence"] + [
            f"prob_{class_id}" for class_id in CLASS_NAMES
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV: {output_csv}")
    print("Class counts:", dict(sorted(Counter(predictions.values()).items())))


if __name__ == "__main__":
    main()
