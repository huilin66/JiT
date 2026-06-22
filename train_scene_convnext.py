import argparse
import csv
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scene_convnext import (
    CLASS_NAMES,
    SceneDataset,
    balanced_class_weights,
    build_transforms,
    class_counts,
    grouped_train_val_split,
    load_scene_samples,
    save_split_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser("Train ConvNeXt scene classifier")
    parser.add_argument("--data-root", required=True, help="Root containing Drop and Drop_scen_pred.json")
    parser.add_argument("--image-dir", default="", help="Default: data-root/Drop")
    parser.add_argument("--labels-json", default="", help="Default: data-root/Drop_scen_pred.json")
    parser.add_argument("--output-dir", default="run/scene_convnext")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(device, requested):
    if device.type != "cuda" or requested == "fp32":
        return None
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16 if requested == "bf16" else torch.float16


def amp_context(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype else nullcontext()


def create_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, criterion, device, amp_dtype, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0
    class_correct = torch.zeros(len(CLASS_NAMES), dtype=torch.long)
    class_total = torch.zeros(len(CLASS_NAMES), dtype=torch.long)

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in tqdm(loader, desc="train" if training else "val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite scene loss: {loss.item()}")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predictions = logits.argmax(dim=1)
            batch_size = labels.numel()
            loss_sum += loss.item() * batch_size
            correct += predictions.eq(labels).sum().item()
            count += batch_size
            for class_id in CLASS_NAMES:
                mask = labels == class_id
                class_total[class_id] += mask.sum().cpu()
                class_correct[class_id] += predictions[mask].eq(labels[mask]).sum().cpu()

    per_class = {
        class_id: class_correct[class_id].item() / max(1, class_total[class_id].item())
        for class_id in CLASS_NAMES
    }
    macro_accuracy = sum(per_class.values()) / len(per_class)
    return {
        "loss": loss_sum / max(1, count),
        "accuracy": correct / max(1, count),
        "macro_accuracy": macro_accuracy,
        "per_class": per_class,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_accuracy, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "model_name": args.model,
            "image_size": args.image_size,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(device, args.amp_dtype)
    scaler = create_scaler(device.type == "cuda" and amp_dtype == torch.float16)

    data_root = Path(args.data_root)
    image_dir = Path(args.image_dir) if args.image_dir else data_root / "Drop"
    labels_json = Path(args.labels_json) if args.labels_json else data_root / "Drop_scen_pred.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_scene_samples(image_dir, labels_json)
    train_samples, val_samples, val_groups = grouped_train_val_split(
        samples, val_fraction=args.val_fraction, seed=args.seed
    )
    save_split_manifest(output_dir / "split_manifest.json", train_samples, val_samples, val_groups, args.seed)
    print(f"Train: {len(train_samples)} {class_counts(train_samples)}")
    print(f"Val: {len(val_samples)} {class_counts(val_samples)}")

    train_transform, eval_transform = build_transforms(args.image_size)
    train_loader = DataLoader(
        SceneDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        SceneDataset(val_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(CLASS_NAMES))
    model.to(device)
    class_weights = None if args.no_class_weights else balanced_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_accuracy = -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_accuracy", -math.inf))
        print(f"Resumed from epoch {start_epoch}")

    history_path = output_dir / "training_history.csv"
    write_header = not history_path.exists() or start_epoch == 0
    with open(history_path, "a", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_acc"],
        )
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            train_metrics = run_epoch(model, train_loader, criterion, device, amp_dtype, optimizer, scaler)
            val_metrics = run_epoch(model, val_loader, criterion, device, amp_dtype)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            print(
                f"Epoch {epoch + 1}/{args.epochs} lr={lr:.3e} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_acc={val_metrics['macro_accuracy']:.4f} per_class={val_metrics['per_class']}"
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_macro_acc": val_metrics["macro_accuracy"],
                }
            )
            history_file.flush()
            is_best = val_metrics["macro_accuracy"] > best_accuracy
            if is_best:
                best_accuracy = val_metrics["macro_accuracy"]
            save_checkpoint(
                output_dir / "checkpoint-last.pth",
                model, optimizer, scheduler, scaler, epoch, best_accuracy, args,
            )
            if is_best:
                save_checkpoint(
                    output_dir / "checkpoint-best.pth",
                    model, optimizer, scheduler, scaler, epoch, best_accuracy, args,
                )
                print(f"Saved new best checkpoint: macro_acc={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
