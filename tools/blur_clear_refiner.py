#!/usr/bin/env python3
"""Train Blur->Clear refiners and apply them to flat prediction folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Direct execution (`python tools/blur_clear_refiner.py`) puts only tools/ on
# sys.path. Add the JiT repository root so root-level model modules resolve on
# both Linux servers and Windows workstations.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_jit import BackgroundRestorationSubnet
from model_msdt_refiner import MSDTDetailRefiner


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_GROUP_REGEX = r"(?i)^(day|night)(?:raindrop)?(?:__|[_-])(\d+)"


def image_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.relative_to(root).as_posix()
            if key in result:
                raise ValueError(f"Duplicate relative image path: {key}")
            result[key] = path
    if not result:
        raise ValueError(f"No images below {root}")
    return result


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB").save(
        path, format="PNG", compress_level=6
    )


def group_key(name: str, pattern: re.Pattern[str]) -> str:
    match = pattern.match(Path(name).stem)
    if match is None:
        raise ValueError(f"Cannot parse scene/triplet group from filename: {name!r}")
    return "_".join(str(value).lower() for value in match.groups())


class BlurClearDataset(Dataset):
    def __init__(self, blur: dict[str, Path], clear: dict[str, Path], names: list[str], patch: int, train: bool):
        self.blur, self.clear, self.names = blur, clear, names
        self.patch, self.train = patch, train

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        blur, clear = load_rgb(self.blur[name]), load_rgb(self.clear[name])
        if blur.shape != clear.shape:
            raise ValueError(f"Pair shape mismatch: {name}")
        _, height, width = blur.shape
        if min(height, width) < self.patch:
            scale = self.patch / min(height, width)
            size = (math.ceil(height * scale), math.ceil(width * scale))
            blur = F.interpolate(blur[None], size=size, mode="bilinear", align_corners=False)[0]
            clear = F.interpolate(clear[None], size=size, mode="bilinear", align_corners=False)[0]
            _, height, width = blur.shape
        if self.train:
            top = random.randint(0, height - self.patch)
            left = random.randint(0, width - self.patch)
        else:
            top = (height - self.patch) // 2
            left = (width - self.patch) // 2
        blur = blur[:, top:top + self.patch, left:left + self.patch]
        clear = clear[:, top:top + self.patch, left:left + self.patch]
        if self.train:
            if random.random() < 0.5:
                blur, clear = blur.flip(-1), clear.flip(-1)
            if random.random() < 0.5:
                blur, clear = blur.flip(-2), clear.flip(-2)
            k = random.randrange(4)
            if k:
                blur, clear = torch.rot90(blur, k, (-2, -1)), torch.rot90(clear, k, (-2, -1))
        return blur, clear, name


class BlurClearModel(nn.Module):
    def __init__(self, kind: str, base_dim: int, blocks: int, frequency: bool, max_residual: float):
        super().__init__()
        self.kind, self.max_residual = kind, float(max_residual)
        if kind == "background":
            self.refiner = BackgroundRestorationSubnet(3, base_dim, blocks)
            nn.init.zeros_(self.refiner.output_conv.weight)
            nn.init.zeros_(self.refiner.output_conv.bias)
        elif kind == "detail":
            self.refiner = MSDTDetailRefiner(3, base_dim, blocks, frequency, max_residual)
        else:
            raise ValueError(kind)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.kind == "background":
            return image + self.max_residual * torch.tanh(self.refiner(image))
        return self.refiner(image, image)


def sobel(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    kx = tensor.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    return F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)


def rgb_to_y(image: torch.Tensor) -> torch.Tensor:
    coefficients = image.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1) / 255.0
    return (image * coefficients).sum(dim=1, keepdim=True) + 16.0 / 255.0


def ssim_y(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = rgb_to_y(prediction), rgb_to_y(target)
    window_size = min(11, min(x.shape[-2:]) if min(x.shape[-2:]) % 2 else min(x.shape[-2:]) - 1)
    sigma = 1.5 * window_size / 11.0
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]; padding = window_size // 2
    def filt(value): return F.conv2d(F.pad(value, (padding,) * 4, mode="reflect"), kernel)
    mu_x, mu_y = filt(x), filt(y); xx, yy, xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    var_x, var_y, cov = filt(x * x) - xx, filt(y * y) - yy, filt(x * y) - xy
    return (((2 * xy + .01 ** 2) * (2 * cov + .03 ** 2)) /
            ((xx + yy + .01 ** 2) * (var_x + var_y + .03 ** 2))).mean()


class ValidationMetrics:
    def __init__(self, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("LPIPS is required for score-best validation") from exc
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.lpips.requires_grad_(False)

    @torch.no_grad()
    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        prediction, target = prediction.float().clamp(0, 1), target.float().clamp(0, 1)
        y_pred, y_target = rgb_to_y(prediction), rgb_to_y(target)
        mse = F.mse_loss(y_pred, y_target)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        ssim = ssim_y(prediction, target)
        perceptual = self.lpips(prediction * 2 - 1, target * 2 - 1).mean()
        return {"PSNR_Y": float(psnr), "SSIM_Y": float(ssim), "LPIPS": float(perceptual)}


def loss_terms(pred: torch.Tensor, target: torch.Tensor, edge_weight: float, freq_weight: float):
    rec = torch.sqrt((pred - target) ** 2 + 1e-6).mean()
    px, py = sobel(pred.float()); tx, ty = sobel(target.float())
    edge = F.l1_loss(px, tx) + F.l1_loss(py, ty)
    pf = torch.fft.rfft2(pred.float(), norm="ortho")
    tf = torch.fft.rfft2(target.float(), norm="ortho")
    freq = F.l1_loss(torch.log1p(pf.abs()), torch.log1p(tf.abs()))
    return rec + edge_weight * edge + freq_weight * freq, rec, edge, freq


@torch.no_grad()
def validate(model, loader, device, amp_dtype, edge_weight, freq_weight, metrics, max_batches=0):
    model.eval(); totals = np.zeros(4, dtype=np.float64); metric_totals = {"PSNR_Y": 0., "SSIM_Y": 0., "LPIPS": 0.}; samples = 0; count = 0
    for blur, clear, _ in loader:
        blur, clear = blur.to(device), clear.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            pred = model(blur)
        loss, rec, edge, freq = loss_terms(pred, clear, edge_weight, freq_weight)
        values = metrics.compute(pred, clear); batch = blur.shape[0]
        for key in metric_totals: metric_totals[key] += values[key] * batch
        samples += batch
        totals += [loss.item(), rec.item(), edge.item(), freq.item()]
        count += 1
        if max_batches and count >= max_batches:
            break
    result = {key: value / max(samples, 1) for key, value in metric_totals.items()}
    result["Score"] = result["PSNR_Y"] + 10.0 * result["SSIM_Y"] - 5.0 * result["LPIPS"]
    return totals / max(count, 1), result


def train(args) -> None:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = Path(args.data_root); blur = image_map(data / args.blur_dir); clear = image_map(data / args.clear_dir)
    if set(blur) != set(clear):
        raise ValueError(f"Blur/Clear filename mismatch: blur={len(blur)}, clear={len(clear)}")
    pattern = re.compile(args.group_regex)
    groups = sorted({group_key(name, pattern) for name in blur})
    random.Random(args.seed).shuffle(groups)
    val_count = max(1, round(len(groups) * args.val_ratio)); val_groups = set(groups[:val_count])
    train_names = sorted(name for name in blur if group_key(name, pattern) not in val_groups)
    val_names = sorted(name for name in blur if group_key(name, pattern) in val_groups)
    train_set = BlurClearDataset(blur, clear, train_names, args.patch_size, True)
    val_set = BlurClearDataset(blur, clear, val_names, args.patch_size, False)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    device = torch.device(args.device); amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model = BlurClearModel(args.kind, args.base_dim, args.blocks, bool(args.frequency), args.max_residual).to(device)
    init_checkpoint_sha256 = ""
    if args.init_jit_checkpoint:
        if args.kind != "detail":
            raise ValueError("--init-jit-checkpoint is only supported for --kind detail")
        init_path = Path(args.init_jit_checkpoint)
        try:
            initial = torch.load(init_path, map_location="cpu", weights_only=False)
        except TypeError:
            initial = torch.load(init_path, map_location="cpu")
        if args.init_state_key not in initial:
            raise KeyError(f"{init_path} has no state {args.init_state_key!r}")
        prefix = "detail_refiner."
        refiner_state = {
            key[len(prefix):]: value
            for key, value in initial[args.init_state_key].items()
            if key.startswith(prefix)
        }
        if not refiner_state:
            raise KeyError(f"No {prefix} weights in {init_path}:{args.init_state_key}")
        model.refiner.load_state_dict(refiner_state, strict=True)
        init_checkpoint_sha256 = sha256(init_path)
        print(f"Loaded JiT MSDT detail refiner from {init_path}:{args.init_state_key}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"; best_psnr = -float("inf"); best_score = -float("inf")
    validation_metrics = ValidationMetrics(device)
    config = {key: value for key, value in vars(args).items() if key != "func"}
    config.update(train_images=len(train_names), val_images=len(val_names),
                  init_jit_checkpoint_sha256=init_checkpoint_sha256)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["epoch", "loss", "rec", "edge", "freq", "PSNR_Y", "SSIM_Y", "LPIPS", "Score", "lr"])
        for epoch in range(1, args.epochs + 1):
            model.train(); running = 0.0
            bar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")
            for step, (blur_batch, clear_batch, _) in enumerate(bar, start=1):
                blur_batch = blur_batch.to(device, non_blocking=True); clear_batch = clear_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    pred = model(blur_batch)
                    loss, _, _, _ = loss_terms(pred, clear_batch, args.edge_weight, args.freq_weight)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update(); running += loss.item()
                bar.set_postfix(loss=f"{loss.item():.4f}")
                if args.max_train_steps and step >= args.max_train_steps:
                    break
            scheduler.step()
            values, quality = validate(
                model, val_loader, device, amp_dtype, args.edge_weight, args.freq_weight,
                validation_metrics, max_batches=args.max_val_batches,
            )
            writer.writerow([epoch, *values.tolist(), quality["PSNR_Y"], quality["SSIM_Y"], quality["LPIPS"], quality["Score"], optimizer.param_groups[0]["lr"]]); handle.flush()
            payload = {"model": model.state_dict(), "epoch": epoch, "metrics": quality,
                       "best_psnr": max(best_psnr, quality["PSNR_Y"]), "best_score": max(best_score, quality["Score"]), "config": config}
            torch.save(payload, output / "model_latest.pth")
            if quality["PSNR_Y"] > best_psnr:
                best_psnr = quality["PSNR_Y"]; torch.save(payload, output / "model_best_psnr.pth")
            if quality["Score"] > best_score:
                best_score = quality["Score"]; torch.save(payload, output / "model_best_score.pth"); torch.save(payload, output / "model_best.pth")
            print(json.dumps({"epoch": epoch, "val_loss": values[0], **quality,
                              "best_psnr": best_psnr, "best_score": best_score}))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def infer(args) -> None:
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = BlurClearModel(config["kind"], config["base_dim"], config["blocks"], bool(config["frequency"]), config["max_residual"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device); model.to(device).eval()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    source = image_map(Path(args.input_dir)); names = sorted(source)
    nested = [name for name in names if Path(name).name != name]
    if nested:
        raise ValueError(f"Inference input must contain flat PNGs; first nested path: {nested[0]}")
    if args.expected_count and len(names) != args.expected_count: raise ValueError(f"Expected {args.expected_count}, got {len(names)}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.png")) and not args.overwrite: raise FileExistsError(f"Output has PNGs: {output}")
    for name in tqdm(names, desc="Blur->Clear refinement"):
        image = load_rgb(source[name]).unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            refined = model(image).float().clamp(0, 1)
        result = image.float() + args.strength * (refined - image.float())
        save_rgb(result[0], output / Path(name).name)
    archive = Path(args.archive_path); archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names: zf.write(output / Path(name).name, arcname=Path(name).name)
    manifest = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(Path(args.checkpoint)),
                "init_jit_checkpoint": config.get("init_jit_checkpoint", ""),
                "init_jit_checkpoint_sha256": config.get("init_jit_checkpoint_sha256", ""),
                "init_state_key": config.get("init_state_key", ""), "input_dir": str(Path(args.input_dir).resolve()),
                "count": len(names), "strength": args.strength, "archive": str(archive.resolve()), "archive_sha256": sha256(archive)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"); print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train"); tr.set_defaults(func=train)
    tr.add_argument("--kind", choices=["background", "detail"], required=True); tr.add_argument("--data-root", required=True)
    tr.add_argument("--blur-dir", default="Blur"); tr.add_argument("--clear-dir", default="Clear"); tr.add_argument("--output-dir", required=True)
    tr.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX); tr.add_argument("--val-ratio", type=float, default=.1)
    tr.add_argument("--patch-size", type=int, default=256); tr.add_argument("--batch-size", type=int, default=8); tr.add_argument("--val-batch-size", type=int, default=8)
    tr.add_argument("--epochs", type=int, default=12); tr.add_argument("--lr", type=float, default=1e-4); tr.add_argument("--min-lr", type=float, default=1e-6)
    tr.add_argument("--weight-decay", type=float, default=1e-4); tr.add_argument("--edge-weight", type=float, default=.05); tr.add_argument("--freq-weight", type=float, default=.01)
    tr.add_argument("--base-dim", type=int, default=32); tr.add_argument("--blocks", type=int, default=2); tr.add_argument("--frequency", type=int, choices=[0,1], default=1)
    tr.add_argument("--max-residual", type=float, default=.2); tr.add_argument("--num-workers", type=int, default=8); tr.add_argument("--seed", type=int, default=1234)
    tr.add_argument("--device", default="cuda:0"); tr.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16"); tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--max-train-steps", type=int, default=0, help="Smoke test limiter; 0 uses the full epoch")
    tr.add_argument("--max-val-batches", type=int, default=0, help="Smoke test limiter; 0 uses full validation")
    tr.add_argument("--init-jit-checkpoint", default="", help="Initialize detail refiner from a JiT checkpoint")
    tr.add_argument("--init-state-key", choices=["model", "model_ema1", "model_ema2"], default="model_ema1")
    inf = sub.add_parser("infer"); inf.set_defaults(func=infer)
    inf.add_argument("--checkpoint", required=True); inf.add_argument("--input-dir", required=True); inf.add_argument("--output-dir", required=True); inf.add_argument("--archive-path", required=True)
    inf.add_argument("--strength", type=float, default=1.0); inf.add_argument("--expected-count", type=int, default=0); inf.add_argument("--device", default="cuda:0")
    inf.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16"); inf.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if hasattr(args, "strength") and not 0 <= args.strength <= 1: parser.error("--strength must be in [0,1]")
    args.func(args)


if __name__ == "__main__": main()
