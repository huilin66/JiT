import argparse
import csv
import json
import re
import shutil
import time
import zipfile
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from denoiser import Denoiser
from main_jit import get_args_parser


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
HISTORY_FIELDS = [
    "timestamp",
    "model_name",
    "archive_name",
    "checkpoint",
    "state_key",
    "architecture",
    "use_scene",
    "use_bg_subnet",
    "steps",
    "patch_size",
    "stride",
    "tile_batch_size",
    "input_dir",
    "num_images",
    "runtime_seconds",
    "psnr_y",
    "ssim_y",
    "lpips",
    "score",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser("Generate a JiT raindrop-removal submission")
    parser.add_argument("--input-dir", required=True, help="Test Drop directory; nested images are supported")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth file or directory")
    parser.add_argument("--output-root", default="run/submissions")
    parser.add_argument("--history-csv", default="", help="Default: output-root/submission_history.csv")
    parser.add_argument("--model-name", default="", help="Used in model_timestamp.zip")
    parser.add_argument("--model", default="", help="Default: read from checkpoint, otherwise JiT-B/16")
    parser.add_argument("--state-key", default="auto", choices=["auto", "model_ema1", "model_ema2", "model"])
    parser.add_argument("--use-bg-subnet", default="auto", choices=["auto", "0", "1"])
    parser.add_argument("--use-scene", action="store_true")
    parser.add_argument("--scene-json", default="", help="Required for --use-scene; maps filename to class id")
    parser.add_argument("--class-num", type=int, default=0, help="0 infers from checkpoint")
    parser.add_argument("--img-size", type=int, default=0, help="0 infers from checkpoint, otherwise 256")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--tile-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--submission-info", default="readme.txt", help="Optional file added to ZIP; empty disables it")
    parser.add_argument("--notes", default="")
    parser.add_argument("--remove-images-after-zip", action="store_true")
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    for name in ("checkpoint-best.pth", "checkpoint-last.pth"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint-best.pth or checkpoint-last.pth under {path}")


def choose_state_dict(checkpoint, requested_key):
    if requested_key != "auto":
        if requested_key not in checkpoint:
            raise KeyError(f"Checkpoint has no state key: {requested_key}")
        return requested_key, checkpoint[requested_key]
    for key in ("model_ema1", "model_ema2", "model"):
        if key in checkpoint:
            return key, checkpoint[key]
    raise KeyError("Checkpoint contains none of model_ema1, model_ema2, or model")


def checkpoint_arg(checkpoint, name, default=None):
    saved_args = checkpoint.get("args")
    return getattr(saved_args, name, default) if saved_args is not None else default


def infer_class_num(state_dict, fallback):
    for key, value in state_dict.items():
        if key.endswith("y_embedder.embedding_table.weight"):
            return int(value.shape[0] - 1)
    return fallback


def infer_has_bg_subnet(state_dict):
    return any("bg_subnet" in key for key in state_dict)


def sanitize_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("._-") or "jit"


def default_model_name(checkpoint_path):
    parent = checkpoint_path.parent
    if parent.name.isdigit() and parent.parent.name:
        return parent.parent.name
    return parent.name or checkpoint_path.stem


def list_input_images(input_dir):
    files = sorted(
        path for path in Path(input_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No test images found under: {input_dir}")
    output_names = [f"{path.stem}.png" for path in files]
    duplicates = sorted(name for name, count in Counter(output_names).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate output names cannot be flattened into submission ZIP: {duplicates[0]}")
    return files


def load_scene_labels(path, image_files, class_num):
    if not path:
        raise ValueError("--scene-json is required when --use-scene is enabled")
    with open(path, "r", encoding="utf-8") as file:
        labels = json.load(file)
    missing = [image.name for image in image_files if image.name not in labels]
    if missing:
        raise KeyError(f"Scene JSON misses {len(missing)} test images; first missing: {missing[0]}")
    invalid = [name for name in (image.name for image in image_files) if not 0 <= int(labels[name]) < class_num]
    if invalid:
        raise ValueError(f"Invalid scene label for: {invalid[0]}")
    return {name: int(value) for name, value in labels.items()}


def get_tile_coords(full_size, patch_size, stride):
    if full_size <= patch_size:
        return [0]
    coords = list(range(0, full_size - patch_size + 1, stride))
    if coords[-1] != full_size - patch_size:
        coords.append(full_size - patch_size)
    return coords


def pad_to_patch(image, patch_size):
    _, _, height, width = image.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if not pad_h and not pad_w:
        return image, (height, width)
    mode = "reflect" if height > pad_h and width > pad_w and height > 1 and width > 1 else "replicate"
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), (height, width)


def blend_window(patch_size):
    window_1d = torch.hann_window(patch_size, periodic=False).clamp_min(1e-3)
    window = window_1d[:, None] * window_1d[None, :]
    return (window / window.max()).view(1, 1, patch_size, patch_size)


def autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype == "fp32":
        return nullcontext()
    if amp_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def restore_image(model, image_path, device, patch_size, stride, tile_batch_size, steps, scene_id, amp_dtype):
    image = Image.open(image_path).convert("RGB")
    array = np.array(image, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).div_(255.0)
    tensor = tensor.mul_(2.0).sub_(1.0)
    tensor, original_size = pad_to_patch(tensor, patch_size)
    _, _, height, width = tensor.shape

    x_coords = get_tile_coords(width, patch_size, stride)
    y_coords = get_tile_coords(height, patch_size, stride)
    coords = [(x, y) for y in y_coords for x in x_coords]
    window = blend_window(patch_size)
    output = torch.zeros((1, 3, height, width), dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), dtype=torch.float32)

    for start in range(0, len(coords), tile_batch_size):
        chunk_coords = coords[start:start + tile_batch_size]
        patches = torch.cat(
            [tensor[:, :, y:y + patch_size, x:x + patch_size] for x, y in chunk_coords],
            dim=0,
        ).to(device, non_blocking=True)
        labels = None
        if scene_id is not None:
            labels = torch.full((patches.shape[0],), scene_id, dtype=torch.long, device=device)
        with autocast_context(device, amp_dtype):
            predictions = model.generate_i2i(patches, steps=steps, dummy_labels=labels)
        predictions = predictions.float().cpu()
        for index, (x, y) in enumerate(chunk_coords):
            output[:, :, y:y + patch_size, x:x + patch_size] += predictions[index:index + 1] * window
            weights[:, :, y:y + patch_size, x:x + patch_size] += window

    output = output.div_(weights.clamp_min_(1e-8)).clamp_(-1.0, 1.0)
    original_h, original_w = original_size
    output = output[:, :, :original_h, :original_w]
    output = output.add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    return Image.fromarray(output[0].permute(1, 2, 0).numpy(), mode="RGB")


def create_archive(image_dir, archive_path, submission_info):
    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        raise RuntimeError(f"No prediction PNG files under: {image_dir}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in image_files:
            archive.write(image_path, arcname=image_path.name)
        if submission_info:
            info_path = Path(submission_info)
            if info_path.exists():
                archive.write(info_path, arcname=info_path.name)
            else:
                print(f"Warning: submission info file not found, skipped: {info_path}")


def append_history(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with open(path, "r", newline="", encoding="utf-8-sig") as file:
            header = next(csv.reader(file), [])
        if header != HISTORY_FIELDS:
            raise RuntimeError(f"Existing history CSV has an incompatible header: {path}")
    with open(path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    cli = parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(cli.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    state_key, state_dict = choose_state_dict(checkpoint, cli.state_key)

    architecture = cli.model or checkpoint_arg(checkpoint, "model", "JiT-B/16")
    img_size = cli.img_size or int(checkpoint_arg(checkpoint, "img_size", 256))
    class_num = cli.class_num or infer_class_num(state_dict, int(checkpoint_arg(checkpoint, "class_num", 1000)))
    has_bg_subnet = infer_has_bg_subnet(state_dict) if cli.use_bg_subnet == "auto" else cli.use_bg_subnet == "1"
    if cli.stride <= 0 or cli.stride > img_size:
        raise ValueError(f"stride must be in [1, {img_size}]")

    image_files = list_input_images(cli.input_dir)
    scene_labels = load_scene_labels(cli.scene_json, image_files, class_num) if cli.use_scene else None

    model_args = get_args_parser().parse_args([])
    model_args.model = architecture
    model_args.img_size = img_size
    model_args.class_num = class_num
    model_args.use_bg_subnet = int(has_bg_subnet)
    model_args.num_sampling_steps = cli.steps
    model_args.cfg = 1.0
    model = Denoiser(model_args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = sanitize_name(cli.model_name or default_model_name(checkpoint_path))
    run_name = f"{model_name}_{timestamp}"
    output_root = Path(cli.output_root)
    image_dir = output_root / run_name
    archive_path = output_root / f"{run_name}.zip"
    history_path = Path(cli.history_csv) if cli.history_csv else output_root / "submission_history.csv"
    image_dir.mkdir(parents=True, exist_ok=False)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"State: {state_key}; model: {architecture}; scene: {cli.use_scene}; head: {has_bg_subnet}")
    print(f"Images: {len(image_files)}; device: {device}; output: {archive_path}")
    started = time.perf_counter()
    for image_path in tqdm(image_files, desc="JiT submission inference"):
        scene_id = scene_labels[image_path.name] if scene_labels is not None else None
        prediction = restore_image(
            model=model,
            image_path=image_path,
            device=device,
            patch_size=img_size,
            stride=cli.stride,
            tile_batch_size=cli.tile_batch_size,
            steps=cli.steps,
            scene_id=scene_id,
            amp_dtype=cli.amp_dtype,
        )
        prediction.save(image_dir / f"{image_path.stem}.png", format="PNG", optimize=True)
    runtime = time.perf_counter() - started

    create_archive(image_dir, archive_path, cli.submission_info)
    append_history(
        history_path,
        {
            "timestamp": timestamp,
            "model_name": model_name,
            "archive_name": archive_path.name,
            "checkpoint": str(checkpoint_path.resolve()),
            "state_key": state_key,
            "architecture": architecture,
            "use_scene": int(cli.use_scene),
            "use_bg_subnet": int(has_bg_subnet),
            "steps": cli.steps,
            "patch_size": img_size,
            "stride": cli.stride,
            "tile_batch_size": cli.tile_batch_size,
            "input_dir": str(Path(cli.input_dir).resolve()),
            "num_images": len(image_files),
            "runtime_seconds": round(runtime, 3),
            "psnr_y": "",
            "ssim_y": "",
            "lpips": "",
            "score": "",
            "notes": cli.notes,
        },
    )
    if cli.remove_images_after_zip:
        shutil.rmtree(image_dir)
    print(f"Submission: {archive_path}")
    print(f"History CSV: {history_path}")
    print(f"Runtime: {runtime:.2f}s ({runtime / len(image_files):.3f}s/image)")


if __name__ == "__main__":
    main()
