import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
import shutil
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_images(root, recursive=False):
    root = Path(root)
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in iterator if p.is_file() and is_image(p))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_threaded(items, worker, desc, num_workers=1):
    num_workers = int(num_workers)
    if num_workers <= 1:
        return [worker(item) for item in tqdm(items, desc=desc)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(
            tqdm(
                executor.map(worker, items),
                total=len(items),
                desc=f"{desc} ({num_workers} threads)",
            )
        )


def infer_day_from_name(img_name):
    lower = img_name.lower()
    if lower.startswith("day_") or lower.startswith("d_"):
        return True
    if lower.startswith("night_") or lower.startswith("n_"):
        return False
    return None


def class_id_from_flags(is_day, is_bg_focus):
    # 0: night background-focus
    # 1: night raindrop-focus
    # 2: day background-focus
    # 3: day raindrop-focus
    if not is_day and is_bg_focus:
        return 0
    if not is_day and not is_bg_focus:
        return 1
    if is_day and is_bg_focus:
        return 2
    return 3


def robust_mean(values, low=5.0, high=95.0):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    lo, hi = np.percentile(values, [low, high])
    trimmed = values[(values >= lo) & (values <= hi)]
    if trimmed.size == 0:
        trimmed = values
    return float(np.mean(trimmed))


def image_sharpness_map(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return grad_x * grad_x + grad_y * grad_y


def build_residual_mask(
    drop_img,
    clear_img,
    residual_percentile=90.0,
    residual_min=8.0,
    min_mask_ratio=0.005,
    max_mask_ratio=0.55,
    morph_kernel=5,
):
    diff = cv2.absdiff(drop_img, clear_img).astype(np.float32).mean(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    threshold = max(float(np.percentile(diff, residual_percentile)), residual_min)
    mask = diff >= threshold

    ratio = float(mask.mean())
    if ratio < min_mask_ratio:
        fallback_percentile = max(50.0, residual_percentile - 10.0)
        threshold = max(float(np.percentile(diff, fallback_percentile)), residual_min * 0.5)
        mask = diff >= threshold
        ratio = float(mask.mean())

    if ratio < min_mask_ratio:
        flat = diff.reshape(-1)
        keep = max(1, int(flat.size * min_mask_ratio))
        threshold = float(np.partition(flat, flat.size - keep)[flat.size - keep])
        mask = diff >= threshold
    elif ratio > max_mask_ratio:
        threshold = max(float(np.percentile(diff, 95.0)), residual_min)
        mask = diff >= threshold

    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask = mask_u8 > 0

    return mask, diff, threshold


def predict_focus_scene_by_pair(
    drop_path,
    clear_path,
    scene_count=4,
    focus_threshold=0.0,
    intensity_thresh=100.0,
    residual_percentile=90.0,
    residual_min=8.0,
    min_mask_ratio=0.005,
    max_mask_ratio=0.55,
    morph_kernel=5,
    bg_dilate=9,
    prefer_name_day=True,
):
    drop_img = cv2.imread(str(drop_path), cv2.IMREAD_COLOR)
    clear_img = cv2.imread(str(clear_path), cv2.IMREAD_COLOR)
    if drop_img is None or clear_img is None:
        return None

    resized_clear = 0
    if drop_img.shape[:2] != clear_img.shape[:2]:
        clear_img = cv2.resize(clear_img, (drop_img.shape[1], drop_img.shape[0]), interpolation=cv2.INTER_AREA)
        resized_clear = 1

    mask, residual_map, residual_threshold = build_residual_mask(
        drop_img=drop_img,
        clear_img=clear_img,
        residual_percentile=residual_percentile,
        residual_min=residual_min,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        morph_kernel=morph_kernel,
    )
    if bg_dilate > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilate, bg_dilate))
        bg_exclusion = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        bg_mask = ~bg_exclusion
    else:
        bg_mask = ~mask

    if not np.any(mask) or not np.any(bg_mask):
        return None

    sharpness = image_sharpness_map(drop_img)
    sharp_drop = robust_mean(sharpness[mask])
    sharp_bg = robust_mean(sharpness[bg_mask])
    focus_score = float(np.log(sharp_drop + 1e-6) - np.log(sharp_bg + 1e-6))
    is_raindrop_focus = focus_score > focus_threshold
    is_bg_focus = not is_raindrop_focus

    mean_intensity = float(np.mean(cv2.cvtColor(drop_img, cv2.COLOR_BGR2GRAY)))
    name_day = infer_day_from_name(Path(drop_path).name) if prefer_name_day else None
    is_day = name_day if name_day is not None else mean_intensity > intensity_thresh

    if scene_count == 2:
        class_id = int(is_raindrop_focus)
    elif scene_count == 4:
        class_id = class_id_from_flags(is_day, is_bg_focus)
    else:
        raise ValueError(f"scene_count must be 2 or 4, got {scene_count}")

    return {
        "filename": Path(drop_path).name,
        "class_id": int(class_id),
        "scene_count": int(scene_count),
        "is_day": int(is_day),
        "is_bg_focus": int(is_bg_focus),
        "is_raindrop_focus": int(is_raindrop_focus),
        "focus_score": round(focus_score, 6),
        "sharp_drop": round(sharp_drop, 6),
        "sharp_bg": round(sharp_bg, 6),
        "mask_ratio": round(float(mask.mean()), 6),
        "residual_mean": round(float(np.mean(residual_map)), 6),
        "residual_threshold": round(float(residual_threshold), 6),
        "mean_intensity": round(mean_intensity, 4),
        "resized_clear": resized_clear,
    }


def generate_focus_scene_labels(
    data_root,
    output_json,
    output_csv=None,
    scene_count=4,
    drop_dir_name="Drop",
    clear_dir_name="Clear",
    focus_threshold=0.0,
    intensity_thresh=100.0,
    residual_percentile=90.0,
    residual_min=8.0,
    min_mask_ratio=0.005,
    max_mask_ratio=0.55,
    morph_kernel=5,
    bg_dilate=9,
    prefer_name_day=True,
    num_workers=1,
):
    data_root = Path(data_root)
    drop_dir = data_root / drop_dir_name
    clear_dir = data_root / clear_dir_name
    output_json = Path(output_json)
    output_csv = Path(output_csv) if output_csv else output_json.with_suffix(".csv")
    ensure_dir(output_json.parent)
    ensure_dir(output_csv.parent)

    files = list_images(drop_dir, recursive=False)
    if not files:
        raise RuntimeError(f"No images found in Drop directory: {drop_dir}")

    rows = []
    scene_dict = {}
    failed = []
    missing_clear = []

    def process_one(drop_path):
        clear_path = clear_dir / drop_path.name
        if not clear_path.exists():
            return "missing_clear", drop_path.name, None
        row = predict_focus_scene_by_pair(
            drop_path=drop_path,
            clear_path=clear_path,
            scene_count=scene_count,
            focus_threshold=focus_threshold,
            intensity_thresh=intensity_thresh,
            residual_percentile=residual_percentile,
            residual_min=residual_min,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            morph_kernel=morph_kernel,
            bg_dilate=bg_dilate,
            prefer_name_day=prefer_name_day,
        )
        if row is None:
            return "failed", drop_path.name, None
        return "ok", drop_path.name, row

    results = run_threaded(
        files,
        process_one,
        desc=f"Generate {scene_count}scene focus labels",
        num_workers=num_workers,
    )
    for status, name, row in results:
        if status == "missing_clear":
            missing_clear.append(name)
            continue
        if status == "failed":
            failed.append(name)
            continue
        rows.append(row)
        scene_dict[row["filename"]] = row["class_id"]

    if missing_clear:
        raise RuntimeError(f"Missing {len(missing_clear)} clear pairs; first: {missing_clear[0]}")
    if failed:
        raise RuntimeError(f"Failed to label {len(failed)} images; first: {failed[0]}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scene_dict, f, indent=2, sort_keys=True)

    fieldnames = [
        "filename",
        "class_id",
        "scene_count",
        "is_day",
        "is_bg_focus",
        "is_raindrop_focus",
        "focus_score",
        "sharp_drop",
        "sharp_bg",
        "mask_ratio",
        "residual_mean",
        "residual_threshold",
        "mean_intensity",
        "resized_clear",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {i: Counter(scene_dict.values()).get(i, 0) for i in range(scene_count)}
    print(f"Saved focus scene json: {output_json}")
    print(f"Saved focus scene stats csv: {output_csv}")
    print("Class counts:", counts)
    return scene_dict


def analyze_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return mean_intensity, laplacian_var


def predict_scene_by_heuristics(
    img_path,
    intensity_thresh=100.0,
    laplacian_thresh=500.0,
    prefer_name_day=True,
):
    stats = analyze_image(img_path)
    if stats is None:
        return None

    mean_intensity, laplacian_var = stats
    name_day = infer_day_from_name(Path(img_path).name) if prefer_name_day else None
    is_day = name_day if name_day is not None else mean_intensity > intensity_thresh
    is_bg_focus = laplacian_var > laplacian_thresh
    class_id = class_id_from_flags(is_day, is_bg_focus)
    return {
        "filename": Path(img_path).name,
        "mean_intensity": round(mean_intensity, 4),
        "laplacian_var": round(laplacian_var, 4),
        "is_day": int(is_day),
        "is_bg_focus": int(is_bg_focus),
        "class_id": int(class_id),
    }


def generate_scene_pseudo_labels(
    drop_dir,
    output_json,
    output_csv=None,
    intensity_thresh=100.0,
    laplacian_thresh=500.0,
    prefer_name_day=True,
    num_workers=1,
):
    drop_dir = Path(drop_dir)
    output_json = Path(output_json)
    output_csv = Path(output_csv) if output_csv else output_json.with_suffix(".csv")
    ensure_dir(output_json.parent)
    ensure_dir(output_csv.parent)

    files = list_images(drop_dir, recursive=False)
    if not files:
        raise RuntimeError(f"No images found in Drop directory: {drop_dir}")

    rows = []
    scene_dict = {}
    failed = []

    worker = partial(
        predict_scene_by_heuristics,
        intensity_thresh=intensity_thresh,
        laplacian_thresh=laplacian_thresh,
        prefer_name_day=prefer_name_day,
    )
    results = run_threaded(files, worker, desc="Generate scene pseudo labels", num_workers=num_workers)
    for img_path, row in zip(files, results):
        if row is None:
            failed.append(img_path.name)
            continue
        rows.append(row)
        scene_dict[row["filename"]] = row["class_id"]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scene_dict, f, indent=2, sort_keys=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "mean_intensity",
                "laplacian_var",
                "is_day",
                "is_bg_focus",
                "class_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {i: 0 for i in range(4)}
    for class_id in scene_dict.values():
        counts[int(class_id)] += 1

    print(f"Saved scene json: {output_json}")
    print(f"Saved scene stats csv: {output_csv}")
    print("Class counts:", counts)
    if failed:
        print(f"Warning: failed to read {len(failed)} images. First failed: {failed[0]}")
    return scene_dict


def predict_day_night_label(img_path, intensity_thresh=100.0, prefer_name_day=True):
    stats = analyze_image(img_path)
    if stats is None:
        return None

    mean_intensity, _ = stats
    name_day = infer_day_from_name(Path(img_path).name) if prefer_name_day else None
    is_day = name_day if name_day is not None else mean_intensity > intensity_thresh
    return {
        "filename": Path(img_path).name,
        "class_id": int(is_day),
        "is_day": int(is_day),
        "mean_intensity": round(mean_intensity, 4),
        "label_source": "name" if name_day is not None else "brightness",
    }


def generate_day_night_scene_labels(
    drop_dir,
    output_json,
    output_csv=None,
    intensity_thresh=100.0,
    prefer_name_day=True,
    num_workers=1,
):
    drop_dir = Path(drop_dir)
    output_json = Path(output_json)
    output_csv = Path(output_csv) if output_csv else output_json.with_suffix(".csv")
    ensure_dir(output_json.parent)
    ensure_dir(output_csv.parent)

    files = list_images(drop_dir, recursive=False)
    if not files:
        raise RuntimeError(f"No images found in Drop directory: {drop_dir}")

    rows = []
    scene_dict = {}
    failed = []

    worker = partial(
        predict_day_night_label,
        intensity_thresh=intensity_thresh,
        prefer_name_day=prefer_name_day,
    )
    results = run_threaded(files, worker, desc="Generate day/night 2scene labels", num_workers=num_workers)
    for img_path, row in zip(files, results):
        if row is None:
            failed.append(img_path.name)
            continue
        rows.append(row)
        scene_dict[row["filename"]] = row["class_id"]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scene_dict, f, indent=2, sort_keys=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "class_id",
                "is_day",
                "mean_intensity",
                "label_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {0: Counter(scene_dict.values()).get(0, 0), 1: Counter(scene_dict.values()).get(1, 0)}
    print(f"Saved day/night scene json: {output_json}")
    print(f"Saved day/night scene stats csv: {output_csv}")
    print("Class counts:", counts)
    if failed:
        print(f"Warning: failed to read {len(failed)} images. First failed: {failed[0]}")
    return scene_dict


def copy_data(src_dir, dst_dir, drop_dir_name="Drop", clear_dir_name="Clear", type_name="Day"):
    dst_dir = Path(dst_dir)
    dst_dir_drop = dst_dir / drop_dir_name
    dst_dir_clear = dst_dir / clear_dir_name
    ensure_dir(dst_dir_drop)
    ensure_dir(dst_dir_clear)

    src_dir = Path(src_dir)
    src_dir_drop = src_dir / drop_dir_name
    src_dir_clear = src_dir / clear_dir_name

    if not src_dir_drop.exists() or not src_dir_clear.exists():
        raise RuntimeError(f"Expected {src_dir_drop} and {src_dir_clear} to exist.")

    for src_root, dst_root, desc in [
        (src_dir_drop, dst_dir_drop, f"Copy {type_name} Drop"),
        (src_dir_clear, dst_dir_clear, f"Copy {type_name} Clear"),
    ]:
        sub_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
        for sub_dir in tqdm(sub_dirs, desc=desc):
            for src_file in list_images(sub_dir, recursive=False):
                dst_file = dst_root / f"{type_name}_{sub_dir.name}_{src_file.name}"
                shutil.copy2(src_file, dst_file)


def copy_day_night(day_root, night_root, dst_root, drop_dir_name="Drop", clear_dir_name="Clear"):
    if day_root:
        copy_data(day_root, dst_root, drop_dir_name, clear_dir_name, type_name="Day")
    if night_root:
        copy_data(night_root, dst_root, drop_dir_name, clear_dir_name, type_name="Night")


def check_trainable_folder(data_root, scene_json=None, drop_dir_name="Drop", clear_dir_name="Clear"):
    data_root = Path(data_root)
    drop_dir = data_root / drop_dir_name
    clear_dir = data_root / clear_dir_name
    if not drop_dir.exists():
        raise RuntimeError(f"Missing Drop directory: {drop_dir}")
    if not clear_dir.exists():
        raise RuntimeError(f"Missing Clear directory: {clear_dir}")

    drop_files = {p.name for p in list_images(drop_dir, recursive=False)}
    clear_files = {p.name for p in list_images(clear_dir, recursive=False)}
    missing_clear = sorted(drop_files - clear_files)
    missing_drop = sorted(clear_files - drop_files)

    print(f"Drop images: {len(drop_files)}")
    print(f"Clear images: {len(clear_files)}")
    print(f"Missing Clear pairs: {len(missing_clear)}")
    print(f"Missing Drop pairs: {len(missing_drop)}")
    if missing_clear:
        print(f"First missing Clear: {missing_clear[0]}")
    if missing_drop:
        print(f"First missing Drop: {missing_drop[0]}")

    scene_json = Path(scene_json) if scene_json else data_root / "Drop_scen_pred.json"
    if scene_json.exists():
        with open(scene_json, "r", encoding="utf-8") as f:
            scene_dict = json.load(f)
        missing_scene = sorted(drop_files - set(scene_dict.keys()))
        extra_scene = sorted(set(scene_dict.keys()) - drop_files)
        max_class = max([int(value) for value in scene_dict.values()] or [3])
        counts = {i: 0 for i in range(max(4, max_class + 1))}
        for value in scene_dict.values():
            if int(value) in counts:
                counts[int(value)] += 1
        print(f"Scene json: {scene_json}")
        print(f"Scene labels: {len(scene_dict)}")
        print(f"Missing scene labels: {len(missing_scene)}")
        print(f"Extra scene labels: {len(extra_scene)}")
        print("Scene class counts:", counts)
        if missing_scene:
            print(f"First missing scene label: {missing_scene[0]}")
    else:
        print(f"Scene json not found: {scene_json}")


def count_dataset_images(base_dir):
    base_dir = Path(base_dir)
    total_images = 0
    folder_counts = {}

    for folder in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        count = len(list_images(folder, recursive=False))
        folder_counts[folder.name] = count
        total_images += count
        print(f"{folder.name:<20} | {count}")

    print("-" * 35)
    print(f"Folders: {len(folder_counts)}")
    print(f"Images: {total_images}")


def extract_sample_images(source_dir, dest_dir):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)

    count = 0
    folders = sorted(p for p in source_dir.iterdir() if p.is_dir())
    for folder in tqdm(folders, desc="Extract sample images"):
        files = list_images(folder, recursive=False)
        if not files:
            continue
        src_path = files[0]
        dst_path = dest_dir / f"{folder.name}{src_path.suffix}"
        shutil.copy2(src_path, dst_path)
        count += 1
    print(f"Extracted {count} sample images to {dest_dir}")


def convert_manual_labels(input_excel, output_csv):
    input_excel = Path(input_excel)
    output_csv = Path(output_csv)
    if not input_excel.exists():
        raise RuntimeError(f"Input file not found: {input_excel}")

    df = pd.read_excel(input_excel, sheet_name="Sheet1")
    df["folder_name"] = df["id"].apply(lambda x: str(int(x)).zfill(5))

    def get_class_id(row):
        is_day = str(row["type"]).strip().upper() == "D"
        is_bg_focus = str(row["bg focus"]).strip() == "1"
        return class_id_from_flags(is_day, is_bg_focus)

    df["class_id"] = df.apply(get_class_id, axis=1)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved manual label csv: {output_csv}")
    print(df["class_id"].value_counts().sort_index())


def build_parser():
    parser = argparse.ArgumentParser("Raindrop data tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    copy_parser = subparsers.add_parser("copy", help="Merge Day/Night data into flat Drop/Clear folders")
    copy_parser.add_argument("--day-root", default="", help="Root containing Day Drop/Clear folders")
    copy_parser.add_argument("--night-root", default="", help="Root containing Night Drop/Clear folders")
    copy_parser.add_argument("--dst-root", required=True, help="Output trainable data root")
    copy_parser.add_argument("--drop-dir-name", default="Drop")
    copy_parser.add_argument("--clear-dir-name", default="Clear")

    pseudo_parser = subparsers.add_parser(
        "pseudo-scene",
        help="Generate Drop_scen_pred.json for ScenePairedRainDatasetV2",
    )
    pseudo_parser.add_argument("--data-root", default="", help="Root containing Drop/Clear folders")
    pseudo_parser.add_argument("--drop-dir", default="", help="Drop directory. Overrides --data-root/Drop")
    pseudo_parser.add_argument("--output-json", default="", help="Default: data-root/Drop_scen_pred.json")
    pseudo_parser.add_argument("--output-csv", default="", help="Default: same path as json with .csv suffix")
    pseudo_parser.add_argument("--intensity-thresh", type=float, default=100.0)
    pseudo_parser.add_argument("--laplacian-thresh", type=float, default=500.0)
    pseudo_parser.add_argument("--num-workers", type=int, default=1, help="Thread workers for image analysis")
    pseudo_parser.add_argument(
        "--ignore-name-day",
        action="store_true",
        help="Infer day/night from brightness even if filename starts with Day/Night",
    )

    focus_parser = subparsers.add_parser(
        "pseudo-focus-scene",
        help="Generate 2scene/4scene labels from Drop/Clear residual masks and local sharpness",
    )
    focus_parser.add_argument("--data-root", required=True, help="Root containing Drop/Clear folders")
    focus_parser.add_argument("--output-json", default="", help="Default: data-root/Drop_focus_{2,4}scene.json")
    focus_parser.add_argument("--output-csv", default="", help="Default: same path as json with .csv suffix")
    focus_parser.add_argument("--scene-count", type=int, choices=[2, 4], default=4)
    focus_parser.add_argument("--drop-dir-name", default="Drop")
    focus_parser.add_argument("--clear-dir-name", default="Clear")
    focus_parser.add_argument("--focus-threshold", type=float, default=0.0)
    focus_parser.add_argument("--intensity-thresh", type=float, default=100.0)
    focus_parser.add_argument("--residual-percentile", type=float, default=90.0)
    focus_parser.add_argument("--residual-min", type=float, default=8.0)
    focus_parser.add_argument("--min-mask-ratio", type=float, default=0.005)
    focus_parser.add_argument("--max-mask-ratio", type=float, default=0.55)
    focus_parser.add_argument("--morph-kernel", type=int, default=5)
    focus_parser.add_argument("--bg-dilate", type=int, default=9)
    focus_parser.add_argument("--num-workers", type=int, default=1, help="Thread workers for image analysis")
    focus_parser.add_argument(
        "--ignore-name-day",
        action="store_true",
        help="Infer day/night from brightness even if filename starts with Day/Night",
    )

    dn_parser = subparsers.add_parser(
        "pseudo-day-night-scene",
        help="Generate Drop_dn_2scene.json labels: 0=night, 1=day",
    )
    dn_parser.add_argument("--data-root", default="", help="Root containing Drop/Clear folders")
    dn_parser.add_argument("--drop-dir", default="", help="Drop directory. Overrides --data-root/Drop")
    dn_parser.add_argument("--output-json", default="", help="Default: data-root/Drop_dn_2scene.json")
    dn_parser.add_argument("--output-csv", default="", help="Default: same path as json with .csv suffix")
    dn_parser.add_argument("--intensity-thresh", type=float, default=100.0)
    dn_parser.add_argument("--num-workers", type=int, default=1, help="Thread workers for image analysis")
    dn_parser.add_argument(
        "--ignore-name-day",
        action="store_true",
        help="Infer day/night from brightness even if filename starts with Day/Night",
    )

    check_parser = subparsers.add_parser("check", help="Check Drop/Clear pairs and scene json coverage")
    check_parser.add_argument("--data-root", required=True)
    check_parser.add_argument("--scene-json", default="")
    check_parser.add_argument("--drop-dir-name", default="Drop")
    check_parser.add_argument("--clear-dir-name", default="Clear")

    count_parser = subparsers.add_parser("count", help="Count images under subfolders")
    count_parser.add_argument("--base-dir", required=True)

    sample_parser = subparsers.add_parser("samples", help="Extract one sample image per subfolder")
    sample_parser.add_argument("--source-dir", required=True)
    sample_parser.add_argument("--dest-dir", required=True)

    manual_parser = subparsers.add_parser("manual-labels", help="Convert manual Excel labels to csv")
    manual_parser.add_argument("--input-excel", required=True)
    manual_parser.add_argument("--output-csv", required=True)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "copy":
        copy_day_night(
            day_root=args.day_root,
            night_root=args.night_root,
            dst_root=args.dst_root,
            drop_dir_name=args.drop_dir_name,
            clear_dir_name=args.clear_dir_name,
        )
        check_trainable_folder(args.dst_root, drop_dir_name=args.drop_dir_name, clear_dir_name=args.clear_dir_name)
        return

    if args.command == "pseudo-scene":
        if args.drop_dir:
            drop_dir = Path(args.drop_dir)
            data_root = drop_dir.parent
        elif args.data_root:
            data_root = Path(args.data_root)
            drop_dir = data_root / "Drop"
        else:
            raise RuntimeError("pseudo-scene needs --data-root or --drop-dir")

        output_json = Path(args.output_json) if args.output_json else data_root / "Drop_scen_pred.json"
        output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
        generate_scene_pseudo_labels(
            drop_dir=drop_dir,
            output_json=output_json,
            output_csv=output_csv,
            intensity_thresh=args.intensity_thresh,
            laplacian_thresh=args.laplacian_thresh,
            prefer_name_day=not args.ignore_name_day,
            num_workers=args.num_workers,
        )
        check_trainable_folder(data_root, scene_json=output_json)
        return

    if args.command == "pseudo-focus-scene":
        data_root = Path(args.data_root)
        output_json = (
            Path(args.output_json)
            if args.output_json
            else data_root / f"Drop_focus_{args.scene_count}scene.json"
        )
        output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
        generate_focus_scene_labels(
            data_root=data_root,
            output_json=output_json,
            output_csv=output_csv,
            scene_count=args.scene_count,
            drop_dir_name=args.drop_dir_name,
            clear_dir_name=args.clear_dir_name,
            focus_threshold=args.focus_threshold,
            intensity_thresh=args.intensity_thresh,
            residual_percentile=args.residual_percentile,
            residual_min=args.residual_min,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            morph_kernel=args.morph_kernel,
            bg_dilate=args.bg_dilate,
            prefer_name_day=not args.ignore_name_day,
            num_workers=args.num_workers,
        )
        check_trainable_folder(data_root, scene_json=output_json)
        return

    if args.command == "pseudo-day-night-scene":
        if args.drop_dir:
            drop_dir = Path(args.drop_dir)
            data_root = drop_dir.parent
        elif args.data_root:
            data_root = Path(args.data_root)
            drop_dir = data_root / "Drop"
        else:
            raise RuntimeError("pseudo-day-night-scene needs --data-root or --drop-dir")

        output_json = Path(args.output_json) if args.output_json else data_root / "Drop_dn_2scene.json"
        output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
        generate_day_night_scene_labels(
            drop_dir=drop_dir,
            output_json=output_json,
            output_csv=output_csv,
            intensity_thresh=args.intensity_thresh,
            prefer_name_day=not args.ignore_name_day,
            num_workers=args.num_workers,
        )
        check_trainable_folder(data_root, scene_json=output_json)
        return

    if args.command == "check":
        check_trainable_folder(
            args.data_root,
            scene_json=args.scene_json or None,
            drop_dir_name=args.drop_dir_name,
            clear_dir_name=args.clear_dir_name,
        )
        return

    if args.command == "count":
        count_dataset_images(args.base_dir)
        return

    if args.command == "samples":
        extract_sample_images(args.source_dir, args.dest_dir)
        return

    if args.command == "manual-labels":
        convert_manual_labels(args.input_excel, args.output_csv)
        return


if __name__ == "__main__":
    main()
