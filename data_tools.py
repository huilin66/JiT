import argparse
import csv
import json
import os
import shutil
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

    for img_path in tqdm(files, desc="Generate scene pseudo labels"):
        row = predict_scene_by_heuristics(
            img_path,
            intensity_thresh=intensity_thresh,
            laplacian_thresh=laplacian_thresh,
            prefer_name_day=prefer_name_day,
        )
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
        counts = {i: 0 for i in range(4)}
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
    pseudo_parser.add_argument(
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
