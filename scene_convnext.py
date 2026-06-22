import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {
    0: "night_background_focus",
    1: "night_raindrop_focus",
    2: "day_background_focus",
    3: "day_raindrop_focus",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_images(root, recursive=True):
    root = Path(root)
    iterator = root.rglob("*") if recursive else root.iterdir()
    files = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No images found under: {root}")
    counts = Counter(path.name for path in files)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate image basenames are unsupported: {duplicates[0]}")
    return files


def load_scene_samples(image_dir, labels_json, recursive=True):
    image_files = list_images(image_dir, recursive=recursive)
    with open(labels_json, "r", encoding="utf-8") as file:
        labels = json.load(file)

    missing = [path.name for path in image_files if path.name not in labels]
    if missing:
        raise KeyError(f"Scene JSON misses {len(missing)} images; first missing: {missing[0]}")

    samples = []
    for path in image_files:
        label = int(labels[path.name])
        if label not in CLASS_NAMES:
            raise ValueError(f"Invalid scene class {label} for {path.name}")
        samples.append((path, label))
    return samples


def scene_group_key(path):
    parts = Path(path).stem.split("_")
    if len(parts) >= 2 and parts[0].lower() in {"day", "night", "d", "n"}:
        prefix = "day" if parts[0].lower() in {"day", "d"} else "night"
        return f"{prefix}_{parts[1]}"
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return Path(path).stem


def grouped_train_val_split(samples, val_fraction=0.1, seed=42):
    groups = {}
    for sample in samples:
        groups.setdefault(scene_group_key(sample[0]), []).append(sample)
    if len(groups) < 2:
        raise RuntimeError("At least two scene groups are required for train/validation split")

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    val_count = max(1, round(len(group_names) * val_fraction))
    val_count = min(val_count, len(group_names) - 1)
    val_groups = set(group_names[:val_count])

    train_samples = []
    val_samples = []
    for group_name, group_samples in groups.items():
        target = val_samples if group_name in val_groups else train_samples
        target.extend(group_samples)
    return train_samples, val_samples, sorted(val_groups)


def save_split_manifest(path, train_samples, val_samples, val_groups, seed):
    manifest = {
        "seed": seed,
        "val_groups": val_groups,
        "train": [sample[0].name for sample in train_samples],
        "val": [sample[0].name for sample in val_samples],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def build_transforms(image_size=224):
    resize_size = round(image_size * 256 / 224)
    train_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class SceneDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class SceneInferenceDataset(Dataset):
    def __init__(self, image_dir, transform, recursive=True):
        self.image_files = list_images(image_dir, recursive=recursive)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        path = self.image_files[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, path.name


def class_counts(samples):
    counts = Counter(label for _, label in samples)
    return {class_id: counts.get(class_id, 0) for class_id in CLASS_NAMES}


def balanced_class_weights(samples):
    counts = class_counts(samples)
    if any(count == 0 for count in counts.values()):
        missing = [class_id for class_id, count in counts.items() if count == 0]
        raise RuntimeError(f"Training split has no samples for scene classes: {missing}")
    total = sum(counts.values())
    weights = [total / (len(CLASS_NAMES) * counts[class_id]) for class_id in CLASS_NAMES]
    return torch.tensor(weights, dtype=torch.float32)
