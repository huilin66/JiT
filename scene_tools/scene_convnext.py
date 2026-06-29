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


def load_label_dict(labels_json):
    with open(labels_json, "r", encoding="utf-8") as file:
        labels = json.load(file)
    return {name: int(label) for name, label in labels.items()}


def infer_num_classes_from_labels(labels_json):
    labels = load_label_dict(labels_json)
    if not labels:
        raise RuntimeError(f"Scene JSON is empty: {labels_json}")
    min_label = min(labels.values())
    max_label = max(labels.values())
    if min_label < 0:
        raise ValueError(f"Scene labels must be non-negative, got {min_label}")
    return max_label + 1


def normalize_class_names(num_classes, class_names=None):
    if class_names is None:
        if num_classes == len(CLASS_NAMES):
            return dict(CLASS_NAMES)
        return {class_id: f"class_{class_id}" for class_id in range(num_classes)}
    if isinstance(class_names, str):
        names = [name.strip() for name in class_names.split(",") if name.strip()]
        if len(names) != num_classes:
            raise ValueError(
                f"Expected {num_classes} comma-separated class names, got {len(names)}"
            )
        return {class_id: names[class_id] for class_id in range(num_classes)}
    if isinstance(class_names, dict):
        normalized = {int(key): str(value) for key, value in class_names.items()}
    else:
        normalized = {index: str(value) for index, value in enumerate(class_names)}
    missing = [class_id for class_id in range(num_classes) if class_id not in normalized]
    if missing:
        raise ValueError(f"Missing class names for class ids: {missing}")
    return {class_id: normalized[class_id] for class_id in range(num_classes)}


def load_scene_samples(image_dir, labels_json, recursive=True, num_classes=None):
    image_files = list_images(image_dir, recursive=recursive)
    labels = load_label_dict(labels_json)
    if num_classes is None or int(num_classes) <= 0:
        num_classes = infer_num_classes_from_labels(labels_json)
    num_classes = int(num_classes)

    missing = [path.name for path in image_files if path.name not in labels]
    if missing:
        raise KeyError(f"Scene JSON misses {len(missing)} images; first missing: {missing[0]}")

    samples = []
    for path in image_files:
        label = int(labels[path.name])
        if label < 0 or label >= num_classes:
            raise ValueError(
                f"Invalid scene class {label} for {path.name}; num_classes={num_classes}"
            )
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

    rng = random.Random(seed)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    group_label_counts = {
        group_name: Counter(label for _, label in group_samples)
        for group_name, group_samples in groups.items()
    }
    total_counts = Counter(label for _, label in samples)
    target_counts = {
        class_id: max(1, round(count * val_fraction))
        for class_id, count in total_counts.items()
        if count > 0
    }

    val_groups = set()
    val_counts = Counter()
    class_order = sorted(
        target_counts,
        key=lambda class_id: sum(
            1 for group_count in group_label_counts.values() if group_count[class_id] > 0
        ),
    )
    for class_id in class_order:
        candidates = [
            group_name for group_name in group_names
            if group_name not in val_groups and group_label_counts[group_name][class_id] > 0
        ]
        candidates.sort(key=lambda group_name: group_label_counts[group_name][class_id], reverse=True)
        while val_counts[class_id] < target_counts[class_id] and candidates:
            group_name = candidates.pop(0)
            val_groups.add(group_name)
            val_counts.update(group_label_counts[group_name])

    if not val_groups:
        val_groups.add(group_names[0])
    if len(val_groups) >= len(groups):
        # Keep at least one group for training, preferring to return the largest
        # validation group back to train.
        largest_group = max(val_groups, key=lambda group_name: len(groups[group_name]))
        val_groups.remove(largest_group)

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


def class_counts(samples, num_classes=None):
    if num_classes is None:
        num_classes = max([label for _, label in samples] or [len(CLASS_NAMES) - 1]) + 1
    counts = Counter(label for _, label in samples)
    return {class_id: counts.get(class_id, 0) for class_id in range(int(num_classes))}


def balanced_class_weights(samples, num_classes=None):
    counts = class_counts(samples, num_classes=num_classes)
    if any(count == 0 for count in counts.values()):
        missing = [class_id for class_id, count in counts.items() if count == 0]
        raise RuntimeError(f"Training split has no samples for scene classes: {missing}")
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[class_id]) for class_id in counts]
    return torch.tensor(weights, dtype=torch.float32)
