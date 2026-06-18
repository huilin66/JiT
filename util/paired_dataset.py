import random
from pathlib import Path

from PIL import Image
from torchvision.transforms import functional as TF
import torch


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def _resize_short_side(img, size):
    width, height = img.size
    if min(width, height) >= size:
        return img
    scale = size / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    return img.resize(new_size, Image.BICUBIC)


def _center_crop(img, size):
    width, height = img.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return img.crop((left, top, left + size, top + size))


def _random_crop_pair(rainy, clean, size):
    width, height = rainy.size
    if width == size and height == size:
        return rainy, clean
    left = random.randint(0, width - size)
    top = random.randint(0, height - size)
    box = (left, top, left + size, top + size)
    return rainy.crop(box), clean.crop(box)


class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rainy_dir,
        clean_dir=None,
        img_size=256,
        train=True,
        resize_size=0,
        hflip=True,
        vflip=True,
        rot90=True,
    ):
        self.rainy_dir = Path(rainy_dir)
        self.clean_dir = Path(clean_dir) if clean_dir else None
        self.img_size = img_size
        self.train = train
        self.resize_size = resize_size
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90

        rainy_images = _list_images(self.rainy_dir)
        if not rainy_images:
            raise RuntimeError(f"No images found under rainy_dir: {self.rainy_dir}")

        if self.clean_dir is None:
            self.samples = [(rainy, None, rainy.relative_to(self.rainy_dir).as_posix()) for rainy in rainy_images]
        else:
            clean_images = _list_images(self.clean_dir)
            clean_by_rel_stem = {
                clean.relative_to(self.clean_dir).with_suffix("").as_posix(): clean
                for clean in clean_images
            }
            clean_by_name_stem = {clean.stem: clean for clean in clean_images}
            samples = []
            missing = []
            for rainy in rainy_images:
                rel = rainy.relative_to(self.rainy_dir)
                clean = self.clean_dir / rel
                if not clean.exists():
                    clean = clean_by_rel_stem.get(rel.with_suffix("").as_posix())
                if clean is None:
                    clean = clean_by_name_stem.get(rainy.stem)
                if clean is None:
                    missing.append(rel.as_posix())
                    continue
                samples.append((rainy, clean, rel.as_posix()))
            if not samples:
                raise RuntimeError(
                    f"No paired images found. rainy_dir={self.rainy_dir}, clean_dir={self.clean_dir}"
                )
            if missing:
                print(f"Skipped {len(missing)} rainy images without clean pairs. First missing: {missing[0]}")
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _prepare_pair(self, rainy, clean):
        if self.resize_size > 0:
            target_size = (self.resize_size, self.resize_size)
            rainy = rainy.resize(target_size, Image.BICUBIC)
            clean = clean.resize(target_size, Image.BICUBIC)
        else:
            rainy = _resize_short_side(rainy, self.img_size)
            clean = _resize_short_side(clean, self.img_size)

        if self.train:
            rainy, clean = _random_crop_pair(rainy, clean, self.img_size)
            if self.hflip and random.random() < 0.5:
                rainy = TF.hflip(rainy)
                clean = TF.hflip(clean)
            if self.vflip and random.random() < 0.5:
                rainy = TF.vflip(rainy)
                clean = TF.vflip(clean)
            if self.rot90:
                k = random.randint(0, 3)
                if k:
                    angle = 90 * k
                    rainy = TF.rotate(rainy, angle)
                    clean = TF.rotate(clean, angle)
        else:
            rainy = _center_crop(rainy, self.img_size)
            clean = _center_crop(clean, self.img_size)

        return rainy, clean

    def __getitem__(self, index):
        rainy_path, clean_path, name = self.samples[index]
        rainy = Image.open(rainy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB") if clean_path else rainy.copy()
        rainy, clean = self._prepare_pair(rainy, clean)
        label = torch.tensor(0, dtype=torch.long)
        return TF.pil_to_tensor(rainy), TF.pil_to_tensor(clean), label, name
