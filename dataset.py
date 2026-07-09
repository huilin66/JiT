import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedRainDataset(Dataset):
    def __init__(self, rain_dir, clean_dir, transform=None):
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        # 假设你的雨滴图和干净图文件名完全一致
        self.image_files = sorted(os.listdir(rain_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        rain_img = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        rain_img, clean_img = self.transform(rain_img, clean_img)
        dummy_labels = torch.zeros(1, dtype=torch.long)
        return rain_img, clean_img, dummy_labels


class ScenePairedRainDataset(Dataset):
    def __init__(self, rain_dir, clean_dir, transform=None, scene_path=None):
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        # 假设你的雨滴图和干净图文件名完全一致
        self.image_files = sorted(os.listdir(rain_dir))
        self.transform = transform
        self.scene_path = scene_path
        self.scene_info = self._get_scene()

    def _get_scene(self):
        scene_path = self.scene_path
        df = pd.read_csv(scene_path, header=0, index_col=False)
        df["folder_name"] = df["folder_name"].astype(str).str.zfill(5)

        # 构造联合主键，例如 "N_00001", "D_00003"
        # df['type'] 里面是 'N' 或者 'D'
        df["union_key"] = df["type"].str.strip().str.upper() + "_" + df["folder_name"]

        # 制作成超快查询字典: {"N_00001": 1, "D_00003": 3, ...}
        scene_info = dict(zip(df["union_key"], df["class_id"]))

        return scene_info

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        rain_img = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        rain_img, clean_img = self.transform(rain_img, clean_img)

        # 1. 拆解文件名
        parts = img_name.split("_")
        time_type = parts[0]  # 'Day' 或 'Night'
        folder_name = parts[1]  # '00003'

        # 2. 将 'Day' 转换为 'D', 'Night' 转换为 'N'，对齐 CSV 里的格式
        time_prefix = "D" if time_type.lower() == "day" else "N"

        # 3. 拼装成联合主键去查表
        union_key = f"{time_prefix}_{folder_name}"  # 得到 "D_00003"

        # 4. 查表，查不到默认给 0
        class_id = int(self.scene_info.get(union_key, 0))
        dummy_labels = torch.zeros(1, dtype=torch.long) + class_id
        return rain_img, clean_img, dummy_labels


class ScenePairedRainDatasetV2(Dataset):
    def __init__(self, rain_dir, clean_dir, transform=None, scene_path=None):
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        # 假设你的雨滴图和干净图文件名完全一致
        self.image_files = sorted(os.listdir(rain_dir))
        self.transform = transform
        self.scene_path = scene_path
        self.scene_info = self._get_scene()

    def _get_scene(self):
        scene_path = self.scene_path
        with open(scene_path, "r") as f:
            scene_info = json.load(f)

        return scene_info

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        rain_img = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        rain_img, clean_img = self.transform(rain_img, clean_img)
        class_id = int(self.scene_info.get(img_name, 0))
        dummy_labels = torch.zeros(1, dtype=torch.long) + class_id
        return rain_img, clean_img, dummy_labels


class ValPatchDataset(Dataset):
    def __init__(self, rain_dir, clean_dir, patch_size=256, stride=256):
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        self.image_files = sorted(os.listdir(rain_dir))
        self.patch_size = patch_size
        self.stride = stride  # 验证时可以不重叠，stride 设为 256 提速

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        rain_pil = Image.open(rain_path).convert("RGB")
        clean_pil = Image.open(clean_path).convert("RGB")

        # 预处理到 Tensor [3, H, W] 范围 0-1
        rain_t = torch.from_numpy(np.array(rain_pil)).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(np.array(clean_pil)).permute(2, 0, 1).float() / 255.0

        W, H = rain_pil.size

        # 生成切片坐标 (和之前 infer 一样的逻辑)
        def get_coords(full_size, patch_size, stride):
            coords = []
            curr = 0
            while curr + patch_size < full_size:
                coords.append(curr)
                curr += stride
            coords.append(full_size - patch_size)
            return sorted(list(set(coords)))

        x_coords = get_coords(W, self.patch_size, self.stride)
        y_coords = get_coords(H, self.patch_size, self.stride)

        rain_patches, clean_patches = [], []
        for y in y_coords:
            for x in x_coords:
                rain_patches.append(
                    rain_t[:, y : y + self.patch_size, x : x + self.patch_size]
                )
                clean_patches.append(
                    clean_t[:, y : y + self.patch_size, x : x + self.patch_size]
                )

        # 拼成 [N, 3, 256, 256]
        rain_tensor = torch.stack(rain_patches, dim=0)
        clean_tensor = torch.stack(clean_patches, dim=0)

        dummy_labels = torch.zeros(rain_tensor.shape[0], dtype=torch.long)
        return rain_tensor, clean_tensor, dummy_labels


class SceneValPatchDataset(Dataset):
    def __init__(self, rain_dir, clean_dir, patch_size=256, stride=256, scene_path=None):
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        self.image_files = sorted(os.listdir(rain_dir))
        self.patch_size = patch_size
        self.stride = stride  # 验证时可以不重叠，stride 设为 256 提速
        self.scene_path = scene_path
        self.scene_info = self._get_scene()

    def _get_scene(self):
        scene_path = self.scene_path
        df = pd.read_csv(scene_path, header=0, index_col=False)
        df["folder_name"] = df["folder_name"].astype(str).str.zfill(5)

        # 构造联合主键，例如 "N_00001", "D_00003"
        # df['type'] 里面是 'N' 或者 'D'
        df["union_key"] = df["type"].str.strip().str.upper() + "_" + df["folder_name"]

        # 制作成超快查询字典: {"N_00001": 1, "D_00003": 3, ...}
        scene_info = dict(zip(df["union_key"], df["class_id"]))

        return scene_info

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        rain_pil = Image.open(rain_path).convert("RGB")
        clean_pil = Image.open(clean_path).convert("RGB")

        # 预处理到 Tensor [3, H, W] 范围 0-1
        rain_t = torch.from_numpy(np.array(rain_pil)).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(np.array(clean_pil)).permute(2, 0, 1).float() / 255.0

        W, H = rain_pil.size

        # 生成切片坐标 (和之前 infer 一样的逻辑)
        def get_coords(full_size, patch_size, stride):
            coords = []
            curr = 0
            while curr + patch_size < full_size:
                coords.append(curr)
                curr += stride
            coords.append(full_size - patch_size)
            return sorted(list(set(coords)))

        x_coords = get_coords(W, self.patch_size, self.stride)
        y_coords = get_coords(H, self.patch_size, self.stride)

        rain_patches, clean_patches = [], []
        for y in y_coords:
            for x in x_coords:
                rain_patches.append(
                    rain_t[:, y : y + self.patch_size, x : x + self.patch_size]
                )
                clean_patches.append(
                    clean_t[:, y : y + self.patch_size, x : x + self.patch_size]
                )

        # 拼成 [N, 3, 256, 256]
        rain_tensor = torch.stack(rain_patches, dim=0)
        clean_tensor = torch.stack(clean_patches, dim=0)

        parts = img_name.split("_")
        time_type = parts[0]  # 'Day' 或 'Night'
        folder_name = parts[1]  # '00003'
        time_prefix = "D" if time_type.lower() == "day" else "N"
        union_key = f"{time_prefix}_{folder_name}"  # 得到 "D_00003"
        class_id = int(self.scene_info.get(union_key, 0))
        dummy_labels = torch.zeros(rain_tensor.shape[0], dtype=torch.long) + class_id
        # print(rain_tensor.shape, clean_tensor.shape, dummy_labels.shape)
        return rain_tensor, clean_tensor, dummy_labels


class SceneValPatchDatasetV2(Dataset):
    def __init__(self, rain_dir, clean_dir, patch_size=256, stride=256, scene_path=None):
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        self.image_files = sorted(os.listdir(rain_dir))
        self.patch_size = patch_size
        self.stride = stride  # 验证时可以不重叠，stride 设为 256 提速
        self.scene_path = scene_path
        self.scene_info = self._get_scene()

    def _get_scene(self):
        scene_path = self.scene_path
        with open(scene_path, "r") as f:
            scene_info = json.load(f)

        return scene_info

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        rain_pil = Image.open(rain_path).convert("RGB")
        clean_pil = Image.open(clean_path).convert("RGB")

        # 预处理到 Tensor [3, H, W] 范围 0-1
        rain_t = torch.from_numpy(np.array(rain_pil)).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(np.array(clean_pil)).permute(2, 0, 1).float() / 255.0

        W, H = rain_pil.size

        # 生成切片坐标 (和之前 infer 一样的逻辑)
        def get_coords(full_size, patch_size, stride):
            coords = []
            curr = 0
            while curr + patch_size < full_size:
                coords.append(curr)
                curr += stride
            coords.append(full_size - patch_size)
            return sorted(list(set(coords)))

        x_coords = get_coords(W, self.patch_size, self.stride)
        y_coords = get_coords(H, self.patch_size, self.stride)

        rain_patches, clean_patches = [], []
        for y in y_coords:
            for x in x_coords:
                rain_patches.append(
                    rain_t[:, y : y + self.patch_size, x : x + self.patch_size]
                )
                clean_patches.append(
                    clean_t[:, y : y + self.patch_size, x : x + self.patch_size]
                )

        # 拼成 [N, 3, 256, 256]
        rain_tensor = torch.stack(rain_patches, dim=0)
        clean_tensor = torch.stack(clean_patches, dim=0)

        scene_id = self.scene_info.get(img_name, 0)
        # print(f'{img_name}, {scene_id}')
        class_id = int(scene_id)
        dummy_labels = torch.zeros(rain_tensor.shape[0], dtype=torch.long) + class_id
        return rain_tensor, clean_tensor, dummy_labels


class PairedRainDatasetInfer(Dataset):
    def __init__(self, rain_dir, transform=None):
        self.rain_dir = rain_dir
        # 假设你的雨滴图和干净图文件名完全一致
        self.image_files = sorted(os.listdir(rain_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)

        # 读取配对图片
        rain_img = Image.open(rain_path).convert("RGB")

        # 应用 JiT 原本的数据增强和归一化 (注意：两张图必须做相同的裁剪/缩放)
        if self.transform:
            # 简单起见，如果只做 Resize 和 ToTensor，可以直接分别 apply
            rain_img = self.transform(rain_img)

        # 返回的是：(输入网络图, 目标真值图)
        return rain_img, img_name


class PseudoRainTrainDataset(Dataset):
    """Training dataset where "clean" targets are pseudo-labels from a teacher model.

    Returns (rain, pseudo_gt, mask, is_pseudo, dummy_labels) where is_pseudo=1.
    mask is a single-channel restoration mask matching the pseudo dimensions.
    """

    def __init__(self, rain_dir, pseudo_dir, mask_dir=None, transform=None, scene_path=None):
        self.rain_dir = rain_dir
        self.pseudo_dir = pseudo_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.scene_path = scene_path
        self.scene_info = self._load_scene_info(scene_path)
        rain_files = sorted(os.listdir(rain_dir))
        pseudo_names = {p.name for p in Path(pseudo_dir).glob("*.png")}
        self.image_files = [f for f in rain_files if f in pseudo_names]
        if mask_dir:
            mask_names = {p.name for p in Path(mask_dir).glob("*_mask.png")}
        else:
            mask_names = set()
        self.has_masks = len(mask_names) > 0
        if not self.image_files:
            raise RuntimeError(
                f"No paired rain/pseudo images found. rain_dir={rain_dir}, pseudo_dir={pseudo_dir}"
            )
        print(f"[PseudoRainTrainDataset] {len(self.image_files)} paired samples "
              f"(rain={len(rain_files)}, pseudo={len(pseudo_names)}, masks={len(mask_names)})")

    def _load_scene_info(self, scene_path):
        if not scene_path:
            return None
        scene_path = Path(scene_path)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene label file not found: {scene_path}")
        if scene_path.suffix.lower() == ".json":
            with open(scene_path, "r", encoding="utf-8") as f:
                return json.load(f)
        df = pd.read_csv(scene_path, header=0, index_col=False)
        if {"type", "folder_name", "class_id"}.issubset(df.columns):
            df["folder_name"] = df["folder_name"].astype(str).str.zfill(5)
            df["union_key"] = df["type"].str.strip().str.upper() + "_" + df["folder_name"]
            return dict(zip(df["union_key"], df["class_id"]))
        raise ValueError(
            "Unsupported scene label file. Expected JSON filename map or CSV with "
            "type, folder_name, class_id columns."
        )

    def _scene_label(self, img_name):
        if self.scene_info is None:
            return 0
        if img_name in self.scene_info:
            return int(self.scene_info.get(img_name, 0))
        parts = img_name.split("_")
        if len(parts) >= 2:
            time_prefix = "D" if parts[0].lower() == "day" else "N"
            return int(self.scene_info.get(f"{time_prefix}_{parts[1]}", 0))
        return 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        pseudo_path = os.path.join(self.pseudo_dir, img_name)

        rain_img = Image.open(rain_path).convert("RGB")
        pseudo_img = Image.open(pseudo_path).convert("RGB")

        if self.mask_dir:
            stem = Path(img_name).stem
            mask_path = os.path.join(self.mask_dir, f"{stem}_mask.png")
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert("L")
            else:
                mask_img = Image.new("L", rain_img.size, 0)
        else:
            mask_img = Image.new("L", rain_img.size, 0)

        if self.transform:
            # Apply same spatial transform to rain, pseudo, and mask.
            rain_img, pseudo_img, mask_img = self.transform(rain_img, pseudo_img, mask_img)

        mask = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0)
        # Ensure mask has channel dim: [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        is_pseudo = torch.ones(1, dtype=torch.float32)
        dummy_labels = torch.zeros(1, dtype=torch.long) + self._scene_label(img_name)
        return rain_img, pseudo_img, mask, is_pseudo, dummy_labels


class MixedRealPseudoDataset(Dataset):
    """Wraps a real dataset and a pseudo dataset into one with uniform interface.

    Returns (rain, target, is_pseudo, mask, dummy_labels) for all samples.
    - is_pseudo=0 for real samples, is_pseudo=1 for pseudo samples.
    - mask: per-pixel restoration mask for pseudo; zeros for real.
    """

    def __init__(self, real_dataset, pseudo_dataset, pseudo_ratio=0.3):
        self.real_dataset = real_dataset
        self.pseudo_dataset = pseudo_dataset
        self.pseudo_ratio = float(pseudo_ratio)
        self.real_len = len(real_dataset)
        self.pseudo_len = len(pseudo_dataset)

        if self.pseudo_ratio <= 0 or self.pseudo_len == 0:
            self.pseudo_per_epoch = 0
        else:
            self.pseudo_per_epoch = int(
                self.real_len * self.pseudo_ratio / (1.0 - self.pseudo_ratio)
            )
            self.pseudo_per_epoch = min(self.pseudo_per_epoch, self.pseudo_len)
            self.pseudo_per_epoch = max(1, self.pseudo_per_epoch)

        self.total_len = self.real_len + self.pseudo_per_epoch
        print(
            f"[MixedRealPseudoDataset] real={self.real_len}, pseudo={self.pseudo_len}, "
            f"pseudo_ratio={self.pseudo_ratio:.2f}, pseudo_per_epoch={self.pseudo_per_epoch}, "
            f"total={self.total_len}"
        )

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.real_len:
            rain, clean, labels = self.real_dataset[idx]
            is_pseudo = torch.zeros(1, dtype=torch.float32)
            # Dummy zero mask matching spatial size.
            if isinstance(rain, torch.Tensor):
                _, h, w = rain.shape
            else:
                w, h = rain.size
            mask = torch.zeros(1, h, w, dtype=torch.float32)
            return rain, clean, is_pseudo, mask, labels
        else:
            pseudo_offset = idx - self.real_len
            if self.pseudo_per_epoch > 0:
                pseudo_idx = int(pseudo_offset * self.pseudo_len / self.pseudo_per_epoch)
            else:
                pseudo_idx = pseudo_offset
            pseudo_idx = pseudo_idx % max(1, self.pseudo_len)
            rain, pseudo, mask, is_pseudo, labels = self.pseudo_dataset[pseudo_idx]
            return rain, pseudo, is_pseudo, mask, labels
