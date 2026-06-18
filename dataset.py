import json
import os

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

        # seed = random.randint(0, 2147483647)
        #
        # if self.transform:
        #     random.seed(seed)
        #     torch.manual_seed(seed)
        #     rain_img = self.transform(rain_img)
        #
        #     random.seed(seed)
        #     torch.manual_seed(seed)
        #     clean_img = self.transform(clean_img)
        #
        # return rain_img, clean_img


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
        scene_path = self.scene_path or r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/record.csv"
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
        scene_path = self.scene_path or r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop_scen_pred.json"
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
        scene_path = self.scene_path or r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/record.csv"
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
        scene_path = self.scene_path or r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop_scen_pred.json"
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
