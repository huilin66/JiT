import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np

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

        # 读取配对图片
        rain_img = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        # 应用 JiT 原本的数据增强和归一化 (注意：两张图必须做相同的裁剪/缩放)


        seed = random.randint(0, 2147483647)
        
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            rain_img = self.transform(rain_img)
            
            random.seed(seed)
            torch.manual_seed(seed)
            clean_img = self.transform(clean_img)
        
        return rain_img, clean_img

class ValPatchDataset(Dataset):
    def __init__(self, rain_dir, clean_dir, patch_size=256, stride=256):
        self.rain_dir = rain_dir
        self.clean_dir = clean_dir
        self.image_files = sorted(os.listdir(rain_dir))
        self.patch_size = patch_size
        self.stride = stride # 验证时可以不重叠，stride 设为 256 提速

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rain_path = os.path.join(self.rain_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        rain_pil = Image.open(rain_path).convert("RGB")
        clean_pil = Image.open(clean_path).convert("RGB")
        
        # 预处理到 Tensor [3, H, W] 范围 0-1
        rain_t = torch.from_numpy(np.array(rain_pil)).permute(2,0,1).float() / 255.0
        clean_t = torch.from_numpy(np.array(clean_pil)).permute(2,0,1).float() / 255.0

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
                rain_patches.append(rain_t[:, y:y+self.patch_size, x:x+self.patch_size])
                clean_patches.append(clean_t[:, y:y+self.patch_size, x:x+self.patch_size])
                
        # 拼成 [N, 3, 256, 256]
        rain_tensor = torch.stack(rain_patches, dim=0)
        clean_tensor = torch.stack(clean_patches, dim=0)

        return rain_tensor, clean_tensor

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