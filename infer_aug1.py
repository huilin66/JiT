import json
import os
import zipfile
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from denoiser import Denoiser
from main_jit import get_args_parser
from scene_predicter import batch_predict_and_save


class EnhancedInferencer:
    def __init__(
        self,
        models,
        model_weights=None,
        use_h_flip=True,
        scales=[1.0],  # e.g., [0.8, 1.0, 1.2]
        scale_weights=None,
        use_color_match=True,
        patch_size=256,
        stride=128,
    ):
        """
        高阶推理引擎：支持多模型集成、多尺度、水平翻转 TTA 及色彩对齐。
        :param models: 模型列表 (支持单个模型传入 [model] 或多模型 [model_H, model_B])
        :param model_weights: 多模型的融合权重，默认平均
        :param use_h_flip: 是否开启水平翻转 TTA
        :param scales: 多尺度列表，例如 [0.8, 1.0, 1.2]
        :param scale_weights: 不同尺度的融合权重，默认给 1.0 最大权重
        :param use_color_match: 是否开启输出色彩/亮度向原图对齐
        """
        self.models = models if isinstance(models, list) else [models]
        self.model_weights = model_weights or [1.0 / len(self.models)] * len(
            self.models
        )
        self.use_h_flip = use_h_flip

        self.scales = scales
        if scale_weights is None:
            # 默认给原尺寸最高权重，其他尺寸权重平分
            self.scale_weights = [1.0 if s == 1.0 else 0.5 for s in scales]
        else:
            self.scale_weights = scale_weights

        # 归一化权重
        sum_mw = sum(self.model_weights)
        self.model_weights = [w / sum_mw for w in self.model_weights]
        sum_sw = sum(self.scale_weights)
        self.scale_weights = [w / sum_sw for w in self.scale_weights]

        self.use_color_match = use_color_match
        self.patch_size = patch_size
        self.stride = stride

    def _color_luminance_match(self, orig_tensor, pred_tensor, blend_ratio=0.5):
        """
        极其重要的后处理：将预测图的全局均值和方差向原图轻微对齐，防止色彩漂移拉低 PSNR。
        orig_tensor, pred_tensor shape: [1, 3, H, W], 范围 [-1, 1]
        """
        mu_orig = orig_tensor.mean(dim=(2, 3), keepdim=True)
        std_orig = orig_tensor.std(dim=(2, 3), keepdim=True) + 1e-6

        mu_pred = pred_tensor.mean(dim=(2, 3), keepdim=True)
        std_pred = pred_tensor.std(dim=(2, 3), keepdim=True) + 1e-6

        # 均值-方差匹配
        matched = (pred_tensor - mu_pred) / std_pred * std_orig + mu_orig

        # 保守混合：blend_ratio 控制对齐的强度，0.5 表示保留一半模型自身的色彩特性
        out = (1.0 - blend_ratio) * pred_tensor + blend_ratio * matched
        return torch.clamp(out, -1.0, 1.0)

    def _core_patch_inference(self, model, img_tensor, step_num, dummy_labels):
        """核心的分块推理逻辑 (复用你原有的 Hanning 逻辑)"""
        _, _, H, W = img_tensor.shape

        # 1. 切片
        x_coords = sorted(
            list(
                set(
                    list(range(0, W - self.patch_size, self.stride))
                    + [W - self.patch_size]
                )
            )
        )
        y_coords = sorted(
            list(
                set(
                    list(range(0, H - self.patch_size, self.stride))
                    + [H - self.patch_size]
                )
            )
        )

        patches = []
        for y in y_coords:
            for x in x_coords:
                patches.append(
                    img_tensor[:, :, y : y + self.patch_size, x : x + self.patch_size]
                )
        batch_tensor = torch.cat(patches, dim=0)  # [N, 3, 256, 256]

        # ==========================================================
        # 💥 核心修复点：将 label 的 batch 数量动态扩展到与 patch 数量 (N) 相同
        # ==========================================================
        if dummy_labels is not None:
            expanded_labels = dummy_labels.expand(batch_tensor.shape[0])
        else:
            expanded_labels = None

        # 2. 推理 (加入了 Mini-batching 防止 A100 OOM)
        pred_patches = []
        batch_size = 16  # 根据显存调整
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for i in range(0, batch_tensor.shape[0], batch_size):
                x_chunk = batch_tensor[i : i + batch_size]
                lbl_chunk = (
                    expanded_labels[i : i + batch_size]
                    if expanded_labels is not None
                    else None
                )
                out_chunk = model.generate_i2i(
                    x_chunk, steps=step_num, dummy_labels=lbl_chunk
                )
                pred_patches.append(out_chunk.float())
        pred_patches = torch.cat(pred_patches, dim=0)

        # 3. Hanning Window 拼合
        out_container = torch.zeros((1, 3, H, W), device=img_tensor.device)
        count_mask = torch.zeros((1, 1, H, W), device=img_tensor.device)

        window_1d = torch.hann_window(self.patch_size, device=img_tensor.device)
        window = (
            (window_1d.unsqueeze(0) * window_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        )

        idx = 0
        for y in y_coords:
            for x in x_coords:
                weighted_patch = pred_patches[idx : idx + 1] * window
                out_container[
                    :, :, y : y + self.patch_size, x : x + self.patch_size
                ] += weighted_patch
                count_mask[:, :, y : y + self.patch_size, x : x + self.patch_size] += (
                    window
                )
                idx += 1

        final_output = out_container / count_mask.clamp_min(1e-8)

        # 💥 关键点：一定要 return 回去！
        return torch.clamp(final_output, -1.0, 1.0)

    def forward(self, img_tensor, step_num, dummy_labels):
        """
        img_tensor: [1, 3, H, W], 范围 [-1, 1], 位于 GPU
        """
        _, _, orig_H, orig_W = img_tensor.shape
        final_ensemble_output = torch.zeros_like(img_tensor)

        # 遍历所有模型 (Model Ensembling)
        for model, model_w in zip(self.models, self.model_weights):
            model_output = torch.zeros_like(img_tensor)

            # 遍历所有尺度 (Multi-Scale Inference)
            for scale, scale_w in zip(self.scales, self.scale_weights):
                if scale != 1.0:
                    scaled_img = F.interpolate(
                        img_tensor,
                        scale_factor=scale,
                        mode="bicubic",
                        align_corners=False,
                    )
                else:
                    scaled_img = img_tensor

                scale_pred_accum = torch.zeros_like(scaled_img)
                tta_count = 0

                # 原始方向推理
                scale_pred_accum += self._core_patch_inference(
                    model, scaled_img, step_num, dummy_labels
                )
                tta_count += 1

                # 水平翻转 TTA (Horizontal Flip)
                if self.use_h_flip:
                    flipped_img = torch.flip(scaled_img, dims=[3])
                    flipped_pred = self._core_patch_inference(
                        model, flipped_img, step_num, dummy_labels
                    )
                    scale_pred_accum += torch.flip(flipped_pred, dims=[3])
                    tta_count += 1

                # 当前尺度的平均结果
                scale_pred = scale_pred_accum / tta_count

                # 将其他尺度的结果缩放回原图尺寸
                if scale != 1.0:
                    scale_pred = F.interpolate(
                        scale_pred,
                        size=(orig_H, orig_W),
                        mode="bicubic",
                        align_corners=False,
                    )

                model_output += scale_pred * scale_w

            final_ensemble_output += model_output * model_w

        # 开启后处理：色彩/亮度校准
        if self.use_color_match:
            final_ensemble_output = self._color_luminance_match(
                img_tensor, final_ensemble_output
            )

        return final_ensemble_output


def img_process(img_path, patch_size=256, stride=128):
    img_pil = Image.open(img_path).convert("RGB")
    W, H = img_pil.size

    # 1. 预处理图像转为 Tensor: [1, 3, H, W]
    img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]

    # 2. 准备切片坐标：确保覆盖全图且边缘对齐
    def get_coords(full_size, patch_size, stride):
        coords = []
        curr = 0
        while curr + patch_size < full_size:
            coords.append(curr)
            curr += stride
        coords.append(full_size - patch_size)
        return sorted(list(set(coords)))

    x_coords = get_coords(W, patch_size, stride)
    y_coords = get_coords(H, patch_size, stride)

    # 3. 提取所有 patches 并组成一个 Batch
    patches = []
    coords = []
    for y in y_coords:
        for x in x_coords:
            patch = img_tensor[:, :, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coords.append((x, y))

    batch_tensor = torch.cat(patches, dim=0)  # [N, 3, 256, 256]

    input_data = {
        "img": batch_tensor,
        "name": os.path.basename(img_path),
        "coords": coords,
        "patch_size": patch_size,
        "orig_size": (H, W),  # 记录原始大图尺寸
    }
    return input_data


def get_hanning_window(patch_size):
    # 生成 1D 汉宁窗 (边缘为 0, 中心为 1)
    window_1d = torch.hann_window(patch_size)
    # 变成 2D [256, 256]
    window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)
    # 扩展为 [1, 1, 256, 256] 以便和 tensor 广播相乘
    return window_2d.unsqueeze(0).unsqueeze(0)


def result_process(pred_patches, input_data, save_dir):
    H, W = input_data["orig_size"]
    patch_size = input_data["patch_size"]

    out_container = torch.zeros((1, 3, H, W), device="cpu")
    count_mask = torch.zeros((1, 1, H, W), device="cpu")  # 变成 1 通道掩码

    # 获取平滑窗口
    window = get_hanning_window(patch_size).to("cpu")

    for i, (x, y) in enumerate(input_data["coords"]):
        # 💥 重点：预测出的 patch 乘以窗口权重！边缘的劣质像素被极大削弱
        weighted_patch = pred_patches[i : i + 1].cpu() * window
        out_container[:, :, y : y + patch_size, x : x + patch_size] += weighted_patch
        count_mask[:, :, y : y + patch_size, x : x + patch_size] += window

    # 4. 还原逻辑：除以累加的权重掩码
    # 加 1e-5 防止边缘未覆盖区域除以 0
    final_output = out_container / count_mask.clamp_min(1e-8)
    final_output = final_output.clamp(-1.0, 1.0)
    final_output = (final_output + 1.0) / 2.0 * 255.0

    final_output = final_output.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()
    pred_img = Image.fromarray(final_output)
    pred_img.save(os.path.join(save_dir, input_data["name"]))


def zip_results(result_dir, csv_path, zip_path="submission.zip"):
    """
    将结果文件夹中的所有图片以及指定的 CSV 文件打包成 ZIP 压缩包。
    """
    print(f"\n📦 开始打包预测结果至压缩包: {zip_path} ...")

    # 使用 ZIP_DEFLATED 进行标准压缩
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. 遍历并添加所有预测出的图片
        img_count = 0
        for root, _, files in os.walk(result_dir):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(root, file)
                    # arcname 决定了文件在 zip 内部的路径。
                    # 这里使用 os.path.basename 让所有图片直接平铺在 zip 根目录，
                    # (如果是打榜，通常要求没有外层文件夹)。
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
                    img_count += 1
        print(f"zip {img_count} image。")

        # 2. 添加指定的 CSV 文件
        if csv_path and os.path.exists(csv_path):
            csv_name = os.path.basename(csv_path)
            zipf.write(csv_path, csv_name)
            print(f"zip {csv_name}")
        else:
            print(f"connot find '{csv_path}', skip")

    print(f"zip result save to {zip_path}\n")


def batch_predict(
    weight_name,
    select_name,
    step_num=10,
    use_bg_subnet=False,
    use_scene_dataset=True,
    device=1,
    use_h_flip=False,  # ✅ 开启水平翻转 (稳提 0.1-0.2 dB)
    scales=[1.0],  # ✅ 如果要开启多尺度，改为 [0.8, 1.0, 1.2]，但速度会变慢 3 倍
    use_color_match=False,  # ✅ 开启色彩匹配
    patch_size=256,
    stride=128,  # 如果速度太慢，这里可以改成 192
):
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print(
        f"load {weight_name}/{select_name}, pred scene{use_scene_dataset}, device: {device} ..."
    )
    image_batch_size = 6
    data_root = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026"
    ckpt_path = f"{data_root}/output/{weight_name}/16/checkpoint-{select_name}.pth"
    # input_dir = f"{data_root}/RainDrop_Val/Drop/00000"
    input_dir = f"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/test-input"

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_dir = f"{data_root}/submission_test/{weight_name}_step{step_num}_{select_name}_{use_scene_dataset}"
    # final_zip_name = f"{data_root}/submission_test/{time_str}.zip"
    save_dir = f"{data_root}/submission_test-input/{weight_name}_step{step_num}_{select_name}_{use_scene_dataset}"
    final_zip_name = f"{data_root}/submission_test-input/{time_str}.zip"
    scene_model_path = "scene_classifier_resnet18.pth"
    csv_file_path = "readme.txt"
    os.makedirs(save_dir, exist_ok=True)
    scene_pred_path = save_dir + "_scene.json"
    if use_scene_dataset:
        if os.path.exists(scene_pred_path):
            with open(scene_pred_path, "r") as f:
                scene_dict = json.load(f)
        else:
            scene_dict = batch_predict_and_save(
                scene_model_path, input_dir, scene_pred_path
            )

    args = get_args_parser().parse_args()
    args.use_bg_subnet = use_bg_subnet
    args.model = "JiT-H/16"
    model = Denoiser(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 💥 重点修复：你的 checkpoint 里存的叫 'model_ema1'，不是 'model_ema'！
    if "model_ema1" in checkpoint:
        state_dict = checkpoint["model_ema1"]
        print("load high quality EMA weight model_ema1")
    else:
        state_dict = checkpoint["model"]
        print("cannot find EMA weight model, use source weight")

    model.load_state_dict(state_dict)
    model.to(device).eval()

    files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    inferencer = EnhancedInferencer(
        models=model,  # 可以是单个 model，也可以是 [model_ema1, model_ema2]
        use_h_flip=use_h_flip,  # ✅ 开启水平翻转 (稳提 0.1-0.2 dB)
        scales=scales,  # ✅ 如果要开启多尺度，改为 [0.8, 1.0, 1.2]，但速度会变慢 3 倍
        use_color_match=use_color_match,  # ✅ 开启色彩匹配
        patch_size=patch_size,
        stride=stride,  # 如果速度太慢，这里可以改成 192
    )

    for img_name in tqdm(files, desc="predict"):
        img_path = os.path.join(input_dir, img_name)

        # 只需要读取一次整图
        img_pil = Image.open(img_path).convert("RGB")
        W, H = img_pil.size
        img_tensor = (
            torch.from_numpy(np.array(img_pil))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )
        img_tensor = img_tensor.div_(255.0) * 2.0 - 1.0  # 转换到 [-1, 1]

        if use_scene_dataset:
            scene_id = scene_dict[img_name]
            dummy_labels = torch.zeros(1, dtype=torch.long, device=device) + scene_id
        else:
            dummy_labels = None

        # 直接调用强大的推理器
        final_tensor = inferencer.forward(img_tensor, step_num, dummy_labels)

        # 还原并保存
        final_output = (final_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0
        final_output = (
            final_output.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        )
        pred_img = Image.fromarray(final_output)
        pred_img.save(os.path.join(save_dir, img_name), optimize=True)
    print(f"result save to {save_dir}")

    zip_results(result_dir=save_dir, csv_path=csv_file_path, zip_path=final_zip_name)


if __name__ == "__main__":
    weight_names = [
        # "JiT-B-raindrop13",
        # "JiT-B-raindrop13",
        # "JiT-H-raindrop24",
        "JiT-H-raindrop22",
    ]
    use_bg_subnet_list = [
        # True,
        # True,
        False,
        # False,
    ]
    use_scene_dataset = [
        # False,
        True,
    ]
    select_name = ["last", "best"]

    for i in range(len(weight_names)):
        # batch_predict(
        #     weight_names[i],
        #     select_name[0],
        #     use_bg_subnet=use_bg_subnet_list[i],
        #     use_scene_dataset=use_scene_dataset[i],
        #     scales=[0.8, 1.0, 1.25],
        # )
        batch_predict(
            weight_names[i],
            select_name[0],
            use_bg_subnet=use_bg_subnet_list[i],
            use_scene_dataset=use_scene_dataset[i],
            scales=[0.8, 1.0, 1.25],
            use_h_flip=True,
            use_color_match=True,
            step_num=1,
        )
