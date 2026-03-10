import json
import os
import zipfile

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from denoiser import Denoiser
from main_jit import get_args_parser
from scene_predicter import batch_predict_and_save


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


def batch_predict(weight_name, select_name,  step_num = 10, pred_scene=True, path_in=False, device=1):
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print(f"load {weight_name}/{select_name}, pred scene{pred_scene}, device: {device} ...")

    data_root = r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026'
    ckpt_path = f"output/{weight_name}/16/checkpoint-{select_name}.pth" if path_in else f"{data_root}/output/{weight_name}/16/checkpoint-{select_name}.pth"
    input_dir = f"{data_root}/RainDrop_Val/Drop/00000"
    save_dir = f"{data_root}/submission/{weight_name}_step{step_num}_{select_name}"
    final_zip_name = f"{data_root}/submission/submission_{weight_name}_step{step_num}_{select_name}.zip"
    scene_model_path = "scene_classifier_resnet18.pth"
    csv_file_path = "readme.txt"
    os.makedirs(save_dir, exist_ok=True)
    scene_pred_path = save_dir + "_scene.json"
    if pred_scene:
        if os.path.exists(scene_pred_path):
            with open(scene_pred_path, "r") as f:
                scene_dict = json.load(f)
        else:
            scene_dict = batch_predict_and_save(
                scene_model_path, input_dir, scene_pred_path
            )

    args = get_args_parser().parse_args()
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

    for img_name in tqdm(files, desc="predict"):
        img_path = os.path.join(input_dir, img_name)
        input_data = img_process(img_path)

        x = input_data["img"].to(device).to(torch.float32).div_(255.0)
        x = x * 2.0 - 1.0
        if pred_scene:
            scene_id = scene_dict[img_name]
            dummy_labels = (
                torch.zeros(x.shape[0], dtype=torch.long, device=x.device) + scene_id
            )
        else:
            dummy_labels = None
        with torch.no_grad():
            output_tensor = model.generate_i2i(
                x, steps=step_num, dummy_labels=dummy_labels
            )

        result_process(output_tensor, input_data, save_dir)

    print(f"result save to {save_dir}")

    zip_results(result_dir=save_dir, csv_path=csv_file_path, zip_path=final_zip_name)


if __name__ == "__main__":

    weight_names = [
        # "JiT-B-raindrop01",
        "JiT-B-raindrop03",
        "JiT-B-raindrop01",
        "JiT-B-raindrop03",
    ]
    pred_scenes = [
        # False,
        True,
        True,
        False,
    ]
    path_list = [
        # True,
        False,
        True,
        False
    ]
    select_name = ["last", "best"]

    for weight_name, pred_scene, path_in in zip(weight_names, pred_scenes, path_list):
        batch_predict(weight_name, select_name[0], pred_scene=pred_scene, path_in=path_in)
        batch_predict(weight_name, select_name[1], pred_scene=pred_scene, path_in=path_in)

