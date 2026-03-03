import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
from denoiser import Denoiser
from main_jit import get_args_parser
import numpy as np
import zipfile


def img_process(img_path, patch_size=256, stride=128):
    img_pil = Image.open(img_path).convert("RGB")
    W, H = img_pil.size
    
    # 1. 预处理图像转为 Tensor: [1, 3, H, W]
    img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0) # [1, 3, H, W]

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
            patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((x, y))
    
    batch_tensor = torch.cat(patches, dim=0) # [N, 3, 256, 256]
    
    input_data = {
        'img': batch_tensor,
        'name': os.path.basename(img_path),
        'coords': coords,
        'patch_size': patch_size,
        'orig_size': (H, W) # 记录原始大图尺寸
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
    H, W = input_data['orig_size']
    patch_size = input_data['patch_size']
    
    out_container = torch.zeros((1, 3, H, W), device='cpu')
    count_mask = torch.zeros((1, 1, H, W), device='cpu') # 变成 1 通道掩码
    
    # 获取平滑窗口
    window = get_hanning_window(patch_size).to('cpu')

    for i, (x, y) in enumerate(input_data['coords']):
        # 💥 重点：预测出的 patch 乘以窗口权重！边缘的劣质像素被极大削弱
        weighted_patch = pred_patches[i:i+1].cpu() * window
        out_container[:, :, y:y+patch_size, x:x+patch_size] += weighted_patch
        count_mask[:, :, y:y+patch_size, x:x+patch_size] += window

    # 4. 还原逻辑：除以累加的权重掩码
    # 加 1e-5 防止边缘未覆盖区域除以 0
    final_output = out_container / count_mask.clamp_min(1e-8)
    final_output = final_output.clamp(-1.0, 1.0) 
    final_output = (final_output + 1.0) / 2.0 * 255.0 
    
    final_output = final_output.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()
    pred_img = Image.fromarray(final_output)
    pred_img.save(os.path.join(save_dir, input_data['name']))

def zip_results(result_dir, csv_path, zip_path="submission.zip"):
    """
    将结果文件夹中的所有图片以及指定的 CSV 文件打包成 ZIP 压缩包。
    """
    print(f"\n📦 开始打包预测结果至压缩包: {zip_path} ...")
    
    # 使用 ZIP_DEFLATED 进行标准压缩
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # 1. 遍历并添加所有预测出的图片
        img_count = 0
        for root, _, files in os.walk(result_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    # arcname 决定了文件在 zip 内部的路径。
                    # 这里使用 os.path.basename 让所有图片直接平铺在 zip 根目录，
                    # (如果是打榜，通常要求没有外层文件夹)。
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
                    img_count += 1
        print(f"✅ 成功打包了 {img_count} 张图片。")

        # 2. 添加指定的 CSV 文件
        if csv_path and os.path.exists(csv_path):
            csv_name = os.path.basename(csv_path)
            zipf.write(csv_path, csv_name)
            print(f"✅ 成功添加了附加文件: {csv_name}")
        else:
            print(f"⚠️ 警告：未找到 CSV 文件 '{csv_path}'，跳过打包该文件。")
            
    print(f"🎉 打包彻底完成！你的提交文件位于: {os.path.abspath(zip_path)}\n")

def batch_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "./output/JiT-B-raindrop/16/checkpoint-last.pth" 
    input_dir = r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Val/Drop/00000'
    save_dir = "./results_raindrop"
    csv_file_path = "./readme.txt" 
    final_zip_name = "./raindrop_submission.zip"
    os.makedirs(save_dir, exist_ok=True)

    args = get_args_parser().parse_args()
    model = Denoiser(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # 💥 重点修复：你的 checkpoint 里存的叫 'model_ema1'，不是 'model_ema'！
    if 'model_ema1' in checkpoint:
        state_dict = checkpoint['model_ema1']
        print("✅ 成功加载了高画质 EMA 权重！")
    else:
        state_dict = checkpoint['model']
        print("⚠️ 警告：未找到 EMA 权重，使用了原始带噪权重！")
        
    model.load_state_dict(state_dict)
    model.to(device).eval()

    files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🚀 开始批量处理 {len(files)} 张图像...")

    for img_name in tqdm(files):
        img_path = os.path.join(input_dir, img_name)
        input_data = img_process(img_path)
        
        x = input_data['img'].to(device).to(torch.float32).div_(255.0)
        x = x * 2.0 - 1.0

        with torch.no_grad():
            output_tensor = model.generate_i2i(x)

        result_process(output_tensor, input_data, save_dir)

    print(f"✨ 处理完成！结果已保存至: {save_dir}")

    zip_results(
        result_dir=save_dir, 
        csv_path=csv_file_path, 
        zip_path=final_zip_name
    )


if __name__ == "__main__":
    batch_predict()