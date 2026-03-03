import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import zipfile

# 导入你现有的模型和参数配置
from denoiser import Denoiser
from main_jit import get_args_parser

def zip_results(result_dir, csv_path, zip_path="submission.zip"):
    """
    将结果文件夹中的所有图片以及指定的 CSV 文件打包成 ZIP 压缩包。
    """
    print(f"\n📦 开始打包预测结果至压缩包: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        img_count = 0
        # 1. 遍历并打包所有预测出的图片
        for root, _, files in os.walk(result_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.basename(file_path) # 平铺在根目录
                    zipf.write(file_path, arcname)
                    img_count += 1
        print(f"✅ 成功打包了 {img_count} 张图片。")

        # 2. 打包指定的 CSV 文件
        if csv_path and os.path.exists(csv_path):
            csv_name = os.path.basename(csv_path)
            zipf.write(csv_path, csv_name)
            print(f"✅ 成功添加了附加文件: {csv_name}")
        else:
            print(f"⚠️ 提示：未找到 CSV 文件 '{csv_path}'，已跳过。")
            
    print(f"🎉 打包彻底完成！文件位于: {os.path.abspath(zip_path)}\n")


def batch_predict_resize():
    # 1. 路径与设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "./output/JiT-B-raindrop/16/checkpoint-last.pth" 
    input_dir = r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Val/Drop/00000'
    save_dir = "./results_resize"
    csv_file_path = "./readme.txt" 
    final_zip_name = "./raindrop_submission_resize.zip"
    os.makedirs(save_dir, exist_ok=True)

    # 2. 初始化模型并加载权重
    args = get_args_parser().parse_args()
    model = Denoiser(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # 优先加载 EMA 平滑权重以保证最高画质
    if 'model_ema1' in checkpoint:
        state_dict = checkpoint['model_ema1']
        print("✅ 成功加载了高画质 EMA 权重！")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("⚠️ 警告：未找到 EMA 权重，使用了原始带噪权重！")
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 3. 准备数据
    files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🚀 开始批量处理 {len(files)} 张图像 (使用 Resize 模式)...")

    # 4. 推理主循环
    for img_name in tqdm(files):
        img_path = os.path.join(input_dir, img_name)
        
        # --- (A) 图像读取与缩放 ---
        img_pil = Image.open(img_path).convert("RGB")
        orig_W, orig_H = img_pil.size # 记录原始大图尺寸
        
        # 强行 Resize 到模型输入尺寸 (例如 256x256)，使用 LANCZOS 保证抗锯齿画质
        img_resized = img_pil.resize((args.img_size, args.img_size), Image.Resampling.LANCZOS)
        
        # --- (B) 预处理转为 Tensor ---
        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(device) # [1, 3, 256, 256]
        x = (img_tensor / 127.5) - 1.0 # 归一化到 [-1.0, 1.0]

        # --- (C) 核心推理 (开启半精度和 10 步演化) ---
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # 调用 10步 Image-to-Image Rectified Flow
                output_tensor = model.generate_i2i(x, steps=10)

        # --- (D) 反归一化与后处理 ---
        out = output_tensor.squeeze(0).cpu().float()
        out = out.clamp(-1.0, 1.0)
        out = (out + 1.0) / 2.0 * 255.0
        out_np = out.permute(1, 2, 0).to(torch.uint8).numpy()
        
        # --- (E) 放大回原分辨率并保存 ---
        out_pil = Image.fromarray(out_np)
        # 将 256x256 再次通过高质量插值放大回 480x720
        final_img = out_pil.resize((orig_W, orig_H), Image.Resampling.LANCZOS)
        final_img.save(os.path.join(save_dir, img_name))

    print(f"✨ 图像推理完成！结果已暂存于: {save_dir}")

    
    zip_results(
        result_dir=save_dir, 
        csv_path=csv_file_path, 
        zip_path=final_zip_name
    )

if __name__ == "__main__":
    batch_predict_resize()