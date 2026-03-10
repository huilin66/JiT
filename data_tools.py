import csv
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def analyze_image(img_path):
    """提取图像的亮度和清晰度特征"""
    # 1. 读取图像并转为灰度图
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 计算平均亮度 (Day vs Night)
    mean_intensity = np.mean(gray)

    # 3. 计算拉普拉斯方差 (Focus Background vs Raindrop)
    # 返回值越大，代表图像整体越锐利/边缘越多 (对焦在背景)
    # 返回值越小，代表图像整体越模糊/平滑 (对焦在雨滴，背景虚化)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return mean_intensity, laplacian_var


def generate_pseudo_labels(
    rain_dir, output_csv, INTENSITY_THRESH=80.0, LAPLACIAN_THRESH=500.0
):
    # ================= 配置区 =================
    # 只需要对带雨图 (Drop) 目录进行特征提取即可

    files = [f for f in os.listdir(rain_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    print(f"🚀 开始分析 {len(files)} 张图像的物理特征...")

    results = []

    for img_name in tqdm(files):
        img_path = os.path.join(rain_dir, img_name)
        mean_int, lap_var = analyze_image(img_path)

        if mean_int is None:
            print(f"⚠️ 警告: 无法读取图像 {img_name}")
            continue

        # 逻辑判断
        is_day = mean_int > INTENSITY_THRESH
        is_bg_focus = lap_var > LAPLACIAN_THRESH

        # 标签映射
        # 0: night_bg_focus
        # 1: night_raindrop_focus
        # 2: day_bg_focus
        # 3: day_raindrop_focus
        if not is_day and is_bg_focus:
            label = 0
        elif not is_day and not is_bg_focus:
            label = 1
        elif is_day and is_bg_focus:
            label = 2
        else:  # is_day and not is_bg_focus
            label = 3

        results.append(
            {
                "filename": img_name,
                "mean_intensity": round(mean_int, 2),
                "laplacian_var": round(lap_var, 2),
                "is_day": int(is_day),
                "is_bg_focus": int(is_bg_focus),
                "label": label,
            }
        )

    # 将结果写入 CSV
    print(f"\n💾 正在保存结果至 {output_csv}...")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "mean_intensity",
                "laplacian_var",
                "is_day",
                "is_bg_focus",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("✨ 自动标注完成！")


def copy_data(
    src_dir,
    dst_dir,
    drop_dir_name,
    clear_dir_name,
    type="Day",
):
    """
    Copy data from src_dir to dst_dir.
    """
    dst_dir_drop = os.path.join(dst_dir, drop_dir_name)
    dst_dir_clear = os.path.join(dst_dir, clear_dir_name)
    os.makedirs(dst_dir_drop, exist_ok=True)
    os.makedirs(dst_dir_clear, exist_ok=True)

    src_dir_drop = os.path.join(src_dir, drop_dir_name)
    src_dir_clear = os.path.join(src_dir, clear_dir_name)

    src_drop_list = os.listdir(src_dir_drop)
    for sub_id_name in tqdm(src_drop_list, desc=f"Copy {type} Drop Data"):
        src_id_path = os.path.join(src_dir_drop, sub_id_name)
        for file_name in os.listdir(src_id_path):
            src_file_path = os.path.join(src_id_path, file_name)
            dst_file_path = os.path.join(
                dst_dir_drop, f"{type}_{sub_id_name}_{file_name}"
            )
            shutil.copy(src_file_path, dst_file_path)

    src_clear_list = os.listdir(src_dir_clear)
    for sub_id_name in tqdm(src_clear_list, desc=f"Copy {type} Clear Data"):
        src_id_path = os.path.join(src_dir_clear, sub_id_name)
        for file_name in os.listdir(src_id_path):
            src_file_path = os.path.join(src_id_path, file_name)
            dst_file_path = os.path.join(
                dst_dir_clear, f"{type}_{sub_id_name}_{file_name}"
            )
            shutil.copy(src_file_path, dst_file_path)


def count_dataset_images(base_dir):
    total_images = 0
    folder_counts = {}

    # 支持的常见图像格式
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    print(f"📁 开始统计目录: {base_dir}")
    print("-" * 35)
    print(f"{'文件夹名称':<15} | {'图片数量'}")
    print("-" * 35)

    # 遍历 Drop 目录下的所有子文件夹，并按名称排序
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)

        # 确保它是一个文件夹
        if os.path.isdir(folder_path):
            # 统计该文件夹下的图像文件数量
            images = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith(valid_extensions)
            ]
            count = len(images)
            folder_counts[folder_name] = count
            total_images += count

            print(f"{folder_name:<15} | {count}")

    print("-" * 35)
    print(f"🎯 统计完成！")
    print(f"📂 总文件夹数: {len(folder_counts)}")
    print(f"🖼️ 总图片总数: {total_images}")


import os
import shutil

from tqdm import tqdm


def extract_sample_images(source_dir, dest_dir):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"📂 创建了用于肉眼审查的文件夹: {dest_dir}")

    # 获取所有子文件夹 (例如 '00166', '00167')
    folders = [
        f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))
    ]
    folders.sort()  # 按文件夹名字典序排好

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    count = 0

    print(f"🚀 开始从 {len(folders)} 个文件夹中抽样图片...")

    for folder_name in tqdm(folders):
        folder_path = os.path.join(source_dir, folder_name)

        # 找出该文件夹下的所有图片
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

        if not files:
            print(f"⚠️ 警告: 文件夹 {folder_name} 是空的，已跳过。")
            continue

        # 排序后，取该文件夹里的第一张图片作为代表
        files.sort()
        sample_file = files[0]
        src_path = os.path.join(folder_path, sample_file)

        # 获取原图的后缀名 (如 .png)
        ext = os.path.splitext(sample_file)[1]

        # 核心：重命名为 "文件夹名.后缀" (例如: 00166.png)
        new_name = f"{folder_name}{ext}"
        dest_path = os.path.join(dest_dir, new_name)

        # 复制文件
        shutil.copy2(src_path, dest_path)
        count += 1

    print(f"\n✨ 抽样完成！")
    print(f"🎯 共成功提取了 {count} 张场景代表图。")
    print(f"👉 请打开 {os.path.abspath(dest_dir)} 文件夹进行肉眼判断。")


import os

import pandas as pd


def convert_manual_labels(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"❌ 找不到输入文件: {input_csv}")
        return

    print(f"🚀 开始读取并转换人工标注数据: {input_csv} ...")
    df = pd.read_excel(input_csv, sheet_name="Sheet1")

    # 1. 自动把 id 补齐为 5 位数的文件夹名 (例如: 1 -> "00001")
    # 这样才能和你实际的 Drop/00001 文件夹完美对应
    df["folder_name"] = df["id"].apply(lambda x: str(int(x)).zfill(5))

    # 2. 定义映射函数
    def get_class_id(row):
        day_night = str(row["type"]).strip().upper()  # 'D' 或 'N'
        bg_focus = str(row["bg focus"]).strip()  # '0' 或 '1'

        is_day = day_night == "D"
        is_bg_focus = bg_focus == "1"

        # 0: 黑夜_背景聚焦 | 1: 黑夜_雨滴聚焦
        # 2: 白天_背景聚焦 | 3: 白天_雨滴聚焦
        if not is_day and is_bg_focus:
            return 0
        elif not is_day and not is_bg_focus:
            return 1
        elif is_day and is_bg_focus:
            return 2
        else:
            return 3

    # 3. 应用映射函数，生成新列 'class_id'
    df["class_id"] = df.apply(get_class_id, axis=1)

    # 4. 只保留我们需要的两列，保存为最终的 CSV
    final_df = df
    final_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n✨ 转换成功！")
    print(f"📊 类别统计结果如下：")
    print(df["class_id"].value_counts().sort_index())
    print(f"\n📁 最终文件已保存至: {os.path.abspath(output_csv)}")
    print(f"💡 文件预览:\n{final_df.head()}")


if __name__ == "__main__":
    pass
    DATA_DAY = (
        r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/DayRainDrop_Train/Drop"
    )
    DATA_NIGHT = (
        r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/NightRainDrop_Train/Drop"
    )
    # count_dataset_images(DATA_DAY)
    # count_dataset_images(DATA_NIGHT)
    # DATA_DAY = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/DayRainDrop_Train"
    # DATA_NIGHT = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/NightRainDrop_Train"
    # DATA_MERGE = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2"
    # DROP_DIR_NAME = "Drop"
    # CLEAR_DIR_NAME = "Clear"
    # copy_data(DATA_DAY, DATA_MERGE, drop_dir_name=DROP_DIR_NAME, type="Day")
    # copy_data(DATA_NIGHT, DATA_MERGE, clear_dir_name=CLEAR_DIR_NAME, type="Night")

    # ================= 配置区 =================
    # 你的 Drop 文件夹绝对路径 (包含 00166, 00167 等子文件夹的父目录)

    # extract_sample_images(DATA_DAY, DATA_DAY + "_overall")
    # extract_sample_images(DATA_NIGHT, DATA_NIGHT + "_overall")

    # rain_dir = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop"
    # output_csv = "/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/train_scene.csv"

    # INTENSITY_THRESH = 100.0  # 大于80认为是白天，小于等于80认为是黑夜
    # LAPLACIAN_THRESH = 500.0  # 大于500认为是背景清晰(Focus BG)，小于等于500认为是背景虚化(Focus Raindrop)
    # day140
    # generate_pseudo_labels(rain_dir, output_csv, INTENSITY_THRESH, LAPLACIAN_THRESH)

    input_csv = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/record.xlsx"  # 你上传的文件名
    output_csv = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/record.csv"  # 你上传的文件名
    convert_manual_labels(input_csv, output_csv)
