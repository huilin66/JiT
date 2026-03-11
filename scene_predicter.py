import json
import os

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ==========================================
# 1. 构建适配你文件结构的 Dataset
# ==========================================
class SceneDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        """
        data_dir: 存放 00001, 00002 等文件夹的父目录 (例如 .../Drop)
        csv_path: 我们之前生成的 final_dataset_labels.csv
        """
        self.transform = transform

        # 读取 CSV 并构建字典
        self.scene_info = self._get_scene(csv_path)

        self.image_paths = []
        self.labels = []

        print("🔍 正在扫描数据集目录...")
        # 遍历所有文件夹，把里面的图片全扫出来
        for image_name in os.listdir(data_dir):
            image_path = os.path.join(data_dir, image_name)
            if os.path.isfile(image_path) and image_name.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                parts = image_name.split("_")
                time_type = parts[0]  # 'Day' 或 'Night'
                folder_name = parts[1]  # '00003'
                time_prefix = "D" if time_type.lower() == "day" else "N"
                union_key = f"{time_prefix}_{folder_name}"  # 得到 "D_00003"
                class_id = int(self.scene_info.get(union_key))

                self.image_paths.append(image_path)
                self.labels.append(class_id)

        print(f"✅ 成功加载 {len(self.image_paths)} 张图片，分为 4 个类别。")

    def _get_scene(self, csv_path):
        df = pd.read_csv(csv_path, header=0, index_col=False)
        df["folder_name"] = df["folder_name"].astype(str).str.zfill(5)

        # 构造联合主键，例如 "N_00001", "D_00003"
        # df['type'] 里面是 'N' 或者 'D'
        df["union_key"] = df["type"].str.strip().str.upper() + "_" + df["folder_name"]

        # 制作成超快查询字典: {"N_00001": 1, "D_00003": 3, ...}
        scene_info = dict(zip(df["union_key"], df["class_id"]))

        return scene_info

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label


# ==========================================
# 2. 核心训练函数
# ==========================================
def train_classifier():
    # --- 配置参数 ---

    DATA_DIR = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop"  # 替换为实际路径
    CSV_PATH = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/record.csv"
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # --- 数据增强与加载 ---
    # 分类任务的标准预处理：缩放、中心裁剪、转Tensor、ImageNet归一化
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),  # 简单的数据增强防过拟合
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = SceneDataset(DATA_DIR, CSV_PATH, transform=train_transform)
    # 为了简单，这里就不切分验证集了，直接用全部数据训练一个过拟合的特征提取器
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # --- 构建 timm 模型 ---
    # 使用轻量级的 resnet18，修改输出层为 4 类
    print("🤖 正在加载预训练的 ConvNet 模型...")
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=4)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # --- 开始训练 ---
    print("🔥 开始训练场景分类器...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 统计准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"Loss": running_loss / total, "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    # --- 保存权重 ---
    save_path = "scene_classifier_resnet18.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 训练完成！模型已保存至: {save_path}")


# ==========================================
# 3. 独立推理函数 (识别)
# ==========================================
def predict_scene(img_path, model_path="scene_classifier_resnet18.pth"):
    """这是一个演示，告诉你未来在 JiT 推理时怎么用这个分类器"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建相同的 transform (注意：推理时用 CenterCrop 而不是 RandomCrop)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载模型
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 处理单张图片
    img = Image.open(img_path).convert("RGB")
    input_tensor = val_transform(img).unsqueeze(0).to(DEVICE)  # 增加 Batch 维度

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        class_id = torch.argmax(probabilities).item()

    print(f"🖼️ 图片: {os.path.basename(img_path)}")
    print(f"🎯 预测场景 ID: {class_id} (置信度: {probabilities[class_id] * 100:.2f}%)")
    return class_id


def batch_predict_and_save(model_path, infer_dir, scene_json):
    # ================= 配置区 =================
    # 你的测试集/验证集带雨图像所在的文件夹 (注意：这里假设是存放图片的平铺目录)
    # 如果你的测试集也有子文件夹，请告诉我，我帮你改成 os.walk 遍历
    # TEST_DIR = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Test/Drop"
    # MODEL_PATH = "scene_classifier_resnet18.pth"
    # OUTPUT_JSON = "test_scene_predictions.json"
    # ==========================================

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Scene predict with {DEVICE}")

    # 1. 初始化相同的预处理逻辑
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 2. 加载训练好的 ResNet18
    print("🤖 加载场景分类器模型权重...")
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件: {model_path}，请先运行训练脚本！")
        return

    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 3. 遍历文件夹进行预测
    predictions_dict = {}
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

    # 筛选出所有图片文件
    img_files = [f for f in os.listdir(infer_dir) if f.lower().endswith(valid_exts)]
    print(f"📂 共找到 {len(img_files)} 张测试图片，开始推理...")

    with torch.no_grad():
        for img_name in tqdm(img_files, desc="scene predict"):
            img_path = os.path.join(infer_dir, img_name)

            try:
                # 读取并预处理图片
                img = Image.open(img_path).convert("RGB")
                input_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

                # 网络前向传播
                output = model(input_tensor)
                class_id = torch.argmax(output[0]).item()

                # 存入字典 (Key 为文件名，Value 为类别 ID)
                predictions_dict[img_name] = class_id

            except Exception as e:
                print(f"⚠️ 处理图片 {img_name} 时发生错误: {e}")

    # 4. 保存为 JSON 文件
    print(f"\n💾 正在保存结果至 {scene_json}...")
    with open(scene_json, "w", encoding="utf-8") as f:
        # indent=4 可以让生成的 JSON 文件有良好的缩进，方便人类阅读
        json.dump(predictions_dict, f, indent=4)

    print("✨ 批量预测完成！")

    # 打印前 5 个预览一下
    preview = {k: predictions_dict[k] for k in list(predictions_dict.keys())[:5]}
    print(f"💡 文件内容预览: \n{json.dumps(preview, indent=4)}")

    return predictions_dict


if __name__ == "__main__":
    # 1. 先执行训练 (大概几分钟就能跑完)
    # train_classifier()

    data_root = (
        r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop"
    )
    save_path = r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2/Drop_scen_pred.json"
    batch_predict_and_save("scene_classifier_resnet18.pth", data_root, save_path)
