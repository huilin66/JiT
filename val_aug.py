import argparse
import copy
import csv
import datetime
import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F

# 导入你原项目中的模块
from dataset import SceneValPatchDatasetV2, ValPatchDataset
from denoiser import Denoiser
from engine_jit import evaluate_best_metric
from main_jit import get_args_parser


# ==========================================
# 0. TTA 模型包装器 (Test-Time Augmentation)
# ==========================================
class TTAModelWrapper(torch.nn.Module):
    """
    在不修改原模型代码的情况下，通过 Wrapper 实现 TTA (翻转、多尺度等)
    """

    def __init__(self, model, tta_config):
        super().__init__()
        self.model = model
        self.tta_config = tta_config if tta_config else {}
        self.h_flip = self.tta_config.get("h_flip", False)
        self.scales = self.tta_config.get("scales", [1.0])  # 默认原图大小

    def forward(self, x, *args, **kwargs):
        preds = []

        # 多尺度测试
        for scale in self.scales:
            if scale != 1.0:
                # 下采样/上采样输入
                x_scaled = F.interpolate(
                    x, scale_factor=scale, mode="bilinear", align_corners=False
                )
            else:
                x_scaled = x

            # 1. 正常前向传播
            out = self.model(x_scaled, *args, **kwargs)
            if scale != 1.0:
                out = F.interpolate(
                    out,
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            preds.append(out)

            # 2. 水平翻转前向传播
            if self.h_flip:
                x_flipped = torch.flip(x_scaled, dims=[3])
                out_flip = self.model(x_flipped, *args, **kwargs)
                out_flip_restored = torch.flip(out_flip, dims=[3])

                if scale != 1.0:
                    out_flip_restored = F.interpolate(
                        out_flip_restored,
                        size=(x.shape[2], x.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                preds.append(out_flip_restored)

        # 融合所有预测结果 (平均)
        final_out = torch.mean(torch.stack(preds), dim=0)
        return final_out


# ==========================================
# 1. 定义评估模型列表 (新增 TTA 策略配置)
# ==========================================
OUTPUT_DIR = "/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/output"
MODELS_TO_EVALUATE = [
    # ---- 基础组：无 TTA ----
    # {
    #     "exp_name": "Experiment_13_Seg_Base",
    #     "model": "JiT-B/16",
    #     "checkpoint": os.path.join(
    #         OUTPUT_DIR, "JiT-B-raindrop13", "16", "checkpoint-last.pth"
    #     ),
    #     "use_scene_dataset": 0,
    #     "use_bg_subnet": 1,
    #     "use_ema": True,
    #     "batch_size": 2,
    #     "tta_config": {},  # 无 TTA
    # },
    {
        "exp_name": "Experiment_22_Scene_Base",
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 1,
        "step": 10,
        "tta_config": {},  # 无 TTA
    },
    {
        "exp_name": "Experiment_22_Scene_Base",
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 1,
        "step": 20,
        "tta_config": {},  # 无 TTA
    },
    {
        "exp_name": "Experiment_22_Scene_Base",
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 1,
        "step": 2,
        "tta_config": {},  # 无 TTA
    },
    {
        "exp_name": "Experiment_22_Scene_Base",
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 1,
        "step": 1,
        "tta_config": {},  # 无 TTA
    },
    # ---- TTA 强化组：复刻你拿高分的策略 ----
    {
        "exp_name": "Experiment_22_Scene_TTA_Full",
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 1,
        "tta_config": {
            "h_flip": True,
            "scales": [0.8, 1.0, 1.25],
            "color_match": True,
            "step": 10,  # 极小步长，最高分策略
            "desc": "h flip, color match, 0.8-1.0-1.25, step 10",
        },
    },
    {
        "exp_name": "Experiment_22_Scene_TTA_Full",
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 1,
        "tta_config": {
            "h_flip": True,
            "scales": [0.8, 1.0, 1.25],
            "color_match": True,
            "step": 1,  # 极小步长，最高分策略
            "desc": "h flip, color match, 0.8-1.0-1.25, step 1",
        },
    },
]


def get_dataloader(args):
    """根据参数动态获取 DataLoader"""
    if not args.use_scene_dataset:
        dataset_val = ValPatchDataset(
            rain_dir=os.path.join(args.data_path, "Drop"),
            clean_dir=os.path.join(args.data_path, "Clear"),
        )
    else:
        dataset_val = SceneValPatchDatasetV2(
            rain_dir=os.path.join(args.data_path, "Drop"),
            clean_dir=os.path.join(args.data_path, "Clear"),
        )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    return data_loader_val


def get_path_level(input_path, level=3):
    if isinstance(input_path, str):
        p = Path(input_path)
        last_parts = p.parts[-level:]
        result = Path(*last_parts)
        return str(result)
    else:
        return None


def main():
    base_args = get_args_parser().parse_args()
    device = torch.device(base_args.device)
    base_args.data_path = (
        r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2"
    )
    summary_results = []

    print(
        f"========== Starting Batch Evaluation for {len(MODELS_TO_EVALUATE)} models ==========\n"
    )

    for idx, config in enumerate(MODELS_TO_EVALUATE):
        print(
            f"\n[{idx + 1}/{len(MODELS_TO_EVALUATE)}] Evaluating: {config['exp_name']}"
        )

        # 获取 TTA 配置
        tta_config = config.get("tta_config", {})
        tta_desc = tta_config.get("desc", "None")
        print(f"Checkpoint: {config['checkpoint']}")
        print(f"TTA Strategy: {tta_desc}")

        current_args = copy.deepcopy(base_args)
        for k, v in config.items():
            if k != "tta_config":
                setattr(current_args, k, v)

        # 将 TTA 参数（如步长和颜色匹配）透传给 args，以便 engine_jit 中的 evaluate_best_metric 使用
        if "step" in tta_config:
            current_args.step = tta_config["step"]
        if "color_match" in tta_config:
            current_args.color_match = tta_config["color_match"]

        data_loader_val = get_dataloader(current_args)

        model = Denoiser(current_args)
        model.to(device)

        if not os.path.exists(current_args.checkpoint):
            print(
                f"[WARNING] Checkpoint not found: {current_args.checkpoint}. Skipping..."
            )
            continue

        checkpoint = torch.load(current_args.checkpoint, map_location="cpu")
        if config.get("use_ema", False) and "model_ema1" in checkpoint:
            model.load_state_dict(checkpoint["model_ema1"], strict=False)
        else:
            model.load_state_dict(checkpoint["model"], strict=False)

        # ----------------------------------------------------
        # 核心改动：用 TTA Wrapper 包装模型
        # 如果当前 config 包含 h_flip 或 scales 需求，则套上 Wrapper
        # ----------------------------------------------------
        if tta_config.get("h_flip", False) or "scales" in tta_config:
            eval_model = TTAModelWrapper(model, tta_config)
        else:
            eval_model = model

        eval_model.eval()

        with torch.no_grad():
            scores = evaluate_best_metric(
                eval_model,
                data_loader_val,
                device,
                pure_val=True,
                steps=current_args.step,
            )

        summary_results.append(
            {
                "exp_name": config.get("exp_name", "Unknown"),
                "model_ckpt_name": get_path_level(config.get("checkpoint", "Unknown")),
                "use_scene_dataset": config.get("use_scene_dataset", 0),
                "use_bg_subnet": config.get("use_bg_subnet", 0),
                "tta_strategy": tta_desc,
                "val_score": f"Score:{scores.get('score', 0):.4f} (P:{scores.get('psnr', 0):.2f}|S:{scores.get('ssim', 0):.4f}|L:{scores.get('lpips', 0):.4f})",
            }
        )

        print(f"Result -> {summary_results[-1]['val_score']}")

        # 清理显存
        del model
        del eval_model
        del checkpoint
        del data_loader_val
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # 7. 将结果保存到 CSV (增加 TTA 策略列)
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"val_results_with_tta_{timestamp}.csv"

    print(f"\n=> Saving results to {csv_filename}...")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "exp_name",
                "model_ckpt name",
                "use_scene_dataset",
                "use_bg_subnet",
                "tta_strategy",
                "val_score",
                "test_score",
                "remark",
            ]
        )

        for res in summary_results:
            writer.writerow(
                [
                    res["exp_name"],
                    res["model_ckpt_name"],
                    res["use_scene_dataset"],
                    res["use_bg_subnet"],
                    res["tta_strategy"],
                    res["val_score"],
                    "",
                    "",
                ]
            )

    print("=> Evaluation finished and saved successfully!")


if __name__ == "__main__":
    main()
