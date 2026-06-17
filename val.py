import argparse
import copy
import csv
import datetime
import gc
import os
from pathlib import Path

import torch

# 导入你原项目中的模块
from dataset import SceneValPatchDatasetV2, ValPatchDataset
from denoiser import Denoiser
from engine_jit import evaluate_best_metric
from main_jit import get_args_parser

# ==========================================
# 1. 在这里定义你要批量评估的模型列表
# ==========================================
OUTPUT_DIR = "/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/output"
MODELS_TO_EVALUATE = [
    # {
    #     "exp_name": "Experiment_11_Base",
    #     "model": "JiT-B/16",
    #     "checkpoint": os.path.join(
    #         OUTPUT_DIR, "JiT-B-raindrop11", "16", "checkpoint-last.pth"
    #     ),
    #     "use_scene_dataset": 0,
    #     "use_bg_subnet": 0,
    #     "use_ema": True,
    #     "batch_size": 2,
    # },
    # {
    #     "exp_name": "Experiment_12_Scene",
    #     "model": "JiT-B/16",
    #     "checkpoint": os.path.join(
    #         OUTPUT_DIR, "JiT-B-raindrop12", "16", "checkpoint-last.pth"
    #     ),
    #     "use_scene_dataset": 1,
    #     "use_bg_subnet": 0,
    #     "use_ema": False,
    #     "batch_size": 2,
    # },
    # {
    #     "exp_name": "Experiment_13_Seg",
    #     "model": "JiT-B/16",
    #     "checkpoint": os.path.join(
    #         OUTPUT_DIR, "JiT-B-raindrop13", "16", "checkpoint-last.pth"
    #     ),
    #     "use_scene_dataset": 0,
    #     "use_bg_subnet": 1,
    #     "use_ema": True,
    #     "batch_size": 2,
    # },
    # {
    #     "exp_name": "Experiment_14_Scene_Seg",
    #     "model": "JiT-B/16",
    #     "checkpoint": os.path.join(
    #         OUTPUT_DIR, "JiT-B-raindrop14", "16", "checkpoint-last.pth"
    #     ),
    #     "use_scene_dataset": 1,
    #     "use_bg_subnet": 1,
    #     "use_ema": False,
    #     "batch_size": 2,
    # },
    # {
    #     "exp_name": "Experiment_22_Scene",
    #     "model": "JiT-H/16",
    #     "checkpoint": os.path.join(
    #         OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
    #     ),
    #     "use_scene_dataset": 1,
    #     "use_bg_subnet": 0,
    #     "use_ema": False,
    #     "batch_size": 2,
    # },
    {
        "exp_name": "Experiment_24_Scene_Seg",
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop24", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 1,
        "use_ema": False,
        "batch_size": 20,
    },
    # 按这个格式继续添加你需要测试的模型...
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
    base_args.data_path = r'/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2'
    summary_results = []

    print(
        f"========== Starting Batch Evaluation for {len(MODELS_TO_EVALUATE)} models ==========\n"
    )

    for idx, config in enumerate(MODELS_TO_EVALUATE):
        print(
            f"\n[{idx + 1}/{len(MODELS_TO_EVALUATE)}] Evaluating: {config['exp_name']}"
        )
        print(f"Checkpoint: {config['checkpoint']}")

        current_args = copy.deepcopy(base_args)
        for k, v in config.items():
            setattr(current_args, k, v)

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

        model.eval()

        with torch.no_grad():
            scores = evaluate_best_metric(model, data_loader_val, device, pure_val=True)

        # 记录所需的所有信息
        summary_results.append(
            {
                "exp_name": config.get("exp_name", "Unknown"),
                "model_ckpt_name": get_path_level(config.get("checkpoint", "Unknown")),
                "use_scene_dataset": config.get("use_scene_dataset", 0),
                "use_bg_subnet": config.get("use_bg_subnet", 0),
                # 将核心分数与具体指标拼接成一个字符串，方便在单列中查看
                "val_score": f"Score:{scores.get('score', 0):.4f} (P:{scores.get('psnr', 0):.2f}|S:{scores.get('ssim', 0):.4f}|L:{scores.get('lpips', 0):.4f})",
            }
        )

        print(f"Result -> {summary_results[-1]['val_score']}")

        # 清理显存
        del model
        del checkpoint
        del data_loader_val
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # 7. 将结果保存到 CSV
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"val_results_{timestamp}.csv"

    print(f"\n=> Saving results to {csv_filename}...")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(
            [
                "exp_name",
                "model_ckpt name",
                "use_scene_dataset",
                "use_bg_subnet",
                "val_score",
                "test_score",
                "remark",
            ]
        )

        # 写入数据
        for res in summary_results:
            writer.writerow(
                [
                    res["exp_name"],
                    res["model_ckpt_name"],
                    res["use_scene_dataset"],
                    res["use_bg_subnet"],
                    res["val_score"],
                    "",  # test_score 留白
                    "",  # remark 留白
                ]
            )

    print("=> Evaluation finished and saved successfully!")


if __name__ == "__main__":
    main()
