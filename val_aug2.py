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
# 1. 直接引入你写好的高阶推理引擎
# ==========================================
class EnhancedInferencer:
    def __init__(
        self,
        models,
        model_weights=None,
        use_h_flip=True,
        scales=[1.0],
        scale_weights=None,
        use_color_match=True,
        patch_size=256,
        stride=128,
    ):
        self.models = models if isinstance(models, list) else [models]
        self.model_weights = model_weights or [1.0 / len(self.models)] * len(
            self.models
        )
        self.use_h_flip = use_h_flip
        self.scales = scales
        if scale_weights is None:
            self.scale_weights = [1.0 if s == 1.0 else 0.5 for s in scales]
        else:
            self.scale_weights = scale_weights

        sum_mw = sum(self.model_weights)
        self.model_weights = [w / sum_mw for w in self.model_weights]
        sum_sw = sum(self.scale_weights)
        self.scale_weights = [w / sum_sw for w in self.scale_weights]

        self.use_color_match = use_color_match
        self.patch_size = patch_size
        self.stride = stride

    def _color_luminance_match(self, orig_tensor, pred_tensor, blend_ratio=0.5):
        mu_orig = orig_tensor.mean(dim=(2, 3), keepdim=True)
        std_orig = orig_tensor.std(dim=(2, 3), keepdim=True) + 1e-6
        mu_pred = pred_tensor.mean(dim=(2, 3), keepdim=True)
        std_pred = pred_tensor.std(dim=(2, 3), keepdim=True) + 1e-6

        matched = (pred_tensor - mu_pred) / std_pred * std_orig + mu_orig
        out = (1.0 - blend_ratio) * pred_tensor + blend_ratio * matched
        return torch.clamp(out, -1.0, 1.0)

    def _core_patch_inference(self, model, img_tensor, step_num, dummy_labels):
        _, _, H, W = img_tensor.shape

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
        batch_tensor = torch.cat(patches, dim=0)

        if dummy_labels is not None:
            expanded_labels = dummy_labels.expand(batch_tensor.shape[0])
        else:
            expanded_labels = None

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
        return torch.clamp(final_output, -1.0, 1.0)

    def forward(self, img_tensor, step_num, dummy_labels):
        _, _, orig_H, orig_W = img_tensor.shape
        final_ensemble_output = torch.zeros_like(img_tensor)

        for model, model_w in zip(self.models, self.model_weights):
            model_output = torch.zeros_like(img_tensor)
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

                scale_pred_accum += self._core_patch_inference(
                    model, scaled_img, step_num, dummy_labels
                )
                tta_count += 1

                if self.use_h_flip:
                    flipped_img = torch.flip(scaled_img, dims=[3])
                    flipped_pred = self._core_patch_inference(
                        model, flipped_img, step_num, dummy_labels
                    )
                    scale_pred_accum += torch.flip(flipped_pred, dims=[3])
                    tta_count += 1

                scale_pred = scale_pred_accum / tta_count

                if scale != 1.0:
                    scale_pred = F.interpolate(
                        scale_pred,
                        size=(orig_H, orig_W),
                        mode="bicubic",
                        align_corners=False,
                    )

                model_output += scale_pred * scale_w
            final_ensemble_output += model_output * model_w

        if self.use_color_match:
            final_ensemble_output = self._color_luminance_match(
                img_tensor, final_ensemble_output
            )

        return final_ensemble_output


# ==========================================
# 2. 接口适配器 (让 evaluate_best_metric 能够调用)
# ==========================================
class TTAValidationAdapter(torch.nn.Module):
    """桥接器：将标准的 model(img, labels) 调用路由到 EnhancedInferencer"""

    def __init__(self, inferencer, step_num):
        super().__init__()
        self.inferencer = inferencer
        self.step_num = step_num

    def forward(self, img, dummy_labels=None, **kwargs):
        # 兼容部分验证集未返回 labels 的情况
        return self.inferencer.forward(img, self.step_num, dummy_labels)


# ==========================================
# 3. 评估配置列表
# ==========================================
OUTPUT_DIR = "/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/output"

MODELS_TO_EVALUATE = [
    {
        "exp_name": "Experiment_22_Scene_TTA_Step10",  # 改了名字方便在 CSV 中区分
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 32,
        "tta_config": {
            "use_h_flip": True,
            "scales": [0.8, 1.0, 1.25],
            "use_color_match": True,
            "step_num": 10,  # 对照组：默认步长
            "desc": "h flip, color match, 0.8-1.0-1.25, step 10",
        },
    },
    {
        "exp_name": "Experiment_22_Scene_TTA_Step1",  # 改了名字方便在 CSV 中区分
        "model": "JiT-H/16",
        "checkpoint": os.path.join(
            OUTPUT_DIR, "JiT-H-raindrop22", "16", "checkpoint-last.pth"
        ),
        "use_scene_dataset": 1,
        "use_bg_subnet": 0,
        "use_ema": False,
        "batch_size": 32,
        "tta_config": {
            "use_h_flip": True,
            "scales": [0.8, 1.0, 1.25],
            "use_color_match": True,
            "step_num": 1,  # 实验组：极小步长，最高分策略
            "desc": "h flip, color match, 0.8-1.0-1.25, step 1",
        },
    },
]


def get_dataloader(args):
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
        return str(Path(*last_parts))
    return None


def main():
    base_args = get_args_parser().parse_args()
    device = torch.device("cuda:1")
    base_args.data_path = (
        r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2"
    )
    summary_results = []

    print(
        f"========== Starting Batch Evaluation for {len(MODELS_TO_EVALUATE)} models ==========\n"
    )

    for idx, config in enumerate(MODELS_TO_EVALUATE):
        tta_cfg = config.get("tta_config", {})
        tta_desc = tta_cfg.get("desc", "None")
        print(
            f"\n[{idx + 1}/{len(MODELS_TO_EVALUATE)}] Evaluating: {config['exp_name']}"
        )
        print(f"TTA Strategy: {tta_desc}")

        current_args = copy.deepcopy(base_args)
        for k, v in config.items():
            if k != "tta_config":
                setattr(current_args, k, v)

        data_loader_val = get_dataloader(current_args)
        model = Denoiser(current_args)
        model.to(device)

        if not os.path.exists(current_args.checkpoint):
            print(f"[WARNING] Checkpoint not found: {current_args.checkpoint}")
            continue

        checkpoint = torch.load(current_args.checkpoint, map_location="cpu")
        if config.get("use_ema", False) and "model_ema1" in checkpoint:
            model.load_state_dict(checkpoint["model_ema1"], strict=False)
        else:
            model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()

        # 初始化你写的高阶推理引擎
        inferencer = EnhancedInferencer(
            models=model,
            use_h_flip=tta_cfg.get("use_h_flip", False),
            scales=tta_cfg.get("scales", [1.0]),
            use_color_match=tta_cfg.get("use_color_match", False),
            patch_size=256,
            stride=128,
        )

        # 使用 Adapter 包装引擎，使其接口符合 evaluate_best_metric 的期望
        step_num = tta_cfg.get("step_num", 10)
        eval_model = TTAValidationAdapter(inferencer, step_num)

        with torch.no_grad():
            scores = evaluate_best_metric(
                eval_model,
                data_loader_val,
                device,
                pure_val=True,
                steps=step_num,
            )

        res_str = f"Score:{scores.get('score', 0):.4f} (P:{scores.get('psnr', 0):.2f}|S:{scores.get('ssim', 0):.4f}|L:{scores.get('lpips', 0):.4f})"
        summary_results.append(
            {
                "exp_name": config.get("exp_name", "Unknown"),
                "model_ckpt_name": get_path_level(config.get("checkpoint", "Unknown")),
                "use_scene_dataset": config.get("use_scene_dataset", 0),
                "use_bg_subnet": config.get("use_bg_subnet", 0),
                "tta_strategy": tta_desc,
                "val_score": res_str,
            }
        )
        print(f"Result -> {res_str}")

        del model
        del eval_model
        del inferencer
        del checkpoint
        del data_loader_val
        gc.collect()
        torch.cuda.empty_cache()

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
                ]
            )


if __name__ == "__main__":
    main()
