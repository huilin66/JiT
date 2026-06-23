import copy
import math
import os
import shutil
import sys

import cv2

# 需要安装：pip install lpips piq
import lpips
import numpy as np
import torch
import torch_fidelity
from piq import ssim  # piq 是极好用的图像质量评估库，也可以用你自带的 SSIM
from tqdm import tqdm

import util.lr_sched as lr_sched
import util.misc as misc

# 全局初始化 LPIPS 模型，避免每次 evaluate 都重新加载消耗时间
lpips_vgg = None


def _to_model_range(x, device):
    x = x.to(device, non_blocking=True).to(torch.float32)
    if x.max() > 2.0:
        x = x.div(255.0)
    return x * 2.0 - 1.0


def _rgb_to_y(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def train_one_epoch(
    model,
    criterion,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    criterion.update_weights(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (img_drop, img_clear, dummy_labels) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )

        # normalize image to [-1, 1]
        x = _to_model_range(img_drop, device)
        y = _to_model_range(img_clear, device)
        dummy_labels = dummy_labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_pred = model(x, y, dummy_labels)

        loss = criterion(y_pred, y)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        if hasattr(criterion, "latest_terms"):
            for term_name, term_value in criterion.latest_terms.items():
                metric_logger.update(**{f"loss_{term_name}": float(term_value.item())})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)
                if hasattr(criterion, "latest_terms"):
                    for term_name, term_value in criterion.latest_terms.items():
                        log_writer.add_scalar(
                            f"train_loss_{term_name}",
                            float(term_value.item()),
                            epoch_1000x,
                        )

        if args.max_train_steps > 0 and data_iter_step + 1 >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}; stopping epoch early.")
            break


@torch.no_grad()
def evaluate_best_metric(
    model_without_ddp, data_loader_val, device, pure_val=False, steps=10
):
    """
    计算综合得分: PSNR + 10*SSIM - 5*LPIPS
    得分越高，说明画质越好。
    """
    global lpips_vgg
    if lpips_vgg is None:
        # 使用 VGG 作为 LPIPS 的骨干网络 (标准做法)
        lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    model_without_ddp.eval()

    if not pure_val:
        # 1. 切换到 EMA 权重进行验证
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = model_without_ddp.ema_params1[i]
        model_without_ddp.load_state_dict(ema_state_dict)

    total_scores = {
        "score": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }
    num_samples = 0
    print("Running composite evaluation (PSNR, SSIM, LPIPS) on validation subset...")
    for img_drop, img_clear, dummy_labels in tqdm(data_loader_val):
        B, N, C, H, W = img_drop.shape
        x = _to_model_range(img_drop.view(-1, C, H, W), device)
        y = _to_model_range(img_clear.view(-1, C, H, W), device)
        dummy_labels = dummy_labels.view(-1).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_x = model_without_ddp.generate_i2i(
                x, steps=steps, dummy_labels=dummy_labels
            )
            pred_clamp = pred_x.clamp(-1.0, 1.0)
        val_lpips = lpips_vgg(pred_clamp, y).mean().item()
        pred_01 = (pred_clamp + 1.0) / 2.0
        y_01 = (y + 1.0) / 2.0
        pred_y = _rgb_to_y(pred_01)
        target_y = _rgb_to_y(y_01)

        mse = torch.mean((pred_y - target_y) ** 2, dim=[1, 2, 3])
        val_psnr = -10 * torch.log10(mse + 1e-8).mean().item()
        val_ssim = ssim(pred_y, target_y, data_range=1.0).item()

        composite_score = val_psnr + 10.0 * val_ssim - 5.0 * val_lpips

        total_scores["score"] += composite_score * x.size(0)
        total_scores["psnr"] += val_psnr * x.size(0)
        total_scores["ssim"] += val_ssim * x.size(0)
        total_scores["lpips"] += val_lpips * x.size(0)
        num_samples += x.size(0)

    # 2. 验证结束，把权重切回到在线训练状态
    if not pure_val:
        model_without_ddp.load_state_dict(model_state_dict)
        model_without_ddp.train()

    if misc.is_dist_avail_and_initialized():
        stats = torch.tensor(
            [
                total_scores["score"],
                total_scores["psnr"],
                total_scores["ssim"],
                total_scores["lpips"],
                float(num_samples),
            ],
            device=device,
        )
        torch.distributed.all_reduce(stats)
        total_scores["score"] = stats[0].item()
        total_scores["psnr"] = stats[1].item()
        total_scores["ssim"] = stats[2].item()
        total_scores["lpips"] = stats[3].item()
        num_samples = int(stats[4].item())

    denom = max(1, num_samples)
    total_scores["score"] = total_scores["score"] / denom
    total_scores["psnr"] = total_scores["psnr"] / denom
    total_scores["ssim"] = total_scores["ssim"] / denom
    total_scores["lpips"] = total_scores["lpips"] / denom

    return total_scores
