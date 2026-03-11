import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy

# 需要安装：pip install lpips piq
import lpips
from piq import ssim # piq 是极好用的图像质量评估库，也可以用你自带的 SSIM

# 全局初始化 LPIPS 模型，避免每次 evaluate 都重新加载消耗时间
lpips_vgg = None

def train_one_epoch(model, criterion, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    optimizer.zero_grad()

    criterion.update_weights(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (img_drop, img_clear, dummy_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = img_drop.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        y = img_clear.to(device, non_blocking=True).to(torch.float32).div_(255)
        y = y * 2.0 - 1.0

        dummy_labels.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

@torch.no_grad()
def evaluate_best_metric(model_without_ddp, data_loader_val, device):
    """
    计算综合得分: PSNR + 10*SSIM - 5*LPIPS
    得分越高，说明画质越好。
    """
    global lpips_vgg
    if lpips_vgg is None:
        # 使用 VGG 作为 LPIPS 的骨干网络 (标准做法)
        lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()

    model_without_ddp.eval()
    
    # 1. 切换到 EMA 权重进行验证
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    model_without_ddp.load_state_dict(ema_state_dict)

    total_scores = {
        'score': 0.0,
        'psnr': 0.0,
        'ssim': 0.0,
        'lpips': 0.0,
    }
    total_score = 0.0
    num_samples = 0
    print("Running composite evaluation (PSNR, SSIM, LPIPS) on validation subset...")
    for img_drop, img_clear, dummy_labels in data_loader_val:
        B, N, C, H, W = img_drop.shape
        x = img_drop.view(-1, C, H, W).to(device, non_blocking=True)
        y = img_clear.view(-1, C, H, W).to(device, non_blocking=True)
        dummy_labels = dummy_labels.to(device)
        x = x.to(torch.float32).div_(255.0) * 2.0 - 1.0
        y = y.to(torch.float32).div_(255.0) * 2.0 - 1.0
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_x = model_without_ddp.generate_i2i(x, steps=5, dummy_labels=dummy_labels)
            pred_clamp = pred_x.clamp(-1.0, 1.0)
        val_lpips = lpips_vgg(pred_clamp, y).mean().item()
        pred_01 = (pred_clamp + 1.0) / 2.0
        y_01 = (y + 1.0) / 2.0

        mse = torch.mean((pred_01 - y_01) ** 2, dim=[1, 2, 3])
        val_psnr = -10 * torch.log10(mse + 1e-8).mean().item()
        val_ssim = ssim(pred_01, y_01, data_range=1.0).item()

        composite_score = val_psnr + 10.0 * val_ssim - 5.0 * val_lpips

        total_scores['score'] += composite_score * x.size(0)
        total_scores['psnr'] += val_psnr * x.size(0)
        total_scores['ssim'] += val_ssim * x.size(0)
        total_scores['lpips'] += val_lpips * x.size(0)
        num_samples += x.size(0)

    # 计算平均得分
    total_scores['score'] = total_scores['score'] / max(1, num_samples)
    total_scores['psnr'] = total_scores['psnr'] / max(1, num_samples)
    total_scores['ssim'] = total_scores['ssim'] / max(1, num_samples)
    total_scores['lpips'] = total_scores['lpips'] / max(1, num_samples)

    # 2. 验证结束，把权重切回到在线训练状态
    model_without_ddp.load_state_dict(model_state_dict)
    model_without_ddp.train()

    # DDP 多卡同步均值
    total_scores['score'] = torch.tensor(total_scores['score']).cuda()
    total_scores['psnr'] = torch.tensor(total_scores['psnr']).cuda()
    total_scores['ssim'] = torch.tensor(total_scores['ssim']).cuda()
    total_scores['lpips'] = torch.tensor(total_scores['lpips']).cuda()
    total_scores['score'] = misc.all_reduce_mean(total_scores['score']).item()
    total_scores['psnr'] = misc.all_reduce_mean(total_scores['psnr']).item()
    total_scores['ssim'] = misc.all_reduce_mean(total_scores['ssim']).item()
    total_scores['lpips'] = misc.all_reduce_mean(total_scores['lpips']).item()

    return total_scores