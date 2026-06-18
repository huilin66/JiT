import lpips
import torch
import torch.nn as nn
from pytorch_msssim import ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


class DynamicRaindropLoss(nn.Module):
    def __init__(
        self,
        device,
        target_w_rec=1.0,
        target_w_ssim=1.0,
        target_w_lpips=0.5,
        total_epochs=600,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss()

        # 冻结 VGG 权重
        self.lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()
        for param in self.lpips_vgg.parameters():
            param.requires_grad = False

        # 记录目标权重 (注意：这里我把 SSIM 和 LPIPS 的权重按比例缩小了，防止梯度爆炸)
        # 你依然保持了 10(SSIM) : 5(LPIPS) 的相对比例，即 1.0 : 0.5
        self.target_w_rec = target_w_rec
        self.target_w_ssim = target_w_ssim
        self.target_w_lpips = target_w_lpips
        self.total_epochs = total_epochs

        # 当前实时权重
        self.cur_w_rec = target_w_rec
        self.cur_w_ssim = target_w_ssim
        self.cur_w_lpips = 0.0  # LPIPS 初始为 0！

    def update_weights(self, current_epoch):
        """核心调参逻辑：根据当前 epoch 动态更新 LPIPS 权重"""
        warmup_epochs = int(self.total_epochs * 0.3)  # 前 30% epoch 不用 LPIPS

        if current_epoch < warmup_epochs:
            self.cur_w_lpips = 0.0
        else:
            # 过了 30% 之后，LPIPS 权重从 0 线性增长到目标值，防止 Loss 突变
            progress = (current_epoch - warmup_epochs) / (
                self.total_epochs - warmup_epochs
            )
            self.cur_w_lpips = self.target_w_lpips * progress

        # 打印一下，方便你在控制台监控
        # print(f"[Epoch {current_epoch}] Loss Weights -> Rec: {self.cur_w_rec}, SSIM: {self.cur_w_ssim}, LPIPS: {self.cur_w_lpips:.3f}")

    def forward(self, pred, target):
        loss_rec = self.charbonnier(pred, target)

        pred_01 = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)
        target_01 = ((target + 1.0) / 2.0).clamp(0.0, 1.0)
        loss_ssim = 1.0 - ssim(
            pred_01.float(), target_01.float(), data_range=1.0, size_average=True
        )

        # 只有当 LPIPS 权重 > 0 时才计算前向传播，节省前期训练 30% 的显存和时间！
        if self.cur_w_lpips > 0:
            loss_lpips = self.lpips_vgg(pred, target).mean()
        else:
            loss_lpips = torch.tensor(0.0, device=pred.device)

        total_loss = (
            (self.cur_w_rec * loss_rec)
            + (self.cur_w_ssim * loss_ssim)
            + (self.cur_w_lpips * loss_lpips)
        )

        return total_loss
