import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


class EdgeLoss(nn.Module):
    """L1 distance between Sobel gradients in RGB space."""

    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        kernel_y = kernel_x.t()
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3), persistent=False)

    def _gradient(self, image):
        channels = image.shape[1]
        kernel_x = self.kernel_x.to(device=image.device, dtype=image.dtype)
        kernel_y = self.kernel_y.to(device=image.device, dtype=image.dtype)
        kernel_x = kernel_x.expand(channels, 1, 3, 3)
        kernel_y = kernel_y.expand(channels, 1, 3, 3)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return grad_x, grad_y

    def forward(self, pred, target):
        pred_x, pred_y = self._gradient(pred.float())
        target_x, target_y = self._gradient(target.float())
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class FrequencyLoss(nn.Module):
    """Log-magnitude frequency reconstruction loss.

    FFT is computed in float32 for bf16/fp16 compatibility.
    """

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        return F.l1_loss(pred_mag, target_mag)


class DynamicRaindropLoss(nn.Module):
    def __init__(
        self,
        device,
        target_w_rec=1.0,
        target_w_ssim=1.0,
        target_w_lpips=0.5,
        target_w_edge=0.0,
        target_w_freq=0.0,
        total_epochs=600,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = EdgeLoss().to(device)
        self.frequency = FrequencyLoss()

        self.lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()
        for parameter in self.lpips_vgg.parameters():
            parameter.requires_grad = False

        self.target_w_rec = float(target_w_rec)
        self.target_w_ssim = float(target_w_ssim)
        self.target_w_lpips = float(target_w_lpips)
        self.target_w_edge = float(target_w_edge)
        self.target_w_freq = float(target_w_freq)
        self.total_epochs = int(total_epochs)

        self.cur_w_rec = self.target_w_rec
        self.cur_w_ssim = self.target_w_ssim
        self.cur_w_lpips = 0.0
        self.cur_w_edge = self.target_w_edge
        self.cur_w_freq = self.target_w_freq
        self.latest_terms = {}

    def update_weights(self, current_epoch):
        """Keep the existing delayed LPIPS schedule."""
        warmup_epochs = int(self.total_epochs * 0.3)
        if current_epoch < warmup_epochs:
            self.cur_w_lpips = 0.0
        else:
            denominator = max(1, self.total_epochs - warmup_epochs)
            progress = (current_epoch - warmup_epochs) / denominator
            progress = min(max(progress, 0.0), 1.0)
            self.cur_w_lpips = self.target_w_lpips * progress

    def forward(self, pred, target):
        loss_rec = self.charbonnier(pred, target)

        pred_01 = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)
        target_01 = ((target + 1.0) / 2.0).clamp(0.0, 1.0)
        loss_ssim = 1.0 - ssim(
            pred_01.float(),
            target_01.float(),
            data_range=1.0,
            size_average=True,
        )

        if self.cur_w_lpips > 0:
            loss_lpips = self.lpips_vgg(pred.float(), target.float()).mean()
        else:
            loss_lpips = pred.new_zeros((), dtype=torch.float32)

        if self.cur_w_edge > 0:
            loss_edge = self.edge(pred, target)
        else:
            loss_edge = pred.new_zeros((), dtype=torch.float32)

        if self.cur_w_freq > 0:
            loss_freq = self.frequency(pred, target)
        else:
            loss_freq = pred.new_zeros((), dtype=torch.float32)

        total_loss = (
            self.cur_w_rec * loss_rec
            + self.cur_w_ssim * loss_ssim
            + self.cur_w_lpips * loss_lpips
            + self.cur_w_edge * loss_edge
            + self.cur_w_freq * loss_freq
        )

        # Detached values are available to the training engine for logging.
        self.latest_terms = {
            "rec": loss_rec.detach(),
            "ssim": loss_ssim.detach(),
            "lpips": loss_lpips.detach(),
            "edge": loss_edge.detach(),
            "freq": loss_freq.detach(),
        }
        return total_loss
