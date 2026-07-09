import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
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


class DeepLPIPSLoss(nn.Module):
    """Multi-layer VGG feature matching loss (deep perceptual loss).

    Instead of a single LPIPS score, this computes L1 distance across multiple
    VGG feature layers, which preserves local texture better than scalar LPIPS.
    Layer weights are learned from LPIPS' own calibration but can be scaled per
    layer to control coarse vs fine texture emphasis.
    """

    # Default weights: finer layers (early VGG) get higher weight for texture.
    DEFAULT_LAYER_WEIGHTS = {
        "relu1_2": 1.0,
        "relu2_2": 0.8,
        "relu3_3": 0.5,
    }

    def __init__(self, layers=("relu1_2", "relu2_2", "relu3_3"), device="cuda"):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Map layer names to VGG feature indices.
        self.layers = list(layers)
        layer_indices = {"relu1_2": 4, "relu2_2": 9, "relu3_3": 16, "relu4_3": 23}
        max_index = max(layer_indices[name] for name in self.layers if name in layer_indices)
        self.slices = {}
        prev_index = 0
        for i in range(max_index + 1):
            module = vgg[i]
            if hasattr(module, "weight"):
                module.requires_grad_(False)
            if i in layer_indices.values():
                name = [k for k, v in layer_indices.items() if v == i][0]
                self.slices[name] = nn.Sequential(
                    *list(vgg[prev_index : i + 1])
                )
                prev_index = i + 1

        self.net = nn.ModuleDict(self.slices)
        self.net.to(device).eval()

        # Per-layer L1 scale from LPIPS pre-trained weights (approximate).
        self.register_buffer(
            "layer_scale",
            torch.tensor(
                [self.DEFAULT_LAYER_WEIGHTS.get(name, 0.5) for name in self.layers],
                dtype=torch.float32,
            ),
        )
        # Normalizer: VGG mean/std for [-1,1] input after ImageNet normalization.
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("vgg_mean", vgg_mean)
        self.register_buffer("vgg_std", vgg_std)

    def _normalize(self, x):
        # x is in [-1, 1]; convert to [0,1] then apply ImageNet normalization.
        x = (x + 1.0) / 2.0
        return (x - self.vgg_mean.to(x.device)) / self.vgg_std.to(x.device)

    def forward(self, pred, target):
        pred = self._normalize(pred.float())
        target = self._normalize(target.float())

        total = pred.new_zeros((), dtype=torch.float32)
        for idx, name in enumerate(self.layers):
            if name not in self.net:
                continue
            feat_pred = self.net[name](pred)
            feat_target = self.net[name](target)
            # Normalize by spatial dimensions to make layers comparable.
            spatial_dim = feat_pred.shape[2] * feat_pred.shape[3]
            layer_loss = F.l1_loss(feat_pred, feat_target) / float(spatial_dim)
            total = total + self.layer_scale[idx].to(pred.device) * layer_loss
        return total


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
        use_deep_lpips=False,
        deep_lpips_layers="relu1_2,relu2_2,relu3_3",
        lpips_warmup_ratio=0.3,
        pseudo_loss_weight=0.25,
    ):
        super().__init__()
        self.pseudo_loss_weight = float(pseudo_loss_weight)
        self.charbonnier = CharbonnierLoss()
        self.edge = EdgeLoss().to(device)
        self.frequency = FrequencyLoss()

        self.use_deep_lpips = bool(use_deep_lpips)
        self.lpips_warmup_ratio = float(lpips_warmup_ratio)

        if self.use_deep_lpips:
            layers = tuple(l.strip() for l in deep_lpips_layers.split(",") if l.strip())
            self.deep_lpips = DeepLPIPSLoss(layers=layers, device=device)
            self.lpips_vgg = None
        else:
            self.deep_lpips = None
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
        """LPIPS schedule: from epoch 0 if warmup_ratio=0, else delayed as before."""
        warmup_epochs = int(self.total_epochs * self.lpips_warmup_ratio)
        if warmup_epochs <= 0 or current_epoch < warmup_epochs:
            self.cur_w_lpips = 0.0 if warmup_epochs > 0 else self.target_w_lpips
        else:
            denominator = max(1, self.total_epochs - warmup_epochs)
            progress = (current_epoch - warmup_epochs) / denominator
            progress = min(max(progress, 0.0), 1.0)
            self.cur_w_lpips = self.target_w_lpips * progress

    def forward(self, pred, target, is_pseudo=None, x_rainy=None, mask=None):
        """Compute loss. When is_pseudo and mask are provided, applies mask-weighted
        loss to pseudo samples and standard loss to real samples.

        Args:
            pred: model prediction [B, 3, H, W] in [-1, 1]
            target: ground truth (real or pseudo) [B, 3, H, W] in [-1, 1]
            is_pseudo: [B, 1] float tensor, 1=pseudo, 0=real. None means all real.
            x_rainy: [B, 3, H, W] rainy input in [-1, 1] (needed for pseudo loss).
            mask: [B, 1, H, W] restoration mask in [0, 1] for pseudo samples.
        """
        if is_pseudo is None or not is_pseudo.any():
            return self._forward_standard(pred, target)

        # Split real and pseudo samples.
        is_pseudo_bool = is_pseudo.squeeze(-1) > 0.5
        is_real_bool = ~is_pseudo_bool
        total = pred.new_zeros((), dtype=torch.float32)
        self.latest_terms = {}

        if is_real_bool.any():
            loss_real = self._forward_standard(pred[is_real_bool], target[is_real_bool])
            real_terms = {k: v.detach() for k, v in self.latest_terms.items()}
            total = total + loss_real
        else:
            loss_real = pred.new_zeros((), dtype=torch.float32)
            real_terms = {}

        if is_pseudo_bool.any():
            loss_pseudo = self._forward_pseudo(
                pred[is_pseudo_bool],
                target[is_pseudo_bool],
                x_rainy[is_pseudo_bool],
                mask[is_pseudo_bool],
            )
            pseudo_terms = {k: v.detach() for k, v in self.latest_terms.items()}
            if is_real_bool.any():
                total = total + self.pseudo_loss_weight * loss_pseudo
            else:
                total = total + loss_pseudo
        else:
            loss_pseudo = pred.new_zeros((), dtype=torch.float32)
            pseudo_terms = {}

        self.latest_terms = {}
        self.latest_terms.update(
            {f"real_{k}": v for k, v in real_terms.items()}
        )
        self.latest_terms.update(
            {f"pseudo_{k}": v for k, v in pseudo_terms.items()}
        )
        self.latest_terms["real_total"] = loss_real.detach()
        self.latest_terms["pseudo_total"] = loss_pseudo.detach()
        return total

    def _forward_standard(self, pred, target):
        """Standard loss for real GT pairs."""
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
            if self.use_deep_lpips:
                loss_lpips = self.deep_lpips(pred.float(), target.float())
            else:
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

        self.latest_terms = {
            "rec": loss_rec.detach(),
            "ssim": loss_ssim.detach(),
            "lpips": loss_lpips.detach(),
            "edge": loss_edge.detach(),
            "freq": loss_freq.detach(),
        }
        return total_loss

    def _forward_pseudo(self, pred, pseudo, x_rainy, mask):
        """Mask-weighted pseudo loss.

        L_pseudo = mask * Charbonnier(pred, pseudo)
                 + 0.05 * mask * edge(pred, pseudo)
                 + 0.05 * (1 - mask) * edge(pred, input)
        """
        # Per-pixel Charbonnier weighted by mask.
        diff = torch.sqrt((pred - pseudo) ** 2 + 1e-3 ** 2)
        loss_rec = (mask * diff).mean()

        loss_edge_pseudo = pred.new_zeros((), dtype=torch.float32)
        loss_edge_input = pred.new_zeros((), dtype=torch.float32)

        if self.cur_w_edge > 0 or True:  # Always compute edge for pseudo (per spec).
            # Edge loss in masked (rain) regions.
            loss_edge_pseudo = self._masked_edge_loss(pred, pseudo, mask)
            # Edge loss in non-masked (background) regions — prevent texture flattening.
            inv_mask = 1.0 - mask
            if inv_mask.mean() > 0.01:
                loss_edge_input = self._masked_edge_loss(pred, x_rainy, inv_mask)

        total = loss_rec + 0.05 * loss_edge_pseudo + 0.05 * loss_edge_input
        self.latest_terms = {
            "rec": loss_rec.detach(),
            "edge_pseudo": loss_edge_pseudo.detach(),
            "edge_input": loss_edge_input.detach(),
        }
        return total

    def _masked_edge_loss(self, pred, target, mask):
        """Edge loss (Sobel gradient L1) weighted by mask."""
        pred_x, pred_y = self.edge._gradient(pred.float())
        target_x, target_y = self.edge._gradient(target.float())
        # Resize mask to match gradient dimensions if needed.
        if mask.shape[-2:] != pred_x.shape[-2:]:
            mask_resized = F.interpolate(
                mask, size=pred_x.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            mask_resized = mask
        mask_resized = mask_resized.to(pred_x.dtype)
        loss = mask_resized * (F.l1_loss(pred_x, target_x, reduction="none")
                               + F.l1_loss(pred_y, target_y, reduction="none"))
        return loss.mean()
