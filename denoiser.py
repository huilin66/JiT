import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.task = getattr(args, "task", "generation")
        self.is_restoration = self.task == "restoration"
        self.condition_mode = getattr(args, "condition_mode", "concat")
        if self.condition_mode != "concat":
            raise NotImplementedError("Only concat conditioning is currently supported.")

        net_in_channels = 6 if self.is_restoration else 3
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=net_in_channels,
            out_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.restoration_bridge = getattr(args, "restoration_bridge", "condition")
        self.condition_noise_scale = getattr(args, "condition_noise_scale", 0.0)
        self.predict_residual = getattr(args, "predict_residual", True)
        self.recon_l1_weight = getattr(args, "recon_l1_weight", 0.0)
        self.residual_l1_weight = getattr(args, "residual_l1_weight", 0.0)
        self.ssim_loss_weight = getattr(args, "ssim_loss_weight", 0.0)
        self.gradient_loss_weight = getattr(args, "gradient_loss_weight", 0.0)

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        if self.label_drop_prob <= 0:
            return labels
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _model_input(self, z, condition=None):
        if condition is None:
            return z
        return torch.cat([z, condition], dim=1)

    def _predict_clean(self, z, t, labels, condition=None):
        x_pred = self.net(self._model_input(z, condition), t.flatten(), labels)
        if self.is_restoration and condition is not None and self.predict_residual:
            x_pred = condition + x_pred
        return x_pred

    def _sample_source(self, clean, condition):
        if self.restoration_bridge == "condition":
            source = condition
            if self.condition_noise_scale > 0:
                source = source + torch.randn_like(source) * self.condition_noise_scale
            return source
        if self.restoration_bridge == "noise":
            return torch.randn_like(clean) * self.noise_scale
        raise NotImplementedError(f"Unknown restoration bridge: {self.restoration_bridge}")

    @staticmethod
    def _ssim_loss(x, y):
        x = ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        y = ((y.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        sigma_x = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        return (1.0 - ssim).clamp(0.0, 2.0).mean()

    @staticmethod
    def _gradient_loss(x, y):
        x = x.float()
        y = y.float()
        channels = x.shape[1]
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1) / 8.0
        kernel_y = kernel_x.transpose(2, 3)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
        target_grad_x = F.conv2d(y, kernel_x, padding=1, groups=channels)
        target_grad_y = F.conv2d(y, kernel_y, padding=1, groups=channels)
        return F.l1_loss(grad_x, target_grad_x) + F.l1_loss(grad_y, target_grad_y)

    def forward(self, x, labels, condition=None):
        if condition is not None:
            return self.forward_restoration(x, labels, condition)

        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self._predict_clean(z, t, labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    def forward_restoration(self, clean, labels, condition):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(clean.size(0), device=clean.device).view(-1, *([1] * (clean.ndim - 1)))
        source = self._sample_source(clean, condition)

        z = t * clean + (1 - t) * source
        v = (clean - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self._predict_clean(z, t, labels_dropped, condition)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        loss = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        if self.recon_l1_weight > 0:
            loss = loss + self.recon_l1_weight * F.l1_loss(x_pred, clean)
        if self.residual_l1_weight > 0:
            loss = loss + self.residual_l1_weight * F.l1_loss(x_pred - condition, clean - condition)
        if self.ssim_loss_weight > 0:
            loss = loss + self.ssim_loss_weight * self._ssim_loss(x_pred, clean)
        if self.gradient_loss_weight > 0:
            loss = loss + self.gradient_loss_weight * self._gradient_loss(x_pred, clean)

        return loss

    @torch.no_grad()
    def generate(self, labels, condition=None):
        device = labels.device
        bsz = labels.size(0)

        if condition is not None:
            if self.restoration_bridge == "condition":
                z = condition.clone()
            elif self.restoration_bridge == "noise":
                z = self.noise_scale * torch.randn_like(condition)
            else:
                raise NotImplementedError(f"Unknown restoration bridge: {self.restoration_bridge}")
        else:
            z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)

        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, condition)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, condition)
        return z

    @torch.no_grad()
    def restore(self, condition, labels=None):
        if labels is None:
            labels = torch.zeros(condition.size(0), dtype=torch.long, device=condition.device)
        return self.generate(labels, condition=condition)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, condition=None):
        # conditional
        x_cond = self._predict_clean(z, t, labels, condition)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        if self.cfg_scale == 1.0 or self.label_drop_prob <= 0:
            return v_cond

        # unconditional
        x_uncond = self._predict_clean(z, t, torch.full_like(labels, self.num_classes), condition)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, condition=None):
        v_pred = self._forward_sample(z, t, labels, condition)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, condition=None):
        v_pred_t = self._forward_sample(z, t, labels, condition)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, condition)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
