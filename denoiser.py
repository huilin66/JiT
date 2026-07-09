import torch
import torch.nn as nn
from model_jit import JiT_models
from model_msdt_refiner import MSDTDetailRefiner

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            use_bg_subnet = args.use_bg_subnet,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.use_detail_refiner = bool(getattr(args, "use_detail_refiner", 0))
        self.freeze_jit = bool(getattr(args, "freeze_jit", 0))
        self.unfreeze_jit_last_blocks = int(getattr(args, "unfreeze_jit_last_blocks", 0))
        if self.freeze_jit and not self.use_detail_refiner:
            raise ValueError("--freeze_jit 1 requires --use_detail_refiner 1")
        if self.unfreeze_jit_last_blocks > 0 and not self.use_detail_refiner:
            raise ValueError("--unfreeze_jit_last_blocks requires --use_detail_refiner 1")
        if self.unfreeze_jit_last_blocks > 0 and self.freeze_jit:
            raise ValueError(
                "--unfreeze_jit_last_blocks > 0 is incompatible with --freeze_jit 1. "
                "Use --freeze_jit 0 and --unfreeze_jit_last_blocks N instead."
            )
        if self.use_detail_refiner and not self.freeze_jit and self.unfreeze_jit_last_blocks <= 0:
            raise ValueError(
                "Joint refiner training requires --unfreeze_jit_last_blocks N. "
                "Full JiT fine-tuning is intentionally disabled by default."
            )
        if self.use_detail_refiner:
            self.detail_refiner = MSDTDetailRefiner(
                in_channels=3,
                base_dim=getattr(args, "refiner_base_dim", 32),
                num_blocks=getattr(args, "refiner_num_blocks", 2),
                use_frequency=bool(getattr(args, "refiner_use_frequency", 1)),
                max_residual=getattr(args, "refiner_max_residual", 0.25),
            )
        else:
            self.detail_refiner = None

        if self.freeze_jit:
            self.net.requires_grad_(False)
            self.net.eval()
        elif self.unfreeze_jit_last_blocks > 0:
            # Freeze all JiT blocks first, then selectively unfreeze the last N.
            self.net.requires_grad_(True)
            self.net.eval()  # BN/ dropout in eval mode for stability
            total_blocks = len(self.net.blocks)
            freeze_until = max(0, total_blocks - self.unfreeze_jit_last_blocks)
            for i, block in enumerate(self.net.blocks):
                if i < freeze_until:
                    block.requires_grad_(False)
            # Keep patch embed, pos embed, time/class embedders frozen.
            self.net.x_embedder.requires_grad_(False)
            self.net.t_embedder.requires_grad_(False)
            self.net.y_embedder.requires_grad_(False)
            # Also freeze final_layer for safety (it was zero-initialized).
            self.net.final_layer.requires_grad_(False)
            if hasattr(self.net, 'bg_subnet') and self.net.use_bg_subnet:
                self.net.bg_subnet.requires_grad_(False)
            frozen_count = sum(p.numel() for p in self.net.parameters() if not p.requires_grad)
            trainable_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            print(
                f"[Denoiser] Partial JiT unfreeze: last {self.unfreeze_jit_last_blocks}/"
                f"{total_blocks} blocks trainable ({trainable_count} trainable / "
                f"{frozen_count} frozen params in JiT)"
            )

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

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

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_jit:
            # The frozen JiT must produce a stable ODE endpoint while the
            # refiner is training, even when the enclosing Denoiser is in train mode.
            self.net.eval()
        elif self.unfreeze_jit_last_blocks > 0:
            # Partial unfreeze: JiT blocks that are trainable stay in train mode,
            # but frozen portions (embedders, early blocks) stay in eval mode.
            # We let super().train(mode) activate everything, then selectively
            # revert frozen blocks to eval.
            pass  # Individual blocks already have requires_grad=False where needed.
        return self

    def _prepare_labels(self, x, dummy_labels):
        if dummy_labels is None:
            return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return dummy_labels.squeeze(-1).view(-1).to(device=x.device, dtype=torch.long)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x_rainy, x_clean, dummy_labels=None):
        """
        Training forward supporting three modes:
          - JiT-only rectified flow (use_detail_refiner=0)
          - Refiner-only: frozen JiT ODE -> refiner (use_detail_refiner=1, freeze_jit=1)
          - Joint: JiT ODE with gradient -> refiner (use_detail_refiner=1, freeze_jit=0)
        """
        dummy_labels = self._prepare_labels(x_rainy, dummy_labels)

        if self.use_detail_refiner:
            if self.freeze_jit:
                # Stage-2 training: JiT completes the ODE without retaining its
                # graph, then MSDT runs exactly once and receives all gradients.
                jit_prediction = self._generate_i2i_base(
                    x_rainy,
                    steps=self.steps,
                    dummy_labels=dummy_labels,
                )
            else:
                # Joint training: JiT ODE retains gradients so both JiT (unfrozen
                # blocks) and the refiner receive gradients.
                jit_prediction = self._generate_i2i_base_with_grad(
                    x_rainy,
                    steps=self.steps,
                    dummy_labels=dummy_labels,
                )
            return self.detail_refiner(x_rainy, jit_prediction)
        # 1. 随机采样时间步 t (范围 0~1)
        # t=0 代表完全是雨图，t=1 代表完全是干净图
        t = torch.rand(x_rainy.size(0), device=x_rainy.device).view(-1, *([1] * (x_rainy.ndim - 1)))

        # 2. 图像插值混合 (Rectified Flow 核心公式)
        z = t * x_clean + (1 - t) * x_rainy

        # 3. 网络看着混合后的图像 z，预测终点 (完全干净的图)
        x_pred = self.net(z, t.flatten(), dummy_labels)

        return x_pred

    # def forward(self, x, labels):
    #     labels_dropped = self.drop_labels(labels) if self.training else labels

    #     t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
    #     e = torch.randn_like(x) * self.noise_scale

    #     z = t * x + (1 - t) * e
    #     v = (x - z) / (1 - t).clamp_min(self.t_eps)

    #     x_pred = self.net(z, t.flatten(), labels_dropped)
    #     v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

    #     # l2 loss
    #     loss = (v - v_pred) ** 2
    #     loss = loss.mean(dim=(1, 2, 3)).mean()

    #     return loss
    # ==========================================
    # 2. 推理阶段：10步渐进式去雨 (Euler ODE 求解器)
    # ==========================================
    @torch.no_grad()
    def _generate_i2i_base(self, x_rainy, steps=10, dummy_labels=None):
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        z = x_rainy.clone()
        bsz = z.size(0)
        device = z.device
        dummy_labels = self._prepare_labels(z, dummy_labels)
        # 构建时间轴: 例如 steps=10 时，t 从 0.0, 0.1, 0.2 ... 到 1.0
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        # 逐步演化
        for i in range(steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # 1. 网络预测终点 (x_clean)
            x_pred = self.net(z, t.flatten(), dummy_labels)

            # 2. 计算当前时刻的去雨速度 (Velocity)
            # 数学推导: v = (x_clean - z) / (1 - t)
            v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

            # 3. 往前走一小步 (Euler Step)
            # z_next = z_current + Δt * v
            z = z + (t_next - t) * v_pred

        return z

    def _generate_i2i_base_with_grad(self, x_rainy, steps=10, dummy_labels=None):
        """Gradient-enabled version for joint JiT+refiner training.

        When steps=1 the ODE reduces to a single forward pass:
        z_next = x_rainy + (1-0) * (net(x_rainy, 0) - x_rainy) / 1 = net(x_rainy, 0)
        """
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        z = x_rainy.clone()
        bsz = z.size(0)
        device = z.device
        dummy_labels = self._prepare_labels(z, dummy_labels)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        for i in range(steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            x_pred = self.net(z, t.flatten(), dummy_labels)
            v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
            z = z + (t_next - t) * v_pred

        return z

    @torch.no_grad()
    def generate_i2i(self, x_rainy, steps=10, dummy_labels=None):
        """Complete the JiT ODE and optionally refine its endpoint once."""
        jit_prediction = self._generate_i2i_base(
            x_rainy,
            steps=steps,
            dummy_labels=dummy_labels,
        )
        if self.detail_refiner is None:
            return jit_prediction
        return self.detail_refiner(x_rainy, jit_prediction)

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
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
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

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
