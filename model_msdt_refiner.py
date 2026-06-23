"""Lightweight MSDT-style detail refiner for JiT raindrop removal.

This module is intentionally not a full reimplementation of the original MSDT.
It keeps the parts that are complementary to JiT:
  * three-scale processing;
  * spatial + frequency enhancement blocks;
  * coarse-to-fine gated feature fusion;
  * gated residual refinement of the JiT clean-image prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEnhancedBlock(nn.Module):
    """Spatial/frequency residual block.

    The FFT path always runs in float32 because CUDA FFT support for bf16/fp16
    is limited and can be numerically fragile. The result is cast back to the
    incoming feature dtype before residual fusion.
    """

    def __init__(self, channels: int, expansion: int = 2, use_frequency: bool = True):
        super().__init__()
        hidden = channels * expansion
        self.use_frequency = bool(use_frequency)

        self.norm = nn.GroupNorm(1, channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        if self.use_frequency:
            # Real and imaginary components are concatenated along channels.
            self.frequency = nn.Sequential(
                nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=True),
            )
        else:
            self.frequency = None

        # Small residual scales keep a newly attached refiner stable.
        self.spatial_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.frequency_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def _frequency_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frequency is None:
            return torch.zeros_like(x)

        input_dtype = x.dtype
        height, width = x.shape[-2:]
        # Disable autocast explicitly for FFT and complex tensor construction.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            spectrum = torch.fft.rfft2(x_float, norm="ortho")
            spectrum_ri = torch.cat((spectrum.real, spectrum.imag), dim=1)
            spectrum_ri = self.frequency(spectrum_ri)
            real, imag = spectrum_ri.chunk(2, dim=1)
            restored = torch.fft.irfft2(
                torch.complex(real, imag),
                s=(height, width),
                norm="ortho",
            )
        return restored.to(dtype=input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        spatial = self.spatial(normalized)
        frequency = self._frequency_forward(normalized)
        return (
            x
            + self.spatial_scale.to(dtype=x.dtype) * spatial
            + self.frequency_scale.to(dtype=x.dtype) * frequency
        )


class GatedScaleFusion(nn.Module):
    """Fuse an upsampled coarse feature into a finer-scale feature."""

    def __init__(self, fine_channels: int, coarse_channels: int):
        super().__init__()
        self.coarse_proj = nn.Conv2d(coarse_channels, fine_channels, 1, bias=True)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * fine_channels, fine_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(fine_channels, fine_channels, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(fine_channels, fine_channels, 3, padding=1, bias=True),
            nn.GELU(),
        )

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        coarse = F.interpolate(
            coarse,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse = self.coarse_proj(coarse)
        gate = self.gate(torch.cat((fine, coarse), dim=1))
        return fine + self.mix(gate * coarse)


@dataclass
class RefinerOutput:
    image: torch.Tensor
    delta: torch.Tensor
    mask: torch.Tensor


class MSDTDetailRefiner(nn.Module):
    """Three-scale detail refiner applied after a JiT clean-image prediction.

    Input features are built from:
        [rainy image, JiT prediction, rainy image - JiT prediction]
    producing nine channels for RGB input.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        num_blocks: int = 2,
        use_frequency: bool = True,
        max_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if base_dim <= 0:
            raise ValueError(f"base_dim must be positive, got {base_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if max_residual <= 0:
            raise ValueError(f"max_residual must be positive, got {max_residual}")

        self.in_channels = in_channels
        self.refiner_in_channels = in_channels * 3
        self.max_residual = float(max_residual)

        dim1 = base_dim
        dim2 = base_dim * 2
        dim4 = base_dim * 4

        self.stem_1x = nn.Conv2d(self.refiner_in_channels, dim1, 3, padding=1)
        self.stem_2x = nn.Conv2d(self.refiner_in_channels, dim2, 3, padding=1)
        self.stem_4x = nn.Conv2d(self.refiner_in_channels, dim4, 3, padding=1)

        self.blocks_4x = nn.Sequential(
            *[
                FrequencyEnhancedBlock(dim4, use_frequency=use_frequency)
                for _ in range(num_blocks)
            ]
        )
        self.fuse_4x_to_2x = GatedScaleFusion(dim2, dim4)
        self.blocks_2x = nn.Sequential(
            *[
                FrequencyEnhancedBlock(dim2, use_frequency=use_frequency)
                for _ in range(num_blocks)
            ]
        )
        self.fuse_2x_to_1x = GatedScaleFusion(dim1, dim2)
        self.blocks_1x = nn.Sequential(
            *[
                FrequencyEnhancedBlock(dim1, use_frequency=use_frequency)
                for _ in range(num_blocks)
            ]
        )

        self.delta_head = nn.Conv2d(dim1, in_channels, 3, padding=1, bias=True)
        self.mask_head = nn.Conv2d(dim1, 1, 3, padding=1, bias=True)

        # Initial behavior is exactly the JiT prediction: delta == 0.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -2.0)

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        rainy: torch.Tensor,
        jit_prediction: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], RefinerOutput]:
        if rainy.shape != jit_prediction.shape:
            raise ValueError(
                "rainy and jit_prediction must have identical shapes, got "
                f"{tuple(rainy.shape)} and {tuple(jit_prediction.shape)}"
            )
        if rainy.ndim != 4 or rainy.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channels, got {tuple(rainy.shape)}"
            )

        combined = torch.cat(
            (rainy, jit_prediction, rainy - jit_prediction), dim=1
        )
        height, width = combined.shape[-2:]
        half_size = (max(1, height // 2), max(1, width // 2))
        quarter_size = (max(1, height // 4), max(1, width // 4))

        input_1x = combined
        input_2x = self._resize(combined, half_size)
        input_4x = self._resize(combined, quarter_size)

        feature_4x = self.blocks_4x(self.stem_4x(input_4x))
        feature_2x = self.stem_2x(input_2x)
        feature_2x = self.fuse_4x_to_2x(feature_2x, feature_4x)
        feature_2x = self.blocks_2x(feature_2x)

        feature_1x = self.stem_1x(input_1x)
        feature_1x = self.fuse_2x_to_1x(feature_1x, feature_2x)
        feature_1x = self.blocks_1x(feature_1x)

        delta = self.max_residual * torch.tanh(self.delta_head(feature_1x))
        mask = torch.sigmoid(self.mask_head(feature_1x))
        image = jit_prediction + mask * delta

        if return_details:
            return {
                "image": image,
                "delta": delta,
                "mask": mask,
            }
        return image
