"""Generator for audio super-resolution (48kHz -> 96kHz).

Architecture inspired by HiFi-GAN but adapted for bandwidth extension:
- Takes 48kHz waveform input
- Upsamples 2x via transposed convolutions
- Multi-receptive-field fusion blocks for quality
- Outputs 96kHz waveform with generated high-frequency content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def remove_weight_norm(module):
    """Remove weight norm compatible with both old and new PyTorch API."""
    try:
        remove_parametrizations(module, "weight")
    except (ValueError, KeyError):
        try:
            torch.nn.utils.remove_weight_norm(module)
        except Exception:
            pass

from .constants import (
    GENERATOR_CHANNELS,
    GENERATOR_UPSAMPLE_RATES,
    GENERATOR_UPSAMPLE_KERNELS,
    GENERATOR_RESBLOCK_KERNELS,
    GENERATOR_RESBLOCK_DILATIONS,
    LEAKY_RELU_SLOPE,
)


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d, padding=(kernel_size * d - d) // 2
                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=1, padding=(kernel_size - 1) // 2
                ))
            )

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LEAKY_RELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class Generator(nn.Module):
    """Audio super-resolution generator.

    Takes low-sample-rate audio and upsamples it while generating
    plausible high-frequency content via learned transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 512,
        upsample_rates: list[int] = GENERATOR_UPSAMPLE_RATES,
        upsample_kernel_sizes: list[int] = GENERATOR_UPSAMPLE_KERNELS,
        resblock_kernel_sizes: list[int] = GENERATOR_RESBLOCK_KERNELS,
        resblock_dilation_sizes: list[list[int]] = GENERATOR_RESBLOCK_DILATIONS,
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)

        # Input projection — wider kernel to capture more temporal context
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, channels, 7, padding=3))

        # Upsampling path with residual blocks
        ch = channels
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    ch, ch // 2,
                    kernel_size=k, stride=u,
                    padding=(k - u) // 2,
                ))
            )
            ch_out = ch // 2

            for j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch_out, rk, rd))

            ch = ch_out

        # High-frequency detail branch — parallel path focused on harmonics
        self.hf_branch = nn.Sequential(
            weight_norm(nn.Conv1d(ch, ch, 15, padding=7)),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            weight_norm(nn.Conv1d(ch, ch, 3, padding=1)),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
        )

        # Output projection
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

        # Skip connection from upsampled input
        total_upsample = 1
        for r in upsample_rates:
            total_upsample *= r
        self.upsample_factor = total_upsample

        assert len(self.resblocks) % self.num_upsamples == 0, (
            f"Number of resblocks ({len(self.resblocks)}) must be evenly divisible "
            f"by num_upsamples ({self.num_upsamples}). Got "
            f"{len(resblock_kernel_sizes)} resblock kernels × {self.num_upsamples} "
            f"upsample stages = {len(self.resblocks)} resblocks total."
        )
        self.n_resblocks = len(self.resblocks) // self.num_upsamples

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input waveform (batch, 1, samples) at low sample rate

        Returns:
            Upsampled waveform (batch, 1, samples * upsample_factor)
        """
        # Create upsampled skip via interpolation (linear baseline)
        skip = F.interpolate(x, scale_factor=float(self.upsample_factor), mode="linear", align_corners=False)

        # Feature extraction
        x = self.conv_pre(x)

        # Upsample with residual blocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            x = self.ups[i](x)

            # Apply all resblocks for this upsample level and sum
            xs = 0
            for j in range(self.n_resblocks):
                idx = i * self.n_resblocks + j
                xs = xs + self.resblocks[idx](x)
            x = xs / self.n_resblocks

        # High-frequency detail
        hf = self.hf_branch(x)
        x = x + hf

        x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        # Trim to match skip length
        min_len = min(x.shape[-1], skip.shape[-1])
        x = x[..., :min_len]
        skip = skip[..., :min_len]

        # Residual learning: generate the high-frequency residual
        return x + skip

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for m in self.hf_branch:
            if isinstance(m, nn.Conv1d):
                remove_weight_norm(m)
