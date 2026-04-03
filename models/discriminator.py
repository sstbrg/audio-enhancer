"""Multi-scale and multi-period discriminators for audio GAN training.

Based on HiFi-GAN discriminator design, adapted for high-sample-rate audio.
The multi-period discriminator captures periodic structure (harmonics),
while the multi-scale discriminator handles different frequency bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MPD_PERIODS, MSD_SCALES, LEAKY_RELU_SLOPE
from torch.nn.utils.parametrizations import weight_norm, spectral_norm


class PeriodDiscriminator(nn.Module):
    """Sub-discriminator that reshapes 1D audio into 2D based on a period."""

    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            norm_f(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
            norm_f(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
            norm_f(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        # Reshape 1D -> 2D by period
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multiple period-based sub-discriminators."""

    def __init__(self, periods: list[int] = MPD_PERIODS):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        real_outputs = []
        fake_outputs = []
        real_fmaps = []
        fake_fmaps = []

        for d in self.discriminators:
            real_out, real_fmap = d(y)
            fake_out, fake_fmap = d(y_hat)
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_fmaps.append(real_fmap)
            fake_fmaps.append(fake_fmap)

        return real_outputs, fake_outputs, real_fmaps, fake_fmaps


class ScaleDiscriminator(nn.Module):
    """Sub-discriminator operating at a single scale."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, 7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm_f(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm_f(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multiple scale-based sub-discriminators with progressive downsampling."""

    def __init__(self, num_scales: int = MSD_SCALES):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=(i == 0))
            for i in range(num_scales)
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2)
            for _ in range(num_scales - 1)
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        real_outputs = []
        fake_outputs = []
        real_fmaps = []
        fake_fmaps = []

        for i, d in enumerate(self.discriminators):
            if i > 0:
                y = self.pools[i - 1](y)
                y_hat = self.pools[i - 1](y_hat)
            real_out, real_fmap = d(y)
            fake_out, fake_fmap = d(y_hat)
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_fmaps.append(real_fmap)
            fake_fmaps.append(fake_fmap)

        return real_outputs, fake_outputs, real_fmaps, fake_fmaps
