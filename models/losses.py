"""Loss functions for audio GAN training.

Includes adversarial, feature matching, and perceptual spectral losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .constants import OUTPUT_SAMPLE_RATE


def generator_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    """Least-squares generator adversarial loss."""
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)
    return loss


def discriminator_loss(
    real_outputs: list[torch.Tensor],
    fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """Least-squares discriminator loss."""
    loss = 0
    for dr, df in zip(real_outputs, fake_outputs):
        loss += torch.mean((1 - dr) ** 2) + torch.mean(df ** 2)
    return loss


def feature_loss(
    real_fmaps: list[list[torch.Tensor]],
    fake_fmaps: list[list[torch.Tensor]],
) -> torch.Tensor:
    """L1 feature matching loss across discriminator layers."""
    loss = 0
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for real_feat, fake_feat in zip(real_layers, fake_layers):
            loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss


class STFTLoss(nn.Module):
    """Single-resolution STFT loss (spectral convergence + log magnitude)."""

    def __init__(self, fft_size: int, hop_size: int, win_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("window", torch.hann_window(win_size))

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y, y_hat: (batch, 1, time) or (batch, time)
        if y.dim() == 3:
            y = y.squeeze(1)
            y_hat = y_hat.squeeze(1)

        y_stft = torch.stft(y, self.fft_size, self.hop_size, self.win_size,
                            self.window, return_complex=True)
        y_hat_stft = torch.stft(y_hat, self.fft_size, self.hop_size, self.win_size,
                                self.window, return_complex=True)

        y_mag = torch.abs(y_stft)
        y_hat_mag = torch.abs(y_hat_stft)

        # Spectral convergence (per batch item, then averaged)
        sc_loss = torch.norm(y_mag - y_hat_mag, p="fro", dim=(1, 2)) / (torch.norm(y_mag, p="fro", dim=(1, 2)) + 1e-8)
        sc_loss = sc_loss.mean()

        # Log magnitude loss
        log_loss = F.l1_loss(torch.log(y_mag + 1e-8), torch.log(y_hat_mag + 1e-8))

        return sc_loss + log_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for perceptual audio quality."""

    def __init__(self):
        super().__init__()
        # Multiple resolutions capture different time-frequency trade-offs
        self.stft_losses = nn.ModuleList([
            STFTLoss(512, 50, 240),
            STFTLoss(1024, 120, 600),
            STFTLoss(2048, 240, 1200),
            STFTLoss(4096, 480, 2400),
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0
        for stft_loss in self.stft_losses:
            loss += stft_loss(y_hat, y)
        return loss / len(self.stft_losses)


class MelSpectrogramLoss(nn.Module):
    """L1 mel spectrogram loss for perceptual quality.

    Instantiates the MelSpectrogram transform once in __init__ instead of
    re-creating it every forward pass (avoids recomputing filterbanks on GPU).
    """

    def __init__(
        self,
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        n_fft: int = 4096,
        hop_length: int = 480,
        n_mels: int = 128,
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_max=sample_rate // 2,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.squeeze(1)
            y_hat = y_hat.squeeze(1)

        y_mel = self.mel_transform(y)
        y_hat_mel = self.mel_transform(y_hat)

        return F.l1_loss(torch.log(y_hat_mel + 1e-8), torch.log(y_mel + 1e-8))


# Backwards-compatible function wrapper (deprecated)
def mel_spectrogram_loss(y_hat, y, sample_rate=OUTPUT_SAMPLE_RATE):
    """Deprecated: use MelSpectrogramLoss module instead."""
    return MelSpectrogramLoss(sample_rate=sample_rate).to(y.device)(y_hat, y)
