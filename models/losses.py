"""Loss functions for audio GAN training.

Includes adversarial, feature matching, and perceptual spectral losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .constants import (
    HF_BAND_FFT_SIZE,
    HF_BAND_HIGH_FREQ,
    HF_BAND_HOP_SIZE,
    HF_BAND_LOW_FREQ,
    HF_BAND_WIN_SIZE,
    OUTPUT_SAMPLE_RATE,
)


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

        # torch.stft does not support float16 and magnitudes can overflow fp16 max.
        # Disable autocast and run entirely in float32.
        with torch.autocast("cuda", enabled=False):
            y = y.float()
            y_hat = y_hat.float()

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
        # Consistent ratios: hop = fft_size // 4, win = fft_size // 2
        self.stft_losses = nn.ModuleList([
            STFTLoss(512, 128, 256),
            STFTLoss(1024, 256, 512),
            STFTLoss(2048, 512, 1024),
            STFTLoss(4096, 1024, 2048),
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

        # MelSpectrogram values can exceed float16 max (65504), causing inf/NaN.
        # Disable autocast and run entirely in float32.
        with torch.autocast("cuda", enabled=False):
            y_mel = self.mel_transform(y.float())
            y_hat_mel = self.mel_transform(y_hat.float())

        return F.l1_loss(torch.log(y_hat_mel + 1e-8), torch.log(y_mel + 1e-8))


class HighFrequencyBandLoss(nn.Module):
    """L1 loss on the 16-24 kHz spectral band for Phase 1 codec artifact restoration.

    Lossy codecs cause the most visible damage in the 16-24 kHz range (cutoffs,
    SBR artifacts, spectral holes). This loss focuses the generator on restoring
    that specific band.
    """

    def __init__(
        self,
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        fft_size: int = HF_BAND_FFT_SIZE,
        hop_size: int = HF_BAND_HOP_SIZE,
        win_size: int = HF_BAND_WIN_SIZE,
        low_freq: float = HF_BAND_LOW_FREQ,
        high_freq: float = HF_BAND_HIGH_FREQ,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("window", torch.hann_window(win_size))

        # Precompute frequency bin mask for the target band
        freqs = torch.linspace(0, sample_rate / 2, fft_size // 2 + 1)
        self.register_buffer("band_mask", (freqs >= low_freq) & (freqs <= high_freq))

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.squeeze(1)
            y_hat = y_hat.squeeze(1)

        with torch.autocast("cuda", enabled=False):
            y = y.float()
            y_hat = y_hat.float()

            y_stft = torch.stft(y, self.fft_size, self.hop_size, self.win_size,
                                self.window, return_complex=True)
            y_hat_stft = torch.stft(y_hat, self.fft_size, self.hop_size, self.win_size,
                                    self.window, return_complex=True)

            # Extract magnitudes in the HF band only
            y_hf = torch.abs(y_stft[:, self.band_mask, :])
            y_hat_hf = torch.abs(y_hat_stft[:, self.band_mask, :])

        return F.l1_loss(y_hat_hf, y_hf)


# Backwards-compatible function wrapper (deprecated)
def mel_spectrogram_loss(y_hat, y, sample_rate=OUTPUT_SAMPLE_RATE):
    """Deprecated: use MelSpectrogramLoss module instead."""
    return MelSpectrogramLoss(sample_rate=sample_rate).to(y.device)(y_hat, y)
