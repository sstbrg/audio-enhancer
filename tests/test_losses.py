"""Tests for loss functions (STFT, mel, adversarial, feature matching)."""

import math

import pytest
import torch

from models.constants import OUTPUT_SAMPLE_RATE
from models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.losses import (
    MelSpectrogramLoss,
    MultiResolutionSTFTLoss,
    STFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    mel_spectrogram_loss,
)


def make_audio(batch: int = 2, samples: int = 22050, sr: int = OUTPUT_SAMPLE_RATE) -> torch.Tensor:
    """Return (batch, 1, samples) sine wave."""
    t = torch.linspace(0, samples / sr, samples)
    wave = torch.sin(2 * math.pi * 440.0 * t)
    return wave.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1).clone()


def make_audio_2d(batch: int = 2, samples: int = 22050) -> torch.Tensor:
    """Return (batch, samples) sine wave."""
    return make_audio(batch, samples).squeeze(1)


class TestSTFTLoss:
    def test_scalar_output(self):
        loss_fn = STFTLoss(fft_size=512, hop_size=50, win_size=240)
        y = make_audio(batch=2, samples=22050)
        y_hat = make_audio(batch=2, samples=22050)
        loss = loss_fn(y_hat, y)
        assert loss.dim() == 0

    def test_no_nan(self):
        loss_fn = STFTLoss(fft_size=1024, hop_size=120, win_size=600)
        y = make_audio(batch=2, samples=22050)
        y_hat = torch.randn_like(y) * 0.1
        loss = loss_fn(y_hat, y)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_no_inf(self):
        loss_fn = STFTLoss(fft_size=2048, hop_size=240, win_size=1200)
        y = make_audio(batch=1, samples=22050)
        y_hat = torch.zeros_like(y)
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)

    def test_same_input_low_loss(self):
        """Identical inputs should give near-zero spectral convergence loss."""
        loss_fn = STFTLoss(fft_size=1024, hop_size=120, win_size=600)
        y = make_audio(batch=1, samples=22050)
        loss = loss_fn(y, y)
        assert loss.item() < 1.0

    def test_accepts_3d_input(self):
        """(batch, 1, time) input should be squeezed internally."""
        loss_fn = STFTLoss(fft_size=512, hop_size=50, win_size=240)
        y = make_audio(batch=2, samples=8192)
        y_hat = torch.randn_like(y) * 0.5
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)

    def test_accepts_2d_input(self):
        """(batch, time) input (no channel dim) should also work."""
        loss_fn = STFTLoss(fft_size=512, hop_size=50, win_size=240)
        y = make_audio_2d(batch=2, samples=8192)
        y_hat = torch.randn_like(y) * 0.5
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)

    def test_float16_no_crash(self):
        """STFT loss must handle float16 inputs gracefully (runs in float32 internally)."""
        loss_fn = STFTLoss(fft_size=512, hop_size=50, win_size=240)
        y = make_audio(batch=1, samples=8192).half()
        y_hat = torch.randn(1, 1, 8192, dtype=torch.float16) * 0.1
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)


class TestMultiResolutionSTFTLoss:
    def test_scalar_output(self):
        loss_fn = MultiResolutionSTFTLoss()
        y = make_audio(batch=2, samples=22050)
        y_hat = make_audio(batch=2, samples=22050)
        loss = loss_fn(y_hat, y)
        assert loss.dim() == 0

    def test_no_nan(self):
        loss_fn = MultiResolutionSTFTLoss()
        y = make_audio(batch=2, samples=22050)
        y_hat = torch.randn_like(y) * 0.3
        loss = loss_fn(y_hat, y)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_positive_loss(self):
        loss_fn = MultiResolutionSTFTLoss()
        y = make_audio(batch=2, samples=22050)
        y_hat = torch.randn_like(y) * 0.5
        loss = loss_fn(y_hat, y)
        assert loss.item() >= 0.0

    def test_averages_four_resolutions(self):
        """MultiResolutionSTFTLoss uses 4 resolutions; output is their average."""
        loss_fn = MultiResolutionSTFTLoss()
        assert len(loss_fn.stft_losses) == 4


class TestMelSpectrogramLoss:
    def test_scalar_output(self):
        loss_fn = MelSpectrogramLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=2, samples=22050)
        y_hat = make_audio(batch=2, samples=22050)
        loss = loss_fn(y_hat, y)
        assert loss.dim() == 0

    def test_no_nan(self):
        loss_fn = MelSpectrogramLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=2, samples=22050)
        y_hat = torch.randn_like(y) * 0.1
        loss = loss_fn(y_hat, y)
        assert not torch.isnan(loss).any()

    def test_no_inf(self):
        loss_fn = MelSpectrogramLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=2, samples=22050)
        y_hat = torch.zeros_like(y)
        loss = loss_fn(y_hat, y)
        assert not torch.isinf(loss).any()

    def test_same_input_low_loss(self):
        loss_fn = MelSpectrogramLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=1, samples=22050)
        loss = loss_fn(y, y)
        assert loss.item() < 0.1

    def test_float16_no_crash(self):
        """Mel loss must handle float16 inputs (runs internally in float32)."""
        loss_fn = MelSpectrogramLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=1, samples=22050).half()
        y_hat = torch.randn(1, 1, 22050, dtype=torch.float16) * 0.1
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)

    def test_accepts_3d_input(self):
        loss_fn = MelSpectrogramLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=2, samples=16384)
        y_hat = torch.randn_like(y) * 0.5
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)

    def test_accepts_2d_input(self):
        loss_fn = MelSpectrogramLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio_2d(batch=2, samples=16384)
        y_hat = torch.randn_like(y) * 0.5
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)

    def test_deprecated_function_wrapper(self):
        y = make_audio(batch=1, samples=22050)
        y_hat = torch.randn_like(y) * 0.1
        loss = mel_spectrogram_loss(y_hat, y, sample_rate=OUTPUT_SAMPLE_RATE)
        assert torch.isfinite(loss)


class TestAdversarialLoss:
    def _run_discriminators(self, batch: int = 2, samples: int = 22050):
        mpd = MultiPeriodDiscriminator()
        msd = MultiScaleDiscriminator()
        real = make_audio(batch=batch, samples=samples)
        fake = torch.randn_like(real) * 0.5
        with torch.no_grad():
            r_mpd, f_mpd, _, _ = mpd(real, fake)
            r_msd, f_msd, _, _ = msd(real, fake)
        return r_mpd + r_msd, f_mpd + f_msd

    def test_discriminator_loss_scalar(self):
        real_outs, fake_outs = self._run_discriminators()
        loss = discriminator_loss(real_outs, fake_outs)
        assert loss.dim() == 0

    def test_discriminator_loss_positive(self):
        real_outs, fake_outs = self._run_discriminators()
        loss = discriminator_loss(real_outs, fake_outs)
        assert loss.item() >= 0.0

    def test_discriminator_loss_no_nan(self):
        real_outs, fake_outs = self._run_discriminators()
        loss = discriminator_loss(real_outs, fake_outs)
        assert not torch.isnan(loss)

    def test_generator_loss_scalar(self):
        _, fake_outs = self._run_discriminators()
        loss = generator_loss(fake_outs)
        assert loss.dim() == 0

    def test_generator_loss_positive(self):
        _, fake_outs = self._run_discriminators()
        loss = generator_loss(fake_outs)
        assert loss.item() >= 0.0

    def test_generator_loss_no_nan(self):
        _, fake_outs = self._run_discriminators()
        loss = generator_loss(fake_outs)
        assert not torch.isnan(loss)


class TestFeatureMatchingLoss:
    def test_scalar_output(self):
        mpd = MultiPeriodDiscriminator()
        real = make_audio(batch=2, samples=22050)
        fake = torch.randn_like(real) * 0.5
        with torch.no_grad():
            _, _, r_fmap, f_fmap = mpd(real, fake)
        loss = feature_loss(r_fmap, f_fmap)
        assert loss.dim() == 0

    def test_positive_loss(self):
        mpd = MultiPeriodDiscriminator()
        real = make_audio(batch=2, samples=22050)
        fake = torch.randn_like(real) * 0.5
        with torch.no_grad():
            _, _, r_fmap, f_fmap = mpd(real, fake)
        loss = feature_loss(r_fmap, f_fmap)
        assert loss.item() >= 0.0

    def test_no_nan(self):
        mpd = MultiPeriodDiscriminator()
        real = make_audio(batch=2, samples=22050)
        fake = torch.randn_like(real) * 0.5
        with torch.no_grad():
            _, _, r_fmap, f_fmap = mpd(real, fake)
        loss = feature_loss(r_fmap, f_fmap)
        assert not torch.isnan(loss)

    def test_same_input_zero_loss(self):
        """Feature matching loss between identical inputs should be zero."""
        mpd = MultiPeriodDiscriminator()
        real = make_audio(batch=1, samples=16384)
        with torch.no_grad():
            _, _, r_fmap, f_fmap = mpd(real, real)
        loss = feature_loss(r_fmap, f_fmap)
        assert loss.item() < 1e-4

    def test_combined_mpd_msd_feature_loss(self):
        mpd = MultiPeriodDiscriminator()
        msd = MultiScaleDiscriminator()
        real = make_audio(batch=2, samples=22050)
        fake = torch.randn_like(real) * 0.3
        with torch.no_grad():
            _, _, r_fmap_mpd, f_fmap_mpd = mpd(real, fake)
            _, _, r_fmap_msd, f_fmap_msd = msd(real, fake)
        loss = feature_loss(r_fmap_mpd + r_fmap_msd, f_fmap_mpd + f_fmap_msd)
        assert torch.isfinite(loss)
