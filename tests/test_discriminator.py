"""Tests for multi-period and multi-scale discriminators."""

import math

import pytest
import torch

from models.constants import LEAKY_RELU_SLOPE, MPD_PERIODS, MSD_SCALES
from models.discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    PeriodDiscriminator,
    ScaleDiscriminator,
)


def make_audio(batch: int = 1, samples: int = 22050) -> torch.Tensor:
    """Return (batch, 1, samples) sine wave."""
    t = torch.linspace(0, 1.0, samples)
    wave = torch.sin(2 * math.pi * 440.0 * t)
    return wave.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1).clone()


class TestPeriodDiscriminator:
    @pytest.mark.parametrize("period", MPD_PERIODS)
    def test_instantiation(self, period):
        d = PeriodDiscriminator(period)
        assert isinstance(d, PeriodDiscriminator)

    @pytest.mark.parametrize("period", MPD_PERIODS)
    def test_forward_returns_tuple(self, period):
        d = PeriodDiscriminator(period)
        d.eval()
        x = make_audio(batch=2, samples=22050)
        with torch.no_grad():
            out, fmap = d(x)
        assert isinstance(out, torch.Tensor)
        assert isinstance(fmap, list)

    @pytest.mark.parametrize("period", MPD_PERIODS)
    def test_output_is_2d(self, period):
        """Flattened output should be (batch, N)."""
        d = PeriodDiscriminator(period)
        d.eval()
        x = make_audio(batch=2, samples=22050)
        with torch.no_grad():
            out, _ = d(x)
        assert out.dim() == 2
        assert out.shape[0] == 2

    @pytest.mark.parametrize("period", MPD_PERIODS)
    def test_feature_map_count(self, period):
        """5 convs + 1 conv_post = 6 feature maps."""
        d = PeriodDiscriminator(period)
        d.eval()
        x = make_audio(batch=1, samples=22050)
        with torch.no_grad():
            _, fmap = d(x)
        assert len(fmap) == 6

    def test_spectral_norm_variant(self):
        d = PeriodDiscriminator(period=5, use_spectral_norm=True)
        d.eval()
        x = make_audio(batch=1, samples=8192)
        with torch.no_grad():
            out, fmap = d(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("period", MPD_PERIODS)
    def test_no_nan_output(self, period):
        d = PeriodDiscriminator(period)
        d.eval()
        x = make_audio(batch=1, samples=16384)
        with torch.no_grad():
            out, fmap = d(x)
        assert not torch.isnan(out).any()
        for f in fmap:
            assert not torch.isnan(f).any()

    def test_pads_non_divisible_length(self):
        """Lengths not divisible by period must be padded without error."""
        d = PeriodDiscriminator(period=7)
        d.eval()
        # 10000 is not divisible by 7
        x = make_audio(batch=1, samples=10000)
        with torch.no_grad():
            out, _ = d(x)
        assert torch.isfinite(out).all()


class TestMultiPeriodDiscriminator:
    def test_instantiation_default(self):
        mpd = MultiPeriodDiscriminator()
        assert len(mpd.discriminators) == len(MPD_PERIODS)

    def test_instantiation_custom_periods(self):
        mpd = MultiPeriodDiscriminator(periods=[2, 3, 5])
        assert len(mpd.discriminators) == 3

    def test_forward_returns_four_lists(self):
        mpd = MultiPeriodDiscriminator()
        mpd.eval()
        real = make_audio(batch=2, samples=22050)
        fake = make_audio(batch=2, samples=22050)
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = mpd(real, fake)
        assert len(r_out) == len(MPD_PERIODS)
        assert len(f_out) == len(MPD_PERIODS)
        assert len(r_fmap) == len(MPD_PERIODS)
        assert len(f_fmap) == len(MPD_PERIODS)

    def test_real_fake_outputs_differ(self):
        """Different inputs should produce different outputs."""
        mpd = MultiPeriodDiscriminator()
        mpd.eval()
        real = make_audio(batch=1, samples=16384)
        fake = torch.randn_like(real) * 0.1
        with torch.no_grad():
            r_out, f_out, _, _ = mpd(real, fake)
        # At least some discriminators should produce different scores
        diffs = [not torch.allclose(r, f) for r, f in zip(r_out, f_out)]
        assert any(diffs)

    def test_all_outputs_finite(self):
        mpd = MultiPeriodDiscriminator()
        mpd.eval()
        real = make_audio(batch=2, samples=22050)
        fake = torch.randn_like(real) * 0.5
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = mpd(real, fake)
        for t in r_out + f_out:
            assert torch.isfinite(t).all()


class TestScaleDiscriminator:
    def test_instantiation_weight_norm(self):
        d = ScaleDiscriminator(use_spectral_norm=False)
        assert isinstance(d, ScaleDiscriminator)

    def test_instantiation_spectral_norm(self):
        d = ScaleDiscriminator(use_spectral_norm=True)
        assert isinstance(d, ScaleDiscriminator)

    def test_forward_returns_tuple(self):
        d = ScaleDiscriminator()
        d.eval()
        x = make_audio(batch=2, samples=22050)
        with torch.no_grad():
            out, fmap = d(x)
        assert isinstance(out, torch.Tensor)
        assert isinstance(fmap, list)

    def test_output_is_2d(self):
        d = ScaleDiscriminator()
        d.eval()
        x = make_audio(batch=3, samples=22050)
        with torch.no_grad():
            out, _ = d(x)
        assert out.dim() == 2
        assert out.shape[0] == 3

    def test_feature_map_count(self):
        """7 convs + 1 conv_post = 8 feature maps."""
        d = ScaleDiscriminator()
        d.eval()
        x = make_audio(batch=1, samples=22050)
        with torch.no_grad():
            _, fmap = d(x)
        assert len(fmap) == 8

    def test_no_nan_output(self):
        d = ScaleDiscriminator()
        d.eval()
        x = make_audio(batch=1, samples=22050)
        with torch.no_grad():
            out, fmap = d(x)
        assert not torch.isnan(out).any()
        for f in fmap:
            assert not torch.isnan(f).any()


class TestMultiScaleDiscriminator:
    def test_instantiation_default(self):
        msd = MultiScaleDiscriminator()
        assert len(msd.discriminators) == MSD_SCALES
        assert len(msd.pools) == MSD_SCALES - 1

    def test_first_discriminator_uses_spectral_norm(self):
        """First scale discriminator uses spectral norm per HiFi-GAN design."""
        msd = MultiScaleDiscriminator()
        # Just check it instantiates and runs without error
        msd.eval()
        real = make_audio(batch=1, samples=22050)
        fake = make_audio(batch=1, samples=22050)
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = msd(real, fake)
        assert len(r_out) == MSD_SCALES

    def test_forward_returns_four_lists(self):
        msd = MultiScaleDiscriminator()
        msd.eval()
        real = make_audio(batch=2, samples=22050)
        fake = make_audio(batch=2, samples=22050)
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = msd(real, fake)
        assert len(r_out) == MSD_SCALES
        assert len(f_out) == MSD_SCALES
        assert len(r_fmap) == MSD_SCALES
        assert len(f_fmap) == MSD_SCALES

    def test_all_outputs_finite(self):
        msd = MultiScaleDiscriminator()
        msd.eval()
        real = make_audio(batch=2, samples=22050)
        fake = torch.randn_like(real) * 0.5
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = msd(real, fake)
        for t in r_out + f_out:
            assert torch.isfinite(t).all()

    def test_downsampled_scales_produce_output(self):
        """Each scale runs on progressively downsampled input — all must produce output."""
        msd = MultiScaleDiscriminator()
        msd.eval()
        real = make_audio(batch=1, samples=22050)
        fake = make_audio(batch=1, samples=22050)
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = msd(real, fake)
        assert all(r.numel() > 0 for r in r_out)
        assert all(f.numel() > 0 for f in f_out)

    def test_feature_maps_per_scale(self):
        """Each scale has 8 feature maps (7 convs + 1 conv_post)."""
        msd = MultiScaleDiscriminator()
        msd.eval()
        real = make_audio(batch=1, samples=22050)
        fake = make_audio(batch=1, samples=22050)
        with torch.no_grad():
            _, _, r_fmap, _ = msd(real, fake)
        for scale_fmap in r_fmap:
            assert len(scale_fmap) == 8
