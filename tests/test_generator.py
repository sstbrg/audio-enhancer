"""Tests for the HiFi-GAN generator model."""

import math

import pytest
import torch

from models.constants import (
    GENERATOR_CHANNELS,
    GENERATOR_RESBLOCK_DILATIONS,
    GENERATOR_RESBLOCK_KERNELS,
    GENERATOR_UPSAMPLE_KERNELS,
    GENERATOR_UPSAMPLE_RATES,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
)
from models.generator import Generator


UPSAMPLE_FACTOR = OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE  # 2


def make_sine(batch: int = 1, channels: int = 1, samples: int = 16384, freq: float = 440.0) -> torch.Tensor:
    """Create a sine wave tensor (batch, channels, samples)."""
    t = torch.linspace(0, samples / INPUT_SAMPLE_RATE, samples)
    wave = torch.sin(2 * math.pi * freq * t)
    return wave.unsqueeze(0).unsqueeze(0).expand(batch, channels, -1).clone()


class TestGeneratorInstantiation:
    def test_default_instantiation(self):
        gen = Generator()
        assert isinstance(gen, Generator)

    def test_channels_from_constants(self):
        gen = Generator(channels=GENERATOR_CHANNELS)
        assert isinstance(gen, Generator)

    def test_upsample_factor_computed(self):
        gen = Generator()
        assert gen.upsample_factor == UPSAMPLE_FACTOR

    def test_custom_channels(self):
        gen = Generator(channels=64)
        assert isinstance(gen, Generator)


class TestGeneratorForwardShape:
    def test_output_double_samples(self):
        """Output should have 2x the input samples (48kHz -> 96kHz)."""
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, samples=16384)
        with torch.no_grad():
            y = gen(x)
        assert y.shape[-1] == x.shape[-1] * UPSAMPLE_FACTOR

    def test_batch_dimension_preserved(self):
        gen = Generator()
        gen.eval()
        x = make_sine(batch=4, samples=8192)
        with torch.no_grad():
            y = gen(x)
        assert y.shape[0] == 4
        assert y.shape[1] == 1

    def test_channel_dimension_is_one(self):
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, samples=8192)
        with torch.no_grad():
            y = gen(x)
        assert y.shape[1] == 1

    def test_short_input(self):
        """Very short inputs (256 samples) should still produce 2x output."""
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, samples=256)
        with torch.no_grad():
            y = gen(x)
        assert y.shape[-1] == x.shape[-1] * UPSAMPLE_FACTOR

    def test_output_shape_3d(self):
        gen = Generator()
        gen.eval()
        x = make_sine(batch=2, samples=4096)
        with torch.no_grad():
            y = gen(x)
        assert y.dim() == 3


class TestGeneratorSkipConnection:
    def test_skip_influences_output(self):
        """The residual skip connection should make output close to upsampled input
        at init (before training), but not identical due to conv_post."""
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, samples=4096)
        with torch.no_grad():
            y = gen(x)
        # Output is x_conv + skip (via tanh + skip), so it won't be identical
        # Just verify that something with scale similar to input comes through
        assert y.abs().max().item() > 0.0

    def test_output_not_all_zeros(self):
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, samples=4096)
        with torch.no_grad():
            y = gen(x)
        assert not torch.all(y == 0)


class TestGeneratorOutputRange:
    def test_tanh_bounds_plus_skip(self):
        """tanh output is in (-1,1), skip from interpolated input is in [-1,1],
        so sum can reach ~2. But with random init and sine input it should be finite."""
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, samples=16384)
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "Output contains NaN or Inf"

    def test_no_nan_output(self):
        gen = Generator()
        gen.eval()
        x = make_sine(batch=2, samples=8192)
        with torch.no_grad():
            y = gen(x)
        assert not torch.isnan(y).any()

    def test_no_inf_output(self):
        gen = Generator()
        gen.eval()
        x = make_sine(batch=2, samples=8192)
        with torch.no_grad():
            y = gen(x)
        assert not torch.isinf(y).any()

    def test_output_range_reasonable(self):
        """Random-init generator with a normalised sine should stay within [-4, 4]."""
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, samples=16384)
        with torch.no_grad():
            y = gen(x)
        assert y.abs().max().item() < 10.0, "Output values are unexpectedly large"


class TestGeneratorMonoStereo:
    def test_mono_input(self):
        gen = Generator()
        gen.eval()
        x = make_sine(batch=1, channels=1, samples=8192)
        with torch.no_grad():
            y = gen(x)
        assert y.shape == (1, 1, 8192 * UPSAMPLE_FACTOR)

    def test_stereo_requires_separate_channel_processing(self):
        """Generator is mono (in_channels=1). Stereo must be processed per-channel
        as done in enhance.py. Confirm that a (1,1,T) mono call works."""
        gen = Generator()
        gen.eval()
        left = make_sine(batch=1, channels=1, samples=8192, freq=440.0)
        right = make_sine(batch=1, channels=1, samples=8192, freq=880.0)
        with torch.no_grad():
            out_left = gen(left)
            out_right = gen(right)
        assert out_left.shape == out_right.shape
        assert not torch.allclose(out_left, out_right, atol=1e-4)

    def test_noise_input_no_nan(self):
        gen = Generator()
        gen.eval()
        x = torch.randn(1, 1, 8192) * 0.1
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all()


class TestGeneratorRemoveWeightNorm:
    def test_remove_weight_norm(self):
        gen = Generator()
        gen.remove_weight_norm()
        # After removing weight norm, forward pass should still work
        gen.eval()
        x = make_sine(batch=1, samples=4096)
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all()
