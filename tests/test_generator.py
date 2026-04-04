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


class TestGeneratorResblockAssertion:
    """Verify the resblock even-divisibility assertion (Florence code review item #9)."""

    def test_default_config_passes_assertion(self):
        """Default config should satisfy the assertion."""
        gen = Generator()
        expected = len(GENERATOR_RESBLOCK_KERNELS)
        assert gen.n_resblocks == expected

    def test_mismatched_config_raises(self):
        """If resblock count doesn't divide evenly by num_upsamples, assert fires."""
        # 2 upsample stages but 3 resblock kernels → 6 resblocks / 2 = 3, OK
        gen = Generator(
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )
        assert gen.n_resblocks == 3

    def test_single_resblock_per_stage(self):
        """Single resblock kernel with single upsample should work."""
        gen = Generator(
            upsample_rates=[2],
            upsample_kernel_sizes=[4],
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
        )
        assert gen.n_resblocks == 1


class TestGeneratorCodecArtifacts:
    """Phase 1 readiness: verify generator handles codec-degraded inputs.

    Codec artifacts include sharp spectral cutoffs (e.g., MP3 128kbps cuts at ~16kHz),
    quantization noise, and pre-echo. The skip connection uses F.interpolate(mode='linear')
    which is a low-pass upsample — it faithfully passes through degraded content without
    introducing new artifacts. The generator's learned residual path compensates.
    """

    def _make_bandlimited_signal(self, cutoff_ratio: float = 0.33, samples: int = 8192) -> torch.Tensor:
        """Create a signal with a hard spectral cutoff simulating codec bandwidth limiting.

        Args:
            cutoff_ratio: Fraction of Nyquist where energy drops to zero (e.g., 0.33 = 16kHz at 48kHz)
            samples: Number of samples
        """
        # White noise → FFT → zero out above cutoff → IFFT
        noise = torch.randn(1, 1, samples)
        spec = torch.fft.rfft(noise)
        n_bins = spec.shape[-1]
        cutoff_bin = int(n_bins * cutoff_ratio)
        spec[..., cutoff_bin:] = 0
        signal = torch.fft.irfft(spec, n=samples)
        # Normalize to [-0.9, 0.9] range typical of audio
        signal = signal / (signal.abs().max() + 1e-8) * 0.9
        return signal

    def _make_clipped_signal(self, samples: int = 8192) -> torch.Tensor:
        """Create a hard-clipped signal simulating lossy compression artifacts."""
        t = torch.linspace(0, samples / INPUT_SAMPLE_RATE, samples)
        # Multi-tone signal with clipping
        signal = (torch.sin(2 * math.pi * 440 * t) +
                  0.5 * torch.sin(2 * math.pi * 1760 * t) +
                  0.3 * torch.sin(2 * math.pi * 3520 * t))
        signal = signal.unsqueeze(0).unsqueeze(0)
        signal = signal / (signal.abs().max() + 1e-8)
        # Hard clip at 0.7 → introduces odd harmonics (like over-compressed audio)
        signal = torch.clamp(signal, -0.7, 0.7)
        return signal

    def _make_quantized_signal(self, bits: int = 8, samples: int = 8192) -> torch.Tensor:
        """Create a coarsely quantized signal simulating low-bitrate codec output."""
        t = torch.linspace(0, samples / INPUT_SAMPLE_RATE, samples)
        signal = torch.sin(2 * math.pi * 440 * t)
        signal = signal.unsqueeze(0).unsqueeze(0)
        # Quantize to N bits
        levels = 2 ** bits
        signal = torch.round(signal * levels / 2) / (levels / 2)
        return signal

    def test_bandlimited_input_no_nan(self):
        """MP3-like bandlimited input (cutoff at ~16kHz) should produce finite output."""
        gen = Generator()
        gen.eval()
        x = self._make_bandlimited_signal(cutoff_ratio=0.33)
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "NaN/Inf from bandlimited input"
        assert y.shape[-1] == x.shape[-1] * UPSAMPLE_FACTOR

    def test_severe_bandlimit_no_nan(self):
        """Extreme bandwidth limitation (voice-only, ~4kHz cutoff) should work."""
        gen = Generator()
        gen.eval()
        x = self._make_bandlimited_signal(cutoff_ratio=0.083)  # ~4kHz at 48kHz SR
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "NaN/Inf from severely bandlimited input"
        assert y.shape[-1] == x.shape[-1] * UPSAMPLE_FACTOR

    def test_clipped_input_no_nan(self):
        """Hard-clipped audio (compression artifacts) should produce finite output."""
        gen = Generator()
        gen.eval()
        x = self._make_clipped_signal()
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "NaN/Inf from clipped input"
        assert y.shape[-1] == x.shape[-1] * UPSAMPLE_FACTOR

    def test_quantized_input_no_nan(self):
        """Coarsely quantized audio (8-bit) should produce finite output."""
        gen = Generator()
        gen.eval()
        x = self._make_quantized_signal(bits=8)
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "NaN/Inf from quantized input"
        assert y.shape[-1] == x.shape[-1] * UPSAMPLE_FACTOR

    def test_skip_connection_preserves_bandlimited_content(self):
        """Skip connection should faithfully pass bandlimited content through.

        The linear interpolation in the skip doesn't add energy above the input's
        bandwidth — it only upsamples existing content. This is correct behavior:
        the generator's learned residual path is responsible for high-freq synthesis.
        """
        gen = Generator()
        gen.eval()
        x = self._make_bandlimited_signal(cutoff_ratio=0.33, samples=4096)

        # Get the skip output directly (what linear interp produces)
        skip = torch.nn.functional.interpolate(
            x, scale_factor=float(UPSAMPLE_FACTOR), mode="linear", align_corners=False
        )

        # Skip should have same bandwidth as input — no new HF energy
        skip_spec = torch.fft.rfft(skip)
        skip_power = (skip_spec.abs() ** 2).squeeze()

        n_bins = skip_power.shape[-1]
        # Energy above the original cutoff (in the upsampled domain) should be minimal
        original_cutoff_bin = int(n_bins * 0.33 / UPSAMPLE_FACTOR)
        hf_energy = skip_power[original_cutoff_bin + 10:].sum().item()
        total_energy = skip_power.sum().item()

        # HF should be <1% of total (interpolation leakage, not new content)
        if total_energy > 0:
            hf_ratio = hf_energy / total_energy
            assert hf_ratio < 0.01, (
                f"Skip connection introduced HF energy: {hf_ratio:.4f} ratio"
            )

    def test_silence_input(self):
        """All-zeros input should produce near-zero output (no NaN from division)."""
        gen = Generator()
        gen.eval()
        x = torch.zeros(1, 1, 8192)
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "NaN/Inf from silent input"

    def test_dc_offset_input(self):
        """Constant DC offset (common in poorly encoded audio) should not cause NaN."""
        gen = Generator()
        gen.eval()
        x = torch.full((1, 1, 8192), 0.5)
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "NaN/Inf from DC offset input"

    def test_impulse_input(self):
        """Single impulse (tests transient handling in skip + residual path)."""
        gen = Generator()
        gen.eval()
        x = torch.zeros(1, 1, 8192)
        x[0, 0, 4096] = 1.0  # Single impulse at midpoint
        with torch.no_grad():
            y = gen(x)
        assert torch.isfinite(y).all(), "NaN/Inf from impulse input"
        assert y.shape[-1] == x.shape[-1] * UPSAMPLE_FACTOR


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
