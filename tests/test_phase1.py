"""Tests for Phase 1: HF band loss, degradation classes, degradation chain, and dataset."""

import math

import pytest
import torch

from models.constants import OUTPUT_SAMPLE_RATE
from models.losses import HighFrequencyBandLoss


def make_audio(batch: int = 2, samples: int = 22050) -> torch.Tensor:
    """Return (batch, 1, samples) sine wave."""
    t = torch.linspace(0, samples / OUTPUT_SAMPLE_RATE, samples)
    wave = torch.sin(2 * math.pi * 440.0 * t)
    return wave.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1).clone()


class TestHighFrequencyBandLoss:
    def test_scalar_output(self):
        loss_fn = HighFrequencyBandLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=2, samples=22050)
        y_hat = make_audio(batch=2, samples=22050)
        loss = loss_fn(y_hat, y)
        assert loss.dim() == 0

    def test_no_nan(self):
        loss_fn = HighFrequencyBandLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=2, samples=22050)
        y_hat = torch.randn_like(y) * 0.1
        loss = loss_fn(y_hat, y)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_same_input_low_loss(self):
        loss_fn = HighFrequencyBandLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=1, samples=22050)
        loss = loss_fn(y, y)
        assert loss.item() < 0.01

    def test_different_input_higher_loss(self):
        loss_fn = HighFrequencyBandLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=1, samples=22050)
        y_hat = torch.randn_like(y) * 0.5
        loss_diff = loss_fn(y_hat, y)
        loss_same = loss_fn(y, y)
        assert loss_diff.item() > loss_same.item()

    def test_accepts_2d_input(self):
        loss_fn = HighFrequencyBandLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        y = make_audio(batch=2, samples=16384).squeeze(1)
        y_hat = torch.randn_like(y) * 0.5
        loss = loss_fn(y_hat, y)
        assert torch.isfinite(loss)

    def test_band_mask_shape(self):
        loss_fn = HighFrequencyBandLoss(sample_rate=OUTPUT_SAMPLE_RATE)
        assert loss_fn.band_mask.dim() == 1
        assert loss_fn.band_mask.any(), "Band mask should select some frequency bins"


class TestDegradations:
    """Test individual degradation classes."""

    def test_bad_eq(self):
        from data.degradations import BadEQDegradation
        deg = BadEQDegradation(severity=0.5)
        waveform = torch.randn(1, 48000)
        result = deg(waveform, 48000)
        assert result.shape == waveform.shape
        assert torch.isfinite(result).all()

    def test_clipping_hard(self):
        from data.degradations import ClippingDegradation
        deg = ClippingDegradation(severity=0.8)
        waveform = torch.randn(1, 16000)
        result = deg(waveform, 48000)
        assert result.shape == waveform.shape
        assert torch.isfinite(result).all()

    def test_sample_rate_degradation(self):
        from data.degradations import SampleRateDegradation
        deg = SampleRateDegradation(severity=0.5)
        waveform = torch.randn(1, 48000)
        result = deg(waveform, 48000)
        assert result.shape == waveform.shape
        assert torch.isfinite(result).all()

    def test_noise_degradation(self):
        from data.degradations import NoiseDegradation
        deg = NoiseDegradation(severity=0.5)
        waveform = torch.randn(1, 16000)
        result = deg(waveform, 48000)
        assert result.shape == waveform.shape
        assert torch.isfinite(result).all()

    def test_dynamic_compression(self):
        from data.degradations import DynamicCompressionDegradation
        deg = DynamicCompressionDegradation(severity=0.5)
        waveform = torch.randn(1, 16000) * 0.5
        result = deg(waveform, 48000)
        assert result.shape == waveform.shape
        assert torch.isfinite(result).all()

    def test_stereo_damage_mono_passthrough(self):
        from data.degradations import StereoDamageDegradation
        deg = StereoDamageDegradation(severity=0.5)
        mono = torch.randn(1, 16000)
        result = deg(mono, 48000)
        assert result.shape == mono.shape, "Mono should pass through unchanged shape"

    def test_stereo_damage_stereo(self):
        from data.degradations import StereoDamageDegradation
        deg = StereoDamageDegradation(severity=0.5)
        stereo = torch.randn(2, 16000)
        result = deg(stereo, 48000)
        assert result.shape == stereo.shape
        assert torch.isfinite(result).all()

    def test_severity_update(self):
        from data.degradations import Degradation
        deg = Degradation(severity=0.5)
        deg.set_severity(0.8)
        assert deg.severity == 0.8
        deg.set_severity(1.5)  # Clamped
        assert deg.severity == 1.0
        deg.set_severity(-0.1)
        assert deg.severity == 0.0


class TestDegradationChain:
    def test_chain_always_applies_at_least_one(self):
        from data.degradation_chain import DegradationChain
        chain = DegradationChain({
            "p_codec": 0.0, "p_eq": 0.0, "p_compression": 0.0,
            "p_clipping": 0.0, "p_sample_rate": 0.0, "p_noise": 0.0,
            "p_stereo": 0.0, "max_chain_length": 3,
        })
        waveform = torch.randn(1, 16000)
        result = chain(waveform, 48000)
        # Should still apply at least one degradation (codec fallback)
        assert result.shape == waveform.shape

    def test_chain_respects_max_length(self):
        from data.degradation_chain import DegradationChain
        chain = DegradationChain({
            "p_codec": 1.0, "p_eq": 1.0, "p_compression": 1.0,
            "p_clipping": 1.0, "p_sample_rate": 1.0, "p_noise": 1.0,
            "p_stereo": 0.0, "max_chain_length": 2,
        })
        built = chain._build_chain(is_stereo=False)
        assert len(built) <= 2

    def test_curriculum_severity(self):
        from data.degradation_chain import DegradationChain
        from models.constants import CURRICULUM_MILD_MAX
        chain = DegradationChain({
            "curriculum": {"enabled": True, "warmup_epochs": 10, "linear_ramp_epochs": 40},
            "max_chain_length": 3,
        })
        chain.update_epoch(0)
        assert chain.current_severity == pytest.approx(CURRICULUM_MILD_MAX)

        chain.update_epoch(10)  # Start of ramp
        assert chain.current_severity == pytest.approx(CURRICULUM_MILD_MAX)

        chain.update_epoch(50)  # End of ramp
        assert chain.current_severity == pytest.approx(1.0)

    def test_curriculum_disabled(self):
        from data.degradation_chain import DegradationChain
        chain = DegradationChain({
            "curriculum": {"enabled": False},
            "max_chain_length": 3,
        })
        chain.update_epoch(0)
        assert chain.current_severity == 1.0
