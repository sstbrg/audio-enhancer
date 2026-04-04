"""Tests for the upscale potential analysis module (metrics/upscale_potential.py).

Each test class targets one of the four detection functions plus the composite
scorer.  Synthetic audio is generated on-the-fly via numpy + soundfile so that
no external fixtures are needed.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from models.constants import (
    UPSCALE_BIT_DEPTH_RESIDUAL_THRESHOLD,
    UPSCALE_CANDIDATE_BIT_DEPTHS,
    UPSCALE_CEILING_TOLERANCE,
    UPSCALE_CODEC_CUTOFF_DROP_DB,
    UPSCALE_FFT_N,
    UPSCALE_NOISE_FLOOR_RATIO,
    UPSCALE_ROLLOFF_PERCENT,
    UPSCALE_WEIGHT_BIT_DEPTH,
    UPSCALE_WEIGHT_CODEC,
    UPSCALE_WEIGHT_GAP,
    UPSCALE_WEIGHT_SR_CEILING,
)
from metrics.upscale_potential import (
    _bit_depth_headroom,
    _codec_artifacts,
    _composite_score,
    _sample_rate_ceiling,
    _smooth,
    _spectral_gap,
    _spread_power_spectrum,
    analyze_upscale_potential,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE_48K = 48000
SAMPLE_RATE_96K = 96000
DURATION_S = 2.0  # Short signals are enough for unit tests


def _write_sine(
    path: str,
    sr: int = SAMPLE_RATE_48K,
    freq: float = 440.0,
    duration_s: float = DURATION_S,
    bit_depth: int = 24,
    amplitude: float = 0.9,
) -> str:
    """Write a mono sine wave to *path* and return the path."""
    t = np.arange(int(sr * duration_s)) / sr
    data = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    subtype = f"PCM_{bit_depth}" if bit_depth <= 32 else "FLOAT"
    sf.write(path, data, sr, subtype=subtype)
    return path


def _write_bandlimited(
    path: str,
    sr: int = SAMPLE_RATE_96K,
    max_freq: float = 20000.0,
    duration_s: float = DURATION_S,
    bit_depth: int = 24,
) -> str:
    """Write a signal containing harmonics only up to *max_freq* in a file
    whose Nyquist is sr/2 (potentially much higher).  This simulates a
    file that was upsampled from a lower native rate.
    """
    t = np.arange(int(sr * duration_s)) / sr
    data = np.zeros(len(t), dtype=np.float32)
    # Build a harmonic series up to max_freq
    fundamental = 100.0
    for harmonic in range(1, int(max_freq / fundamental) + 1):
        f = fundamental * harmonic
        if f > max_freq:
            break
        data += (0.5 / harmonic) * np.sin(2.0 * math.pi * f * t).astype(np.float32)
    # Normalise
    peak = np.abs(data).max()
    if peak > 0:
        data = data * 0.9 / peak
    subtype = f"PCM_{bit_depth}"
    sf.write(path, data, sr, subtype=subtype)
    return path


def _write_silence(
    path: str,
    sr: int = SAMPLE_RATE_48K,
    duration_s: float = DURATION_S,
    bit_depth: int = 16,
) -> str:
    """Write a silent file."""
    data = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(path, data, sr, subtype=f"PCM_{bit_depth}")
    return path


def _write_with_hard_cutoff(
    path: str,
    sr: int = SAMPLE_RATE_48K,
    cutoff_hz: float = 16000.0,
    duration_s: float = DURATION_S,
    bit_depth: int = 24,
) -> str:
    """Write broadband noise with a hard low-pass brickwall at *cutoff_hz*.

    This simulates an MP3-encoded file that was transcoded to WAV.
    """
    n = int(sr * duration_s)
    # White noise in frequency domain
    rng = np.random.RandomState(42)
    spectrum = rng.randn(n // 2 + 1) + 1j * rng.randn(n // 2 + 1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    # Hard brick-wall at cutoff
    spectrum[freqs > cutoff_hz] = 0.0
    data = np.fft.irfft(spectrum, n=n).astype(np.float32)
    peak = np.abs(data).max()
    if peak > 0:
        data = data * 0.8 / peak
    sf.write(path, data, sr, subtype=f"PCM_{bit_depth}")
    return path


def _write_quantised(
    path: str,
    sr: int = SAMPLE_RATE_48K,
    true_bits: int = 16,
    container_bits: int = 24,
    duration_s: float = DURATION_S,
) -> str:
    """Write a signal whose actual precision is *true_bits* inside a
    *container_bits* container.  This simulates 16-bit audio stored as 24-bit.
    """
    t = np.arange(int(sr * duration_s)) / sr
    # Multi-frequency signal for realistic content
    data = np.zeros(len(t), dtype=np.float64)
    for f in [220.0, 440.0, 880.0, 1760.0]:
        data += 0.2 * np.sin(2.0 * math.pi * f * t)
    # Quantise to true_bits
    step = 2.0 ** (-(true_bits - 1))
    data = np.round(data / step) * step
    data = data.astype(np.float32)
    sf.write(path, data, sr, subtype=f"PCM_{container_bits}")
    return path


# ---------------------------------------------------------------------------
# Tests: _smooth helper
# ---------------------------------------------------------------------------

class TestSmooth:
    def test_identity_for_small_window(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _smooth(arr, 1)
        np.testing.assert_array_equal(result, arr)

    def test_reduces_variance(self):
        rng = np.random.RandomState(0)
        noisy = rng.randn(200)
        smoothed = _smooth(noisy, 20)
        assert smoothed.std() < noisy.std()

    def test_preserves_length(self):
        arr = np.ones(100)
        result = _smooth(arr, 10)
        assert len(result) == len(arr)

    def test_empty_array(self):
        arr = np.array([])
        result = _smooth(arr, 5)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _spread_power_spectrum
# ---------------------------------------------------------------------------

class TestSpreadPowerSpectrum:
    def test_returns_correct_types(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(power, np.ndarray)
        assert isinstance(sr, int)

    def test_freq_range_matches_nyquist(self, tmp_path):
        sr = SAMPLE_RATE_48K
        path = _write_sine(str(tmp_path / "sine.wav"), sr=sr)
        freqs, _power, returned_sr = _spread_power_spectrum(path)
        assert returned_sr == sr
        # Last freq should be at or near Nyquist
        assert abs(freqs[-1] - sr / 2) < (sr / UPSCALE_FFT_N)

    def test_power_is_non_negative(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        _freqs, power, _sr = _spread_power_spectrum(path)
        assert np.all(power >= 0)

    def test_sine_peak_at_correct_frequency(self, tmp_path):
        freq = 1000.0
        sr = SAMPLE_RATE_48K
        path = _write_sine(str(tmp_path / "sine.wav"), sr=sr, freq=freq, duration_s=1.0)
        freqs, power, _sr = _spread_power_spectrum(path)
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        # Within 1 FFT bin of the true frequency
        bin_width = sr / UPSCALE_FFT_N
        assert abs(peak_freq - freq) < bin_width * 2


# ---------------------------------------------------------------------------
# Tests: _sample_rate_ceiling
# ---------------------------------------------------------------------------

class TestSampleRateCeiling:
    def test_fullband_signal_has_high_utilization(self, tmp_path):
        """A 48kHz sine at 20kHz should show near-full bandwidth."""
        sr = SAMPLE_RATE_48K
        path = _write_sine(str(tmp_path / "hf.wav"), sr=sr, freq=20000.0)
        freqs, power, _ = _spread_power_spectrum(path)
        result = _sample_rate_ceiling(freqs, power, sr)
        assert result["nominal_sr"] == sr
        # High-frequency sine should give high utilisation
        assert result["bandwidth_utilization"] > 0.8

    def test_lowband_signal_has_low_utilization(self, tmp_path):
        """A bandlimited-to-5kHz signal in a 96kHz file should show low utilisation."""
        sr = SAMPLE_RATE_96K
        path = _write_bandlimited(str(tmp_path / "low.wav"), sr=sr, max_freq=5000.0)
        freqs, power, _ = _spread_power_spectrum(path)
        result = _sample_rate_ceiling(freqs, power, sr)
        assert result["nominal_sr"] == sr
        assert result["bandwidth_utilization"] < 0.25

    def test_effective_sr_less_than_nominal_for_bandlimited(self, tmp_path):
        sr = SAMPLE_RATE_96K
        path = _write_bandlimited(str(tmp_path / "bl.wav"), sr=sr, max_freq=15000.0)
        freqs, power, _ = _spread_power_spectrum(path)
        result = _sample_rate_ceiling(freqs, power, sr)
        assert result["effective_sr"] < sr

    def test_silent_file_returns_full_bandwidth(self, tmp_path):
        """Silent file should not crash and should report full bandwidth (edge case)."""
        path = _write_silence(str(tmp_path / "silent.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        result = _sample_rate_ceiling(freqs, power, sr)
        # For zero-power input the code treats it as full bandwidth
        assert result["bandwidth_utilization"] == 1.0

    def test_output_keys(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        result = _sample_rate_ceiling(freqs, power, sr)
        assert "effective_sr" in result
        assert "nominal_sr" in result
        assert "bandwidth_utilization" in result


# ---------------------------------------------------------------------------
# Tests: _codec_artifacts
# ---------------------------------------------------------------------------

class TestCodecArtifacts:
    def test_clean_file_no_artifacts(self, tmp_path):
        """A normal sine wave should not trigger codec detection."""
        path = _write_sine(str(tmp_path / "clean.wav"), freq=440.0)
        freqs, power, sr = _spread_power_spectrum(path)
        result = _codec_artifacts(freqs, power, sr, path)
        assert result["codec_artifacts_detected"] is False
        assert result["likely_codec"] == "none"
        assert result["confidence"] == 0.0

    def test_hard_cutoff_detected(self, tmp_path):
        """Broadband noise with a brickwall at 16kHz should trigger cutoff detection.

        Uses a longer signal (5s) so the spread-sampled FFT has enough frames
        to resolve the sharp spectral transition.
        """
        path = _write_with_hard_cutoff(
            str(tmp_path / "cutoff.wav"),
            sr=SAMPLE_RATE_48K,
            cutoff_hz=16000.0,
            duration_s=5.0,
        )
        freqs, power, sr = _spread_power_spectrum(path)
        result = _codec_artifacts(freqs, power, sr, path)
        assert result["codec_artifacts_detected"] is True
        assert result["cutoff_hz"] is not None
        # Detected cutoff should be near the true cutoff
        assert abs(result["cutoff_hz"] - 16000.0) < 2000.0

    def test_output_keys(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        result = _codec_artifacts(freqs, power, sr, path)
        expected_keys = {
            "codec_artifacts_detected", "likely_codec", "confidence",
            "cutoff_hz", "pre_echo_events", "sbr_detected",
        }
        assert set(result.keys()) == expected_keys

    def test_confidence_range(self, tmp_path):
        """Confidence should always be in [0.0, 1.0)."""
        path = _write_with_hard_cutoff(
            str(tmp_path / "cutoff.wav"),
            sr=SAMPLE_RATE_48K,
            cutoff_hz=16000.0,
        )
        freqs, power, sr = _spread_power_spectrum(path)
        result = _codec_artifacts(freqs, power, sr, path)
        assert 0.0 <= result["confidence"] < 1.0

    def test_silent_file_no_artifacts(self, tmp_path):
        path = _write_silence(str(tmp_path / "silent.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        result = _codec_artifacts(freqs, power, sr, path)
        assert result["codec_artifacts_detected"] is False


# ---------------------------------------------------------------------------
# Tests: _bit_depth_headroom
# ---------------------------------------------------------------------------

class TestBitDepthHeadroom:
    def test_native_24bit_rich_signal_minimal_headroom(self, tmp_path):
        """A signal with many harmonics in 24-bit should show high effective depth.

        A pure sine has very few unique amplitudes and fits in 16-bit easily,
        so we use a multi-harmonic signal with noise to exercise the full
        24-bit range.
        """
        sr = SAMPLE_RATE_48K
        n = int(sr * DURATION_S)
        t = np.arange(n) / sr
        rng = np.random.RandomState(42)
        data = np.zeros(n, dtype=np.float64)
        # Rich harmonic content
        for f in [55.0, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0]:
            data += 0.1 * np.sin(2.0 * math.pi * f * t)
        # Add low-level noise to exercise LSBs
        data += rng.randn(n) * 1e-5
        data = (data * 0.9 / np.abs(data).max()).astype(np.float32)
        path = str(tmp_path / "hires.wav")
        sf.write(path, data, sr, subtype="PCM_24")

        result = _bit_depth_headroom(path)
        assert result["declared_bit_depth"] == 24
        # With noise dithering LSBs, effective depth should be high
        assert result["effective_bit_depth"] >= 20
        assert result["headroom_db"] < 30.0

    def test_16bit_in_24bit_container(self, tmp_path):
        """16-bit audio stored as 24-bit should be detected."""
        path = _write_quantised(
            str(tmp_path / "q16in24.wav"),
            true_bits=16,
            container_bits=24,
        )
        result = _bit_depth_headroom(path)
        assert result["declared_bit_depth"] == 24
        # Effective depth should be detected as 16 or close
        assert result["effective_bit_depth"] <= 20
        assert result["headroom_db"] > 0.0

    def test_silent_file(self, tmp_path):
        path = _write_silence(str(tmp_path / "silent.wav"), bit_depth=24)
        result = _bit_depth_headroom(path)
        assert result["declared_bit_depth"] == 24
        # Silent file: effective depth should be very low, headroom should be large
        assert result["effective_bit_depth"] <= 8
        assert result["headroom_db"] > 0.0

    def test_output_keys(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        result = _bit_depth_headroom(path)
        expected_keys = {"effective_bit_depth", "declared_bit_depth", "headroom_db"}
        assert set(result.keys()) == expected_keys

    def test_8bit_in_16bit_container(self, tmp_path):
        """8-bit audio in 16-bit container should show headroom."""
        path = _write_quantised(
            str(tmp_path / "q8in16.wav"),
            true_bits=8,
            container_bits=16,
        )
        result = _bit_depth_headroom(path)
        assert result["declared_bit_depth"] == 16
        assert result["effective_bit_depth"] <= 8
        assert result["headroom_db"] >= 6.0 * (16 - 8) - 12  # Allow tolerance


# ---------------------------------------------------------------------------
# Tests: _spectral_gap
# ---------------------------------------------------------------------------

class TestSpectralGap:
    def test_fullband_small_gap(self, tmp_path):
        """A 48kHz sine at 20kHz should have a modest gap."""
        sr = SAMPLE_RATE_48K
        path = _write_sine(str(tmp_path / "hf.wav"), sr=sr, freq=20000.0)
        freqs, power, _ = _spread_power_spectrum(path)
        result = _spectral_gap(freqs, power, sr)
        assert result["nyquist_hz"] == sr / 2
        # Rolloff should be near the sine frequency
        assert result["rolloff_hz"] >= 15000.0
        assert result["gap_ratio"] < 0.5

    def test_lowband_large_gap(self, tmp_path):
        """A bandlimited-to-5kHz signal at 96kHz should have a large gap."""
        sr = SAMPLE_RATE_96K
        path = _write_bandlimited(str(tmp_path / "low.wav"), sr=sr, max_freq=5000.0)
        freqs, power, _ = _spread_power_spectrum(path)
        result = _spectral_gap(freqs, power, sr)
        assert result["nyquist_hz"] == sr / 2
        assert result["gap_ratio"] > 0.5
        assert result["gap_hz"] > 30000.0

    def test_gap_is_non_negative(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        result = _spectral_gap(freqs, power, sr)
        assert result["gap_hz"] >= 0.0
        assert result["gap_ratio"] >= 0.0

    def test_output_keys(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        result = _spectral_gap(freqs, power, sr)
        expected_keys = {"rolloff_hz", "nyquist_hz", "gap_hz", "gap_ratio"}
        assert set(result.keys()) == expected_keys

    def test_silent_file_full_gap(self, tmp_path):
        path = _write_silence(str(tmp_path / "silent.wav"))
        freqs, power, sr = _spread_power_spectrum(path)
        result = _spectral_gap(freqs, power, sr)
        assert result["gap_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Tests: _composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_perfect_file_scores_zero(self):
        """Full bandwidth, no codec, full bit depth, no gap = score 0."""
        sr_ceil = {"bandwidth_utilization": 1.0, "effective_sr": 96000, "nominal_sr": 96000}
        codec = {"codec_artifacts_detected": False, "confidence": 0.0}
        bit_depth = {"effective_bit_depth": 24, "declared_bit_depth": 24, "headroom_db": 0.0}
        gap = {"gap_ratio": 0.0, "rolloff_hz": 48000.0, "nyquist_hz": 48000.0, "gap_hz": 0.0}
        score = _composite_score(sr_ceil, codec, bit_depth, gap)
        assert score == 0

    def test_worst_case_scores_high(self):
        """Minimal bandwidth, codec artifacts, bit waste, huge gap = high score."""
        sr_ceil = {"bandwidth_utilization": 0.1, "effective_sr": 9600, "nominal_sr": 96000}
        codec = {"codec_artifacts_detected": True, "confidence": 0.9}
        bit_depth = {"effective_bit_depth": 8, "declared_bit_depth": 24, "headroom_db": 96.0}
        gap = {"gap_ratio": 0.9, "rolloff_hz": 4800.0, "nyquist_hz": 48000.0, "gap_hz": 43200.0}
        score = _composite_score(sr_ceil, codec, bit_depth, gap)
        assert score >= 70

    def test_score_in_range(self):
        """Score should always be in [0, 100]."""
        sr_ceil = {"bandwidth_utilization": 0.5, "effective_sr": 48000, "nominal_sr": 96000}
        codec = {"codec_artifacts_detected": True, "confidence": 0.5}
        bit_depth = {"effective_bit_depth": 16, "declared_bit_depth": 24, "headroom_db": 48.0}
        gap = {"gap_ratio": 0.5, "rolloff_hz": 24000.0, "nyquist_hz": 48000.0, "gap_hz": 24000.0}
        score = _composite_score(sr_ceil, codec, bit_depth, gap)
        assert 0 <= score <= 100

    def test_weights_sum_to_one(self):
        """Sanity check: composite weights from constants must sum to 1.0."""
        total = UPSCALE_WEIGHT_SR_CEILING + UPSCALE_WEIGHT_CODEC + UPSCALE_WEIGHT_BIT_DEPTH + UPSCALE_WEIGHT_GAP
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests: analyze_upscale_potential (public API / integration)
# ---------------------------------------------------------------------------

class TestAnalyzeUpscalePotential:
    def test_returns_all_keys(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        result = analyze_upscale_potential(path)
        expected_keys = {
            "sample_rate_ceiling", "codec_artifacts", "bit_depth_headroom",
            "spectral_gap", "composite_score", "summary",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_score_is_integer(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        result = analyze_upscale_potential(path)
        assert isinstance(result["composite_score"], int)

    def test_summary_is_nonempty_string(self, tmp_path):
        path = _write_sine(str(tmp_path / "sine.wav"))
        result = analyze_upscale_potential(path)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_bandlimited_scores_higher_than_fullband(self, tmp_path):
        """A 5kHz-bandlimited file at 96kHz should score higher (more potential)
        than a full-bandwidth 48kHz file.
        """
        full = _write_sine(str(tmp_path / "full.wav"), sr=SAMPLE_RATE_48K, freq=20000.0)
        limited = _write_bandlimited(
            str(tmp_path / "limited.wav"), sr=SAMPLE_RATE_96K, max_freq=5000.0,
        )
        score_full = analyze_upscale_potential(full)["composite_score"]
        score_limited = analyze_upscale_potential(limited)["composite_score"]
        assert score_limited > score_full

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_upscale_potential("/nonexistent/file.wav")

    def test_hard_cutoff_file_detects_codec(self, tmp_path):
        """A file with a hard 16kHz cutoff should be flagged for codec artifacts."""
        path = _write_with_hard_cutoff(
            str(tmp_path / "mp3ish.wav"),
            sr=SAMPLE_RATE_48K,
            cutoff_hz=16000.0,
            duration_s=5.0,
        )
        result = analyze_upscale_potential(path)
        assert result["codec_artifacts"]["codec_artifacts_detected"] is True

    def test_16bit_in_24bit_detected(self, tmp_path):
        """16-bit audio in 24-bit container should show bit depth headroom."""
        path = _write_quantised(
            str(tmp_path / "q16in24.wav"),
            true_bits=16,
            container_bits=24,
        )
        result = analyze_upscale_potential(path)
        bd = result["bit_depth_headroom"]
        assert bd["declared_bit_depth"] == 24
        assert bd["effective_bit_depth"] <= 20
        assert bd["headroom_db"] > 0.0
