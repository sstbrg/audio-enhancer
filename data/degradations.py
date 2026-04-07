"""Audio degradation transforms for Phase 1 training.

Each degradation is a callable class that takes (waveform, sample_rate) and
returns a degraded waveform. Severity controls degradation intensity (0=mild,
1=extreme) and is updated by the curriculum scheduler.
"""

import math
import random
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio.functional as AF

from models.constants import (
    CLIP_HARD_THRESHOLD_RANGE,
    CLIP_SOFT_GAIN_RANGE,
    CODEC_BITRATE_RANGES,
    CODEC_DOUBLE_ENCODE_PROB,
    CODEC_TYPES,
    COMP_ATTACK_RANGE,
    COMP_RATIO_RANGE,
    COMP_RELEASE_RANGE,
    COMP_THRESHOLD_RANGE,
    DEGRADED_BIT_DEPTHS,
    DEGRADED_SAMPLE_RATES,
    EQ_BANDS_RANGE,
    EQ_FREQ_RANGE,
    EQ_GAIN_RANGE,
    EQ_Q_RANGE,
    HUM_FREQ_OPTIONS,
    HUM_HARMONICS,
    NOISE_LEVEL_RANGE,
    STEREO_CROSSTALK_RANGE,
    STEREO_DELAY_RANGE,
    STEREO_WIDTH_RANGE,
)


class Degradation:
    """Base class for audio degradations."""

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        raise NotImplementedError

    def set_severity(self, severity: float):
        self.severity = max(0.0, min(1.0, severity))

    def _lerp(self, low: float, high: float) -> float:
        """Interpolate between low and high based on severity."""
        return low + self.severity * (high - low)


class CodecDegradation(Degradation):
    """Simulate lossy codec artifacts via ffmpeg encode/decode round-trip."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        codec = random.choice(CODEC_TYPES)
        bitrates = CODEC_BITRATE_RANGES[codec]

        # Severity controls which bitrates are available: low severity = high bitrates
        max_idx = max(0, int(len(bitrates) * self.severity))
        available = bitrates[: max_idx + 1] if max_idx > 0 else [bitrates[-1]]
        bitrate = random.choice(available)

        waveform = self._encode_decode(waveform, sample_rate, codec, bitrate)

        # Double-encode with small probability
        if random.random() < CODEC_DOUBLE_ENCODE_PROB:
            codec2 = random.choice(CODEC_TYPES)
            bitrate2 = random.choice(CODEC_BITRATE_RANGES[codec2])
            waveform = self._encode_decode(waveform, sample_rate, codec2, bitrate2)

        return waveform

    def _encode_decode(
        self, waveform: torch.Tensor, sample_rate: int, codec: str, bitrate: int
    ) -> torch.Tensor:
        """Encode to lossy format and decode back."""
        import soundfile as sf

        channels = waveform.shape[0]
        in_path = None
        out_path = None
        decoded_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                in_path = f.name
            with tempfile.NamedTemporaryFile(suffix=f".{codec}", delete=False) as f:
                out_path = f.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                decoded_path = f.name

            # Write input
            sf.write(in_path, waveform.T.numpy(), sample_rate)

            # Map codec to ffmpeg encoder
            encoder_map = {
                "mp3": "libmp3lame",
                "aac": "aac",
                "ogg": "libvorbis",
                "opus": "libopus",
                "wma": "wmav2",
            }
            encoder = encoder_map[codec]

            # Encode
            cmd = [
                "ffmpeg", "-y", "-i", in_path,
                "-c:a", encoder, "-b:a", f"{bitrate}k",
                out_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)

            # Decode back to wav
            cmd = ["ffmpeg", "-y", "-i", out_path, decoded_path]
            subprocess.run(cmd, capture_output=True, timeout=30)

            # Read back
            data, _ = sf.read(decoded_path, dtype="float32")
            result = torch.from_numpy(data).T  # (channels, samples)
            if result.dim() == 1:
                result = result.unsqueeze(0)

            # Match original length
            orig_len = waveform.shape[-1]
            if result.shape[-1] > orig_len:
                result = result[..., :orig_len]
            elif result.shape[-1] < orig_len:
                result = torch.nn.functional.pad(result, (0, orig_len - result.shape[-1]))

            # Match channel count
            if result.shape[0] != channels:
                if channels == 1:
                    result = result[:1]
                else:
                    result = result.expand(channels, -1)

            return result
        except Exception:
            return waveform
        finally:
            for p in [in_path, out_path, decoded_path]:
                if p:
                    Path(p).unlink(missing_ok=True)


class BadEQDegradation(Degradation):
    """Apply random parametric EQ to simulate poor mastering or equipment."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        n_bands = random.randint(*EQ_BANDS_RANGE)
        for _ in range(n_bands):
            # Log-distributed center frequency
            log_lo = math.log10(EQ_FREQ_RANGE[0])
            log_hi = math.log10(EQ_FREQ_RANGE[1])
            center_freq = 10 ** random.uniform(log_lo, log_hi)

            # Severity scales the gain range
            max_gain = self._lerp(3.0, EQ_GAIN_RANGE[1])
            gain_db = random.uniform(-max_gain, max_gain)
            q = random.uniform(*EQ_Q_RANGE)

            waveform = AF.equalizer_biquad(
                waveform, sample_rate, center_freq, gain_db, q
            )

        return waveform


class DynamicCompressionDegradation(Degradation):
    """Simulate over-compression and brick-wall limiting."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Severity controls compression intensity
        threshold_db = self._lerp(
            COMP_THRESHOLD_RANGE[1], COMP_THRESHOLD_RANGE[0]
        )  # More severe = lower threshold
        ratio = self._lerp(COMP_RATIO_RANGE[0], COMP_RATIO_RANGE[1])
        attack_ms = random.uniform(*COMP_ATTACK_RANGE)
        release_ms = random.uniform(*COMP_RELEASE_RANGE)

        threshold = 10 ** (threshold_db / 20.0)

        # Simple envelope-following compressor
        attack_coeff = math.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
        release_coeff = math.exp(-1.0 / (release_ms * sample_rate / 1000.0))

        envelope = torch.zeros(waveform.shape[-1])
        abs_signal = waveform.abs().max(dim=0).values  # Peak across channels

        for i in range(1, len(envelope)):
            if abs_signal[i] > envelope[i - 1]:
                envelope[i] = attack_coeff * envelope[i - 1] + (1 - attack_coeff) * abs_signal[i]
            else:
                envelope[i] = release_coeff * envelope[i - 1]

        # Compute gain reduction
        gain = torch.ones_like(envelope)
        above = envelope > threshold
        if above.any():
            # Compression: output_db = threshold_db + (input_db - threshold_db) / ratio
            db_over = 20 * torch.log10(envelope[above] / threshold + 1e-10)
            db_reduction = db_over * (1.0 - 1.0 / ratio)
            gain[above] = 10 ** (-db_reduction / 20.0)

        return waveform * gain.unsqueeze(0)


class ClippingDegradation(Degradation):
    """Apply hard or soft clipping to simulate ADC overload or saturation."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() < 0.5:
            # Hard clipping
            threshold = self._lerp(
                CLIP_HARD_THRESHOLD_RANGE[1], CLIP_HARD_THRESHOLD_RANGE[0]
            )
            return torch.clamp(waveform, -threshold, threshold)
        else:
            # Soft clipping (tanh saturation)
            gain = self._lerp(CLIP_SOFT_GAIN_RANGE[0], CLIP_SOFT_GAIN_RANGE[1])
            return torch.tanh(gain * waveform) / torch.tanh(torch.tensor(gain))


class SampleRateDegradation(Degradation):
    """Reduce sample rate and/or bit depth, then restore."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        degrade_type = random.choice(["sr", "bitdepth", "both"])

        if degrade_type in ("sr", "both"):
            # Severity determines how low we go
            available_srs = [s for s in DEGRADED_SAMPLE_RATES if s < sample_rate]
            if available_srs:
                max_idx = max(0, int(len(available_srs) * self.severity))
                target_sr = available_srs[max_idx] if max_idx < len(available_srs) else available_srs[-1]
                # Downsample then upsample
                down = AF.resample(waveform, sample_rate, target_sr)
                waveform = AF.resample(down, target_sr, sample_rate)

        if degrade_type in ("bitdepth", "both"):
            bits = random.choice(DEGRADED_BIT_DEPTHS)
            # Quantize
            scale = 2 ** (bits - 1)
            waveform = torch.round(waveform * scale) / scale

        return waveform


class NoiseDegradation(Degradation):
    """Add various types of noise to the signal."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        noise_type = random.choice(["white", "pink", "hum", "hiss"])
        level_db = self._lerp(NOISE_LEVEL_RANGE[0], NOISE_LEVEL_RANGE[1])
        amplitude = 10 ** (level_db / 20.0)

        channels, samples = waveform.shape
        t = torch.arange(samples, dtype=torch.float32) / sample_rate

        if noise_type == "white":
            noise = torch.randn(channels, samples) * amplitude

        elif noise_type == "pink":
            # Approximate 1/f noise by filtering white noise
            white = torch.randn(channels, samples)
            # Simple 1/f approximation via cumulative averaging
            fft = torch.fft.rfft(white)
            freqs = torch.fft.rfftfreq(samples, 1.0 / sample_rate)
            freqs[0] = 1.0  # Avoid division by zero
            fft = fft / torch.sqrt(freqs).unsqueeze(0)
            noise = torch.fft.irfft(fft, n=samples) * amplitude

        elif noise_type == "hum":
            base_freq = random.choice(HUM_FREQ_OPTIONS)
            noise = torch.zeros(channels, samples)
            for h in range(1, HUM_HARMONICS + 1):
                harmonic_amp = amplitude / h  # Harmonics decay
                noise += harmonic_amp * torch.sin(
                    2 * math.pi * base_freq * h * t
                ).unsqueeze(0)

        elif noise_type == "hiss":
            # High-frequency noise (above 4kHz)
            white = torch.randn(channels, samples)
            fft = torch.fft.rfft(white)
            freqs = torch.fft.rfftfreq(samples, 1.0 / sample_rate)
            # Highpass at 4kHz
            mask = (freqs >= 4000).float()
            fft = fft * mask.unsqueeze(0)
            noise = torch.fft.irfft(fft, n=samples) * amplitude

        else:
            noise = torch.zeros(channels, samples)

        return waveform + noise


class StereoDamageDegradation(Degradation):
    """Damage stereo imaging — only applied to stereo signals."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.shape[0] < 2:
            return waveform  # Skip for mono

        damage_type = random.choice(
            ["width_narrow", "phase_issue", "ms_imbalance", "crosstalk"]
        )

        left, right = waveform[0], waveform[1]
        mid = (left + right) / 2
        side = (left - right) / 2

        if damage_type == "width_narrow":
            w = random.uniform(*STEREO_WIDTH_RANGE)
            left = (1 - w) * mid + w * left
            right = (1 - w) * mid + w * right

        elif damage_type == "phase_issue":
            # Random delay on one channel
            delay_ms = random.uniform(*STEREO_DELAY_RANGE)
            delay_samples = int(delay_ms * sample_rate / 1000)
            if delay_samples > 0:
                if random.random() < 0.5:
                    right = torch.roll(right, delay_samples)
                    right[:delay_samples] = 0
                else:
                    left = torch.roll(left, delay_samples)
                    left[:delay_samples] = 0

        elif damage_type == "ms_imbalance":
            # Boost mid or side
            if random.random() < 0.5:
                boost_db = self._lerp(3.0, 12.0)
                mid = mid * (10 ** (boost_db / 20.0))
            else:
                boost_db = self._lerp(3.0, 12.0)
                side = side * (10 ** (boost_db / 20.0))
            left = mid + side
            right = mid - side

        elif damage_type == "crosstalk":
            mix = random.uniform(*STEREO_CROSSTALK_RANGE)
            new_left = left + mix * right
            new_right = right + mix * left
            left, right = new_left, new_right

        return torch.stack([left, right])
