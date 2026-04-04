"""Tests for the inference pipeline (enhance.py / AudioEnhancer)."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from models.constants import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE


def write_sine_wav(
    path: str | Path,
    sample_rate: int,
    duration: float = 0.5,
    channels: int = 1,
    freq: float = 440.0,
) -> Path:
    """Write a synthetic sine-wave WAV file."""
    path = Path(path)
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    wave = 0.5 * np.sin(2 * math.pi * freq * t)
    if channels == 2:
        data = np.stack([wave, wave * 0.8], axis=-1)
    else:
        data = wave[:, np.newaxis]
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


@pytest.fixture()
def config_path(tmp_path):
    """Minimal config for AudioEnhancer with all external stages disabled."""
    cfg = tmp_path / "test_config.yaml"
    cfg.write_text(
        "pipeline:\n"
        "  apollo: false\n"
        "  audiosr: false\n"
        "  gan_upsample: false\n"
        "output:\n"
        f"  sample_rate: {OUTPUT_SAMPLE_RATE}\n"
        "  bit_depth: 24\n"
        "  format: wav\n"
        "audiosr:\n"
        "  model_name: basic\n"
        "  ddim_steps: 50\n"
        "  guidance_scale: 3.5\n"
    )
    return str(cfg)


class TestAudioEnhancerInstantiation:
    def test_instantiates_without_checkpoint(self, config_path):
        from enhance import AudioEnhancer
        enhancer = AudioEnhancer(config_path=config_path, device="cpu")
        assert enhancer.gan_model is None
        assert enhancer.apollo_model is None
        assert enhancer.audiosr_model is None

    def test_device_is_cpu(self, config_path):
        from enhance import AudioEnhancer
        enhancer = AudioEnhancer(config_path=config_path, device="cpu")
        assert enhancer.device == torch.device("cpu")


class TestEnhanceFileSampleRates:
    """Test the fallback resampling path (no GAN, no Apollo, no AudioSR)."""

    def _enhance(self, config_path, input_sr, channels=1, duration=0.2):
        from enhance import AudioEnhancer
        enhancer = AudioEnhancer(config_path=config_path, device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = str(Path(tmpdir) / "input.wav")
            out_path = str(Path(tmpdir) / "output.wav")
            write_sine_wav(in_path, sample_rate=input_sr, duration=duration, channels=channels)
            enhancer.enhance_file(
                in_path, out_path,
                use_apollo=False,
                use_audiosr=False,
                use_gan=False,
            )
            info = sf.info(out_path)
            return info

    def test_44100hz_input_produces_output(self, config_path):
        info = self._enhance(config_path, input_sr=44100)
        assert info.frames > 0

    def test_48000hz_input_produces_output(self, config_path):
        info = self._enhance(config_path, input_sr=48000)
        assert info.frames > 0

    def test_96000hz_input_produces_output(self, config_path):
        info = self._enhance(config_path, input_sr=96000)
        assert info.frames > 0

    def test_output_is_wav(self, config_path):
        info = self._enhance(config_path, input_sr=48000)
        assert info.format == "WAV"

    def test_44100_upsampled_to_target_sr(self, config_path):
        """44.1kHz input should be resampled to the configured target SR."""
        info = self._enhance(config_path, input_sr=44100)
        assert info.samplerate == OUTPUT_SAMPLE_RATE

    def test_48000_upsampled_to_target_sr(self, config_path):
        info = self._enhance(config_path, input_sr=48000)
        assert info.samplerate == OUTPUT_SAMPLE_RATE

    def test_stereo_input(self, config_path):
        info = self._enhance(config_path, input_sr=48000, channels=2)
        assert info.frames > 0


class TestEnhanceFileWithGAN:
    """Test the GAN inference path using a freshly initialised (untrained) generator."""

    def _build_checkpoint(self, tmp_path: Path) -> str:
        """Serialise a random-init generator checkpoint that AudioEnhancer can load."""
        from models.constants import (
            GENERATOR_CHANNELS,
            GENERATOR_RESBLOCK_DILATIONS,
            GENERATOR_RESBLOCK_KERNELS,
            GENERATOR_UPSAMPLE_KERNELS,
            GENERATOR_UPSAMPLE_RATES,
        )
        from models.generator import Generator

        gen = Generator()
        ckpt = {
            "generator": gen.state_dict(),
            "config": {
                "gan": {
                    "generator": {
                        "channels": GENERATOR_CHANNELS,
                        "upsample_rates": GENERATOR_UPSAMPLE_RATES,
                        "upsample_kernel_sizes": GENERATOR_UPSAMPLE_KERNELS,
                        "resblock_kernel_sizes": GENERATOR_RESBLOCK_KERNELS,
                        "resblock_dilation_sizes": GENERATOR_RESBLOCK_DILATIONS,
                    }
                }
            },
        }
        ckpt_path = str(tmp_path / "test_checkpoint.pt")
        torch.save(ckpt, ckpt_path)
        return ckpt_path

    def _make_gan_config(self, tmp_path: Path) -> str:
        cfg = tmp_path / "gan_config.yaml"
        cfg.write_text(
            "pipeline:\n"
            "  apollo: false\n"
            "  audiosr: false\n"
            "  gan_upsample: true\n"
            "output:\n"
            f"  sample_rate: {OUTPUT_SAMPLE_RATE}\n"
            "  bit_depth: 24\n"
            "  format: wav\n"
            "audiosr:\n"
            "  model_name: basic\n"
            "  ddim_steps: 50\n"
            "  guidance_scale: 3.5\n"
        )
        return str(cfg)

    def test_gan_forward_48khz_input(self, tmp_path):
        from enhance import AudioEnhancer
        ckpt_path = self._build_checkpoint(tmp_path)
        cfg_path = self._make_gan_config(tmp_path)
        enhancer = AudioEnhancer(config_path=cfg_path, gan_checkpoint=ckpt_path, device="cpu")
        assert enhancer.gan_model is not None

        in_path = str(tmp_path / "in_48k.wav")
        out_path = str(tmp_path / "out_96k.wav")
        write_sine_wav(in_path, sample_rate=INPUT_SAMPLE_RATE, duration=0.2)
        enhancer.enhance_file(in_path, out_path, use_apollo=False, use_audiosr=False, use_gan=True)

        info = sf.info(out_path)
        assert info.samplerate == OUTPUT_SAMPLE_RATE
        assert info.frames > 0

    def test_gan_forward_44khz_input_resampled(self, tmp_path):
        """44.1kHz input must be resampled to 48kHz before the GAN."""
        from enhance import AudioEnhancer
        ckpt_path = self._build_checkpoint(tmp_path)
        cfg_path = self._make_gan_config(tmp_path)
        enhancer = AudioEnhancer(config_path=cfg_path, gan_checkpoint=ckpt_path, device="cpu")

        in_path = str(tmp_path / "in_44k.wav")
        out_path = str(tmp_path / "out_96k_from44k.wav")
        write_sine_wav(in_path, sample_rate=44100, duration=0.2)
        enhancer.enhance_file(in_path, out_path, use_apollo=False, use_audiosr=False, use_gan=True)

        info = sf.info(out_path)
        assert info.samplerate == OUTPUT_SAMPLE_RATE

    def test_gan_output_no_clip(self, tmp_path):
        """Output of GAN pipeline should be clamped to [-1, 1] by enhance.py."""
        from enhance import AudioEnhancer
        ckpt_path = self._build_checkpoint(tmp_path)
        cfg_path = self._make_gan_config(tmp_path)
        enhancer = AudioEnhancer(config_path=cfg_path, gan_checkpoint=ckpt_path, device="cpu")

        in_path = str(tmp_path / "in.wav")
        out_path = str(tmp_path / "out.wav")
        write_sine_wav(in_path, sample_rate=INPUT_SAMPLE_RATE, duration=0.2)
        enhancer.enhance_file(in_path, out_path, use_apollo=False, use_audiosr=False, use_gan=True)

        data, _ = sf.read(out_path)
        assert np.all(data >= -1.0001)
        assert np.all(data <= 1.0001)


class TestGANChunkedProcessing:
    """Test the chunked GAN processing path used for long audio."""

    def test_chunked_output_length(self):
        """_gan_chunked should produce approximately the right number of output samples."""
        from enhance import AudioEnhancer
        from models.generator import Generator

        gen = Generator()
        gen.eval()

        # Build a minimal enhancer with the generator attached
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "cfg.yaml"
            cfg.write_text(
                "pipeline:\n  apollo: false\n  audiosr: false\n  gan_upsample: true\n"
                f"output:\n  sample_rate: {OUTPUT_SAMPLE_RATE}\n  bit_depth: 24\n  format: wav\n"
                "audiosr:\n  model_name: basic\n  ddim_steps: 50\n  guidance_scale: 3.5\n"
            )
            enhancer = AudioEnhancer(config_path=str(cfg), device="cpu")
            enhancer.gan_model = gen

            # 2 seconds at 48kHz — longer than the 10s chunk, but short enough to be fast
            samples = 48000 * 2
            audio = torch.randn(1, 1, samples) * 0.1

            chunk_size = 48000  # 1s chunks (smaller than default for speed)
            out = enhancer._gan_chunked(audio, chunk_size, overlap=480)

        expected = samples * gen.upsample_factor
        # Allow some tolerance due to trimming at boundaries
        assert abs(out.shape[-1] - expected) < chunk_size * gen.upsample_factor
        assert torch.isfinite(out).all()
