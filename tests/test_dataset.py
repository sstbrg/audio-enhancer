"""Tests for AudioSRDataset — creates synthetic audio files for testing."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from models.constants import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE


def write_sine_wav(
    path: str | Path,
    sample_rate: int,
    duration: float = 1.0,
    channels: int = 1,
    freq: float = 440.0,
    subtype: str = "PCM_16",
) -> Path:
    """Write a synthetic sine-wave WAV file."""
    path = Path(path)
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    wave = 0.5 * np.sin(2 * math.pi * freq * t)
    if channels == 2:
        wave = np.stack([wave, wave * 0.8], axis=-1)
    else:
        wave = wave[:, np.newaxis]
    sf.write(str(path), wave, sample_rate, subtype=subtype)
    return path


@pytest.fixture()
def hires_dir(tmp_path):
    """Temp dir with a single 96kHz/24-bit WAV."""
    write_sine_wav(tmp_path / "hires.wav", sample_rate=96000, subtype="PCM_24")
    return tmp_path


@pytest.fixture()
def midres_dir(tmp_path):
    """Temp dir with a single 48kHz/16-bit WAV."""
    write_sine_wav(tmp_path / "midres.wav", sample_rate=48000, subtype="PCM_16")
    return tmp_path


@pytest.fixture()
def standard_dir(tmp_path):
    """Temp dir with a single 44.1kHz/16-bit WAV."""
    write_sine_wav(tmp_path / "standard.wav", sample_rate=44100, subtype="PCM_16")
    return tmp_path


@pytest.fixture()
def stereo_dir(tmp_path):
    """Temp dir with a stereo 96kHz WAV."""
    write_sine_wav(tmp_path / "stereo.wav", sample_rate=96000, channels=2, subtype="PCM_24")
    return tmp_path


@pytest.fixture()
def multi_tier_dir(tmp_path):
    """Temp dir with one file per quality tier."""
    write_sine_wav(tmp_path / "hires.wav", sample_rate=96000, subtype="PCM_24")
    write_sine_wav(tmp_path / "midres.wav", sample_rate=48000, subtype="PCM_16")
    write_sine_wav(tmp_path / "std.wav", sample_rate=44100, subtype="PCM_16")
    return tmp_path


class TestAudioSRDatasetInstantiation:
    def test_loads_hires_files(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir))
        assert len(ds) == 1

    def test_loads_standard_files(self, standard_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(standard_dir))
        assert len(ds) == 1

    def test_raises_on_empty_dir(self, tmp_path):
        from data.dataset import AudioSRDataset
        with pytest.raises(RuntimeError, match="No audio files found"):
            AudioSRDataset(str(tmp_path))

    def test_multi_tier_count(self, multi_tier_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(multi_tier_dir))
        assert len(ds) == 3


class TestQualityTierClassification:
    def test_hires_classified_correctly(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir))
        assert len(ds.hires_indices) == 1
        assert len(ds.midres_indices) == 0
        assert len(ds.standard_indices) == 0

    def test_midres_classified_correctly(self, midres_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(midres_dir))
        assert len(ds.midres_indices) == 1
        assert len(ds.hires_indices) == 0

    def test_standard_classified_correctly(self, standard_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(standard_dir))
        assert len(ds.standard_indices) == 1
        assert len(ds.hires_indices) == 0

    def test_multi_tier_classification(self, multi_tier_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(multi_tier_dir))
        assert len(ds.hires_indices) == 1
        assert len(ds.midres_indices) == 1
        assert len(ds.standard_indices) == 1


class TestAudioSRDatasetOutput:
    def test_hires_pair_shapes(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir), segment_length=4096)
        lr, hr = ds[0]
        # LR should be segment_length, HR should be segment_length * upsample_factor
        assert lr.shape[-1] == 4096
        assert hr.shape[-1] == 4096 * (OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE)

    def test_standard_pair_shapes(self, standard_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(standard_dir), segment_length=4096)
        lr, hr = ds[0]
        assert lr.shape[-1] == 4096
        assert hr.shape[-1] == 4096 * (OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE)

    def test_output_is_mono(self, stereo_dir):
        """Stereo input should be mixed to mono by the dataset."""
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(stereo_dir), segment_length=4096)
        lr, hr = ds[0]
        assert lr.shape[0] == 1
        assert hr.shape[0] == 1

    def test_tensors_are_float32(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir), segment_length=4096)
        lr, hr = ds[0]
        assert lr.dtype == torch.float32
        assert hr.dtype == torch.float32

    def test_values_in_range(self, hires_dir):
        """Normalized output should be within [-1, 1]."""
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir), segment_length=4096)
        lr, hr = ds[0]
        assert lr.abs().max().item() <= 1.0 + 1e-5
        assert hr.abs().max().item() <= 1.0 + 1e-5

    def test_no_nan_values(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir), segment_length=4096)
        lr, hr = ds[0]
        assert not torch.isnan(lr).any()
        assert not torch.isnan(hr).any()

    def test_no_inf_values(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir), segment_length=4096)
        lr, hr = ds[0]
        assert not torch.isinf(lr).any()
        assert not torch.isinf(hr).any()

    def test_hr_is_double_lr_length(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir), segment_length=4096)
        lr, hr = ds[0]
        assert hr.shape[-1] == lr.shape[-1] * (OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE)


class TestAudioSRDatasetEdgeCases:
    def test_very_short_file(self, tmp_path):
        """Files shorter than segment_length should be zero-padded."""
        from data.dataset import AudioSRDataset
        # Write only 0.05s — much less than default segment_length
        write_sine_wav(tmp_path / "short.wav", sample_rate=96000, duration=0.05)
        ds = AudioSRDataset(str(tmp_path), segment_length=4096)
        lr, hr = ds[0]
        assert lr.shape[-1] == 4096
        assert hr.shape[-1] == 4096 * (OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE)

    def test_dataloader_batch(self, multi_tier_dir):
        """DataLoader should stack samples into correct batch shapes."""
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(multi_tier_dir), segment_length=2048)
        dl = DataLoader(ds, batch_size=3, shuffle=False)
        lr_batch, hr_batch = next(iter(dl))
        assert lr_batch.shape == (3, 1, 2048)
        assert hr_batch.shape == (3, 1, 2048 * (OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE))


class TestSampleWeights:
    def test_sample_weights_length(self, multi_tier_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(multi_tier_dir))
        weights = ds.get_sample_weights(0.7, 0.2, 0.1)
        assert len(weights) == len(ds)

    def test_hires_weight_assigned(self, hires_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(hires_dir))
        weights = ds.get_sample_weights(0.7, 0.2, 0.1)
        assert weights[ds.hires_indices[0]] == 0.7

    def test_standard_weight_assigned(self, standard_dir):
        from data.dataset import AudioSRDataset
        ds = AudioSRDataset(str(standard_dir))
        weights = ds.get_sample_weights(0.7, 0.2, 0.1)
        assert weights[ds.standard_indices[0]] == 0.1
