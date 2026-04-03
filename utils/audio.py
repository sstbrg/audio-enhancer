"""Audio I/O and processing utilities."""

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path


def load_audio(path: str, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """Load audio file, return (waveform, sample_rate).

    Waveform shape: (channels, samples) as float32 in [-1, 1].
    """
    path = str(path)
    info = sf.info(path)
    original_sr = info.samplerate

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # data shape: (samples, channels) -> transpose to (channels, samples)
    waveform = torch.from_numpy(data.T)

    if target_sr is not None and sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int,
               bit_depth: int = 32):
    """Save waveform to file as 32-bit float WAV or FLAC.

    waveform shape: (channels, samples)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = waveform.detach().cpu().numpy().T  # (samples, channels)
    data = np.clip(data, -1.0, 1.0)

    subtype = "FLOAT" if bit_depth == 32 else "PCM_24"
    fmt = path.suffix.lstrip(".").upper()
    if fmt == "FLAC":
        subtype = "PCM_24"  # FLAC doesn't support float

    sf.write(str(path), data, sample_rate, subtype=subtype)


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """High-quality resampling using Kaiser window."""
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    return resampler(waveform)


def get_audio_info(path: str) -> dict:
    """Get audio file metadata."""
    info = sf.info(path)
    return {
        "path": path,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }
