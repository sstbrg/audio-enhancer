"""Dataset for training the 48kHz -> 192kHz GAN upsampler.

Training strategy:
- Collect high-quality 192kHz audio (or resample 96kHz+ sources)
- Downsample to 48kHz as input (simulating the low-res version)
- Train GAN to reconstruct the 192kHz version from the 48kHz input
"""

import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from utils.audio import load_audio, resample_audio


class AudioSRDataset(Dataset):
    """Dataset that creates (low_res, high_res) pairs for training.

    Expects a directory of high-quality audio files (ideally 192kHz or 96kHz+).
    Files at lower sample rates will be upsampled to target_sr first.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int = 192000,
        input_sr: int = 48000,
        segment_length: int = 32768,
        extensions: tuple = (".wav", ".flac", ".aiff", ".aif"),
    ):
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        self.input_sr = input_sr
        self.segment_length = segment_length  # in samples at input_sr
        self.upsample_factor = target_sr // input_sr

        # Find all audio files (follow symlinks)
        import glob
        self.files = []
        for ext in extensions:
            self.files.extend(
                Path(p) for p in glob.glob(str(self.root_dir / "**" / f"*{ext}"), recursive=True)
            )
        self.files = sorted(self.files)

        if not self.files:
            raise RuntimeError(
                f"No audio files found in {root_dir} with extensions {extensions}"
            )

        print(f"Found {len(self.files)} audio files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]

        # Load at native sample rate
        waveform, sr = load_audio(str(path))

        # Convert to mono for training
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target (192kHz) if needed
        if sr != self.target_sr:
            waveform = resample_audio(waveform, sr, self.target_sr)

        # Calculate segment length in high-res samples
        hr_segment = self.segment_length * self.upsample_factor

        # Random crop (or pad if too short)
        if waveform.shape[-1] >= hr_segment:
            start = random.randint(0, waveform.shape[-1] - hr_segment)
            hr = waveform[:, start : start + hr_segment]
        else:
            # Pad short files
            pad = hr_segment - waveform.shape[-1]
            hr = torch.nn.functional.pad(waveform, (0, pad))

        # Create low-res version by downsampling
        lr = resample_audio(hr, self.target_sr, self.input_sr)

        # Normalize
        peak = max(hr.abs().max(), lr.abs().max(), 1e-8)
        hr = hr / peak
        lr = lr / peak

        return lr, hr
