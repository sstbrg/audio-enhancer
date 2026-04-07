"""Phase 1 dataset: creates (degraded, clean) pairs for degradation restoration training.

Loads clean high-quality audio and applies a randomised degradation chain.
A configurable fraction of each batch uses clean Phase 0 SR pairs (48k→96k)
to prevent catastrophic forgetting of super-resolution ability.
"""

import random

import torch
from torch.utils.data import Dataset

from data.dataset import AudioSRDataset
from data.degradation_chain import DegradationChain
from models.constants import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, PHASE0_MIX_RATIO
from utils.audio import resample_audio


class DegradedAudioDataset(Dataset):
    """Phase 1 dataset that wraps AudioSRDataset and applies degradation chains.

    For each sample:
    - With probability (1 - phase0_mix_ratio): load clean audio at target_sr,
      apply degradation chain → return (degraded, clean) pair.
    - With probability phase0_mix_ratio: return a normal Phase 0 SR pair
      (48kHz input, 96kHz target) to prevent forgetting.

    Args:
        root_dir: Root directory containing audio files.
        target_sr: Output sample rate (96000).
        input_sr: Input sample rate for Phase 0 pairs (48000).
        segment_length: Number of samples per segment (at input_sr).
        degradation_config: Degradation section from phase1.yaml.
        phase0_mix_ratio: Fraction of samples that are clean SR pairs.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int = OUTPUT_SAMPLE_RATE,
        input_sr: int = INPUT_SAMPLE_RATE,
        segment_length: int = 32768,
        degradation_config: dict | None = None,
        phase0_mix_ratio: float = PHASE0_MIX_RATIO,
    ):
        # Reuse AudioSRDataset for file scanning and quality classification
        self.sr_dataset = AudioSRDataset(
            root_dir=root_dir,
            target_sr=target_sr,
            input_sr=input_sr,
            segment_length=segment_length,
        )
        self.target_sr = target_sr
        self.input_sr = input_sr
        self.phase0_mix_ratio = phase0_mix_ratio

        # Build degradation chain
        deg_cfg = degradation_config or {}
        self.degradation_chain = DegradationChain(deg_cfg)

    def set_epoch(self, epoch: int):
        """Update degradation severity based on curriculum schedule."""
        self.degradation_chain.update_epoch(epoch)

    def __len__(self):
        return len(self.sr_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Phase 0 mix: return clean SR pair (48k→96k)
        if random.random() < self.phase0_mix_ratio:
            return self.sr_dataset[idx]

        # Phase 1: load clean audio and degrade it
        lr, hr = self.sr_dataset[idx]

        # Apply degradation chain to the HR (clean) signal
        # The model learns to restore: degraded → clean
        degraded = self.degradation_chain(hr.clone(), self.target_sr)

        # Normalize together to keep relative levels
        peak = max(hr.abs().max(), degraded.abs().max(), 1e-8)
        hr = hr / peak
        degraded = degraded / peak

        return degraded, hr
