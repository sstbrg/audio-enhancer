"""Dataset for training the 48kHz -> 96kHz GAN super-resolution model.

Quality-aware training strategy:
- 96kHz/24b sources (EG-IPT, VCTK): real hi-res targets, downsample to 48kHz input
  → model learns genuine harmonics above 24kHz
- 48kHz/24b sources (GTSinger): targets at native SR, input at 48kHz
  → model learns 24-bit dynamic range detail
- 44.1kHz/16b sources (MUSDB, MusicNet, MAESTRO, MoisesDB): resample to 48kHz input,
  upsampled to 96kHz as synthetic target
  → model learns to handle real-world CD-quality input

Batch composition is controlled by ``quality_sampling`` weights in the training config
(phase0.yaml).  Default: 70% hi-res, 20% mid-res, 10% standard.

CD-quality degradation (optional, configured via ``degradation`` in training config):
- Applied on top of the normal pair creation with configurable probability.
- Simulates CD-quality input: downsample to 44.1kHz, requantise to 16-bit with TPDF
  dither, then resample back to 48kHz.  The high-res target is unchanged so the model
  learns to recover the lost information.
"""

import random
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from models.constants import (
    CD_BIT_DEPTH,
    CD_DITHER_AMPLITUDE,
    CD_SAMPLE_RATE,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
)
from utils.audio import load_audio, resample_audio


def _quantize_to_bit_depth(
    waveform: torch.Tensor, bit_depth: int, dither_amplitude: float
) -> torch.Tensor:
    """Quantise a float waveform to ``bit_depth`` bits with TPDF dither.

    The signal is assumed to be in [-1, 1].  TPDF (triangular probability
    density function) dither is added before rounding to reduce quantisation
    distortion.  Output is returned as float32 in [-1, 1].

    Args:
        waveform: Float tensor of shape (channels, samples) in [-1, 1].
        bit_depth: Target integer bit depth (e.g. 16).
        dither_amplitude: Dither amplitude in LSBs (0.5 = TPDF standard).

    Returns:
        Quantised waveform as float32 in [-1, 1].
    """
    levels = 2 ** (bit_depth - 1)   # 32768 for 16-bit
    lsb = 1.0 / levels               # value of one LSB in float space

    # TPDF dither: sum of two independent uniform random variables (each ±0.5 LSB)
    dither = (
        torch.rand_like(waveform) - 0.5
        + torch.rand_like(waveform) - 0.5
    ) * dither_amplitude * lsb

    quantised = torch.round((waveform + dither) * levels) / levels
    return quantised.clamp(-1.0, 1.0)


def _apply_cd_degradation(
    waveform: torch.Tensor,
    current_sr: int,
    input_sr: int,
    cd_sr: int = CD_SAMPLE_RATE,
    bit_depth: int = CD_BIT_DEPTH,
    dither_amplitude: float = CD_DITHER_AMPLITUDE,
) -> torch.Tensor:
    """Simulate CD-quality degradation on a waveform already at ``current_sr``.

    Steps:
    1. Downsample to ``cd_sr`` (44.1 kHz) — loses content above 22.05 kHz.
    2. Quantise to ``bit_depth`` (16-bit) with TPDF dither.
    3. Resample back to ``input_sr`` (48 kHz) — the bandwidth cap stays.

    Args:
        waveform: Float tensor (channels, samples) at ``current_sr``.
        current_sr: Sample rate of the incoming waveform (typically 48000).
        input_sr: Target sample rate for the LR input to the model (48000).
        cd_sr: Intermediate CD sample rate (44100).
        bit_depth: Quantisation bit depth (16).
        dither_amplitude: TPDF dither amplitude in LSBs.

    Returns:
        Degraded waveform at ``input_sr``.
    """
    # Step 1: downsample to CD SR
    degraded = resample_audio(waveform, current_sr, cd_sr)

    # Step 2: 16-bit quantisation with dither
    degraded = _quantize_to_bit_depth(degraded, bit_depth, dither_amplitude)

    # Step 3: resample back to model input SR
    if cd_sr != input_sr:
        degraded = resample_audio(degraded, cd_sr, input_sr)

    return degraded


class AudioSRDataset(Dataset):
    """Quality-aware dataset that creates (low_res, high_res) pairs.

    Scans directories for audio files, detects their native quality,
    and creates appropriate training pairs based on source quality.

    Optional CD-quality degradation can be applied to the LR input with a
    configurable probability, teaching the model to handle sources that have
    been through CD encoding (44.1 kHz / 16-bit quantisation).

    Args:
        root_dir: Root directory to scan for audio files.
        target_sr: High-res target sample rate (default OUTPUT_SAMPLE_RATE).
        input_sr: Low-res input sample rate (default INPUT_SAMPLE_RATE).
        segment_length: Number of LR samples per training segment.
        extensions: Audio file extensions to scan for.
        degradation_prob: Probability [0, 1] of applying CD degradation to the
            LR input.  0.0 disables degradation (default, backward-compatible).
        degradation_cd_sr: CD intermediate sample rate for degradation.
        degradation_bit_depth: Bit depth for quantisation step.
        degradation_dither_amplitude: TPDF dither amplitude in LSBs.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int = OUTPUT_SAMPLE_RATE,
        input_sr: int = INPUT_SAMPLE_RATE,
        segment_length: int = 16384,
        extensions: tuple = (".wav", ".flac", ".aiff", ".aif"),
        degradation_prob: float = 0.0,
        degradation_cd_sr: int = CD_SAMPLE_RATE,
        degradation_bit_depth: int = CD_BIT_DEPTH,
        degradation_dither_amplitude: float = CD_DITHER_AMPLITUDE,
    ):
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        self.input_sr = input_sr
        self.segment_length = segment_length
        self.upsample_factor = target_sr // input_sr
        self.degradation_prob = degradation_prob
        self.degradation_cd_sr = degradation_cd_sr
        self.degradation_bit_depth = degradation_bit_depth
        self.degradation_dither_amplitude = degradation_dither_amplitude

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

        # Classify files by quality tier
        self._classify_files()
        print(f"Found {len(self.files)} audio files in {root_dir}")
        print(f"  Hi-res (>=96kHz): {len(self.hires_indices)} files")
        print(f"  Mid-res (48kHz):  {len(self.midres_indices)} files")
        print(f"  Standard (<=44.1kHz): {len(self.standard_indices)} files")
        if self.degradation_prob > 0.0:
            print(
                f"  CD degradation: prob={self.degradation_prob:.2f}, "
                f"cd_sr={self.degradation_cd_sr}, bit_depth={self.degradation_bit_depth}"
            )

    def get_sample_weights(
        self,
        weight_hires: float,
        weight_midres: float,
        weight_standard: float,
    ) -> list[float]:
        """Return a per-sample weight list for use with WeightedRandomSampler.

        Files in a tier that has no representatives get weight 0 so the
        sampler never tries to draw from an empty set.  When an entire tier
        is absent its configured weight is silently re-distributed by the
        sampler (relative weights still hold among the remaining tiers).

        Args:
            weight_hires: Relative sampling weight for hi-res files (>=96kHz).
            weight_midres: Relative sampling weight for mid-res files (48kHz).
            weight_standard: Relative sampling weight for standard files (<=44.1kHz).

        Returns:
            List of per-file weights with length == len(self.files).
        """
        weights = [0.0] * len(self.files)
        for i in self.hires_indices:
            weights[i] = weight_hires
        for i in self.midres_indices:
            weights[i] = weight_midres
        for i in self.standard_indices:
            weights[i] = weight_standard
        return weights

    def _classify_files(self):
        """Classify files into quality tiers based on sample rate."""
        self.hires_indices = []
        self.midres_indices = []
        self.standard_indices = []

        for i, path in enumerate(self.files):
            try:
                info = sf.info(str(path))
                sr = info.samplerate
                if sr >= 96000:
                    self.hires_indices.append(i)
                elif sr >= 48000:
                    self.midres_indices.append(i)
                else:
                    self.standard_indices.append(i)
            except Exception:
                self.standard_indices.append(i)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]

        # Load at native sample rate
        waveform, sr = load_audio(str(path))

        # Convert to mono for training
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Create LR/HR pair based on source quality
        if sr >= 96000:
            # Hi-res source: genuine target, downsample for input
            hr = self._prepare_segment(waveform, sr, self.target_sr)
            lr = resample_audio(hr, self.target_sr, self.input_sr)
        elif sr >= 48000:
            # Mid-res: resample to target (some synthetic content above native Nyquist)
            hr = self._prepare_segment(waveform, sr, self.target_sr)
            lr = resample_audio(hr, self.target_sr, self.input_sr)
        else:
            # Standard (44.1kHz/16-bit CD): resample to input SR, upsample to target
            # The target is synthetic but teaches the model about CD-quality input
            waveform_48k = resample_audio(waveform, sr, self.input_sr)
            lr = self._prepare_segment_at_sr(waveform_48k, self.input_sr)
            hr = resample_audio(lr, self.input_sr, self.target_sr)

        # Optional CD-quality degradation applied to LR only.
        # HR target is intentionally left clean so the model learns to recover
        # the bandwidth and dynamic range lost through CD encoding.
        if self.degradation_prob > 0.0 and random.random() < self.degradation_prob:
            lr = _apply_cd_degradation(
                lr,
                current_sr=self.input_sr,
                input_sr=self.input_sr,
                cd_sr=self.degradation_cd_sr,
                bit_depth=self.degradation_bit_depth,
                dither_amplitude=self.degradation_dither_amplitude,
            )

        # Normalize
        peak = max(hr.abs().max(), lr.abs().max(), 1e-8)
        hr = hr / peak
        lr = lr / peak

        return lr, hr

    def _prepare_segment(self, waveform: torch.Tensor, src_sr: int, tgt_sr: int) -> torch.Tensor:
        """Resample to target SR and extract a random segment."""
        if src_sr != tgt_sr:
            waveform = resample_audio(waveform, src_sr, tgt_sr)

        hr_segment = self.segment_length * self.upsample_factor

        if waveform.shape[-1] >= hr_segment:
            start = random.randint(0, waveform.shape[-1] - hr_segment)
            return waveform[:, start:start + hr_segment]
        else:
            pad = hr_segment - waveform.shape[-1]
            return torch.nn.functional.pad(waveform, (0, pad))

    def _prepare_segment_at_sr(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract a random segment at the given sample rate (for LR)."""
        if waveform.shape[-1] >= self.segment_length:
            start = random.randint(0, waveform.shape[-1] - self.segment_length)
            return waveform[:, start:start + self.segment_length]
        else:
            pad = self.segment_length - waveform.shape[-1]
            return torch.nn.functional.pad(waveform, (0, pad))
