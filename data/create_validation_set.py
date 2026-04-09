#!/usr/bin/env python3
"""Create a fixed validation set for Phase 1 with per-degradation metadata.

Generates a reproducible set of (degraded, clean) pairs with known degradation
types and severities. This enables per-degradation-type evaluation during
training — we can track whether the model improves on codec artifacts vs EQ
vs compression independently.

Usage:
    python data/create_validation_set.py --data_dir datasets/phase0_combined --output_dir datasets/val_phase1
    python data/create_validation_set.py --data_dir datasets/phase0_combined --num_clips 100 --seed 42

Output:
    <output_dir>/
        metadata.json        — degradation info for each clip
        clean/0000.wav       — clean 96kHz reference
        degraded/0000.wav    — degraded version
        clean/0001.wav
        degraded/0001.wav
        ...
"""

import argparse
import json
import random
from pathlib import Path

import soundfile as sf
import torch

from data.degradations import (
    BadEQDegradation,
    ClippingDegradation,
    CodecDegradation,
    DynamicCompressionDegradation,
    NoiseDegradation,
    SampleRateDegradation,
    StereoDamageDegradation,
)
from models.constants import OUTPUT_SAMPLE_RATE
from utils.audio import load_audio, resample_audio

# One instance of each degradation type at three severity levels
DEGRADATION_MATRIX = [
    ("codec", CodecDegradation, [0.3, 0.6, 0.9]),
    ("eq", BadEQDegradation, [0.3, 0.6, 0.9]),
    ("compression", DynamicCompressionDegradation, [0.3, 0.6, 0.9]),
    ("clipping", ClippingDegradation, [0.3, 0.6, 0.9]),
    ("sample_rate", SampleRateDegradation, [0.3, 0.6, 0.9]),
    ("noise", NoiseDegradation, [0.3, 0.6, 0.9]),
]

SEGMENT_SECONDS = 5.0  # 5-second clips


def find_audio_files(data_dir: Path) -> list[Path]:
    extensions = {".wav", ".flac", ".aiff", ".aif"}
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in extensions)
    return files


def load_segment(path: Path, target_sr: int, segment_samples: int) -> torch.Tensor | None:
    """Load a random segment from an audio file at target_sr."""
    try:
        waveform, sr = load_audio(str(path))
    except Exception:
        return None

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target
    if sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)

    # Extract segment
    if waveform.shape[-1] < segment_samples:
        return None  # Too short

    start = random.randint(0, waveform.shape[-1] - segment_samples)
    segment = waveform[:, start:start + segment_samples]

    # Normalize
    peak = segment.abs().max()
    if peak < 1e-6:
        return None  # Silence
    segment = segment / peak * 0.9

    return segment


def main():
    parser = argparse.ArgumentParser(description="Create fixed Phase 1 validation set")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Source audio directory")
    parser.add_argument("--output_dir", type=str, default="datasets/val_phase1",
                        help="Output directory")
    parser.add_argument("--num_clips", type=int, default=72,
                        help="Total clips (default: 72 = 6 types x 3 severities x 4 clips)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_sr = OUTPUT_SAMPLE_RATE
    segment_samples = int(SEGMENT_SECONDS * target_sr)

    output_dir = Path(args.output_dir)
    clean_dir = output_dir / "clean"
    degraded_dir = output_dir / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    # Find source files
    files = find_audio_files(Path(args.data_dir))
    if not files:
        print(f"No audio files found in {args.data_dir}")
        return
    print(f"Found {len(files)} source files")

    # Compute clips per degradation/severity combo
    n_combos = sum(len(sevs) for _, _, sevs in DEGRADATION_MATRIX)
    clips_per_combo = max(1, args.num_clips // n_combos)

    metadata = []
    clip_idx = 0

    for deg_name, deg_cls, severities in DEGRADATION_MATRIX:
        for severity in severities:
            degradation = deg_cls(severity=severity)
            generated = 0

            # Shuffle files for variety
            shuffled = list(files)
            random.shuffle(shuffled)

            for source_path in shuffled:
                if generated >= clips_per_combo:
                    break

                segment = load_segment(source_path, target_sr, segment_samples)
                if segment is None:
                    continue

                # Apply degradation
                degraded = degradation(segment.clone(), target_sr)

                # Save
                clip_name = f"{clip_idx:04d}.wav"
                sf.write(str(clean_dir / clip_name), segment.squeeze(0).numpy(), target_sr, subtype="PCM_24")
                sf.write(str(degraded_dir / clip_name), degraded.squeeze(0).numpy(), target_sr, subtype="PCM_24")

                metadata.append({
                    "clip_id": clip_idx,
                    "file": clip_name,
                    "source": str(source_path.name),
                    "degradation_type": deg_name,
                    "severity": severity,
                })

                clip_idx += 1
                generated += 1

            print(f"  {deg_name} (severity={severity:.1f}): {generated} clips")

    # Write metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nValidation set: {clip_idx} clips in {output_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
