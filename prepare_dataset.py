#!/usr/bin/env python3
"""Prepare training data for the 48kHz -> 192kHz GAN.

This script helps you build a training dataset from your existing music collection.
It scans for high-quality audio files and resamples them to 192kHz for training.

Sources of high-quality training data:
- Hi-res audio files (96kHz, 176.4kHz, 192kHz, DSD)
- CD-quality (44.1kHz/16-bit) — still useful, will be upsampled
- Any lossless format (FLAC, WAV, AIFF)

Usage:
    python prepare_dataset.py /path/to/music/collection /path/to/training/data
    python prepare_dataset.py ~/Music ~/audio-enhancer/training_data --target-sr 192000
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
from tqdm import tqdm


AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif", ".dsf", ".dff"}


def get_audio_quality(path: Path) -> dict | None:
    """Get audio file quality info. Returns None if not readable."""
    try:
        info = sf.info(str(path))
        return {
            "path": path,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "duration": info.duration,
            "frames": info.frames,
        }
    except Exception:
        return None


def process_file(src: Path, dst: Path, target_sr: int):
    """Resample a single file to target sample rate."""
    try:
        import numpy as np
        from scipy.signal import resample_poly
        from math import gcd

        data, sr = sf.read(str(src), dtype="float32")

        if sr != target_sr:
            # High-quality polyphase resampling
            g = gcd(sr, target_sr)
            up = target_sr // g
            down = sr // g
            # Process each channel
            if data.ndim == 1:
                data = resample_poly(data, up, down)
            else:
                channels = []
                for ch in range(data.shape[1]):
                    channels.append(resample_poly(data[:, ch], up, down))
                data = np.column_stack(channels)

        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), data, target_sr, subtype="FLOAT")
        return True, str(src)
    except Exception as e:
        return False, f"{src}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("source", help="Source music directory")
    parser.add_argument("output", help="Output training data directory")
    parser.add_argument("--target-sr", type=int, default=192000,
                        help="Target sample rate (default: 192000)")
    parser.add_argument("--min-duration", type=float, default=10.0,
                        help="Minimum duration in seconds (default: 10)")
    parser.add_argument("--min-sr", type=int, default=44100,
                        help="Minimum source sample rate (default: 44100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Scan for audio files
    print(f"Scanning {source} for audio files...")
    all_files = []
    for ext in AUDIO_EXTS:
        all_files.extend(source.rglob(f"*{ext}"))
        all_files.extend(source.rglob(f"*{ext.upper()}"))
    all_files = sorted(set(all_files))
    print(f"Found {len(all_files)} audio files")

    # Filter by quality
    print("Analyzing audio quality...")
    qualified = []
    for f in tqdm(all_files):
        info = get_audio_quality(f)
        if info is None:
            continue
        if info["sample_rate"] < args.min_sr:
            continue
        if info["duration"] < args.min_duration:
            continue
        qualified.append(info)

    # Sort by quality (prefer higher sample rates)
    qualified.sort(key=lambda x: x["sample_rate"], reverse=True)

    if args.max_files:
        qualified = qualified[:args.max_files]

    print(f"\nQualified files: {len(qualified)}")
    sr_counts = {}
    for q in qualified:
        sr = q["sample_rate"]
        sr_counts[sr] = sr_counts.get(sr, 0) + 1
    for sr, count in sorted(sr_counts.items(), reverse=True):
        print(f"  {sr:>6} Hz: {count} files")

    total_duration = sum(q["duration"] for q in qualified)
    print(f"  Total duration: {total_duration / 3600:.1f} hours")

    if not qualified:
        print("No qualifying files found!")
        return

    # Process files
    print(f"\nResampling to {args.target_sr} Hz...")
    tasks = []
    for info in qualified:
        src = info["path"]
        rel = src.relative_to(source)
        dst = output / rel.with_suffix(".wav")
        tasks.append((src, dst, args.target_sr))

    success = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, *t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            ok, msg = future.result()
            if ok:
                success += 1
            else:
                errors.append(msg)

    print(f"\nDone! Processed {success}/{len(tasks)} files -> {output}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
