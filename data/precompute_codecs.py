#!/usr/bin/env python3
"""Pre-compute codec-degraded variants of training audio for Phase 1.

Codec encoding via ffmpeg is too slow for on-the-fly augmentation during
training. This script pre-computes degraded WAV files for the most common
codec/bitrate combinations and stores them alongside the originals.

Usage:
    python data/precompute_codecs.py --data_dir datasets/phase0_combined
    python data/precompute_codecs.py --data_dir datasets/phase0_combined --output_dir datasets/codec_variants
    python data/precompute_codecs.py --data_dir datasets/phase0_combined --codecs mp3 aac --workers 8

Output structure:
    <output_dir>/<codec>_<bitrate>/<relative_path>.wav

Each output file is the result of encoding the source to the lossy codec at
the specified bitrate, then decoding back to WAV. The file retains the same
sample rate and channel layout as the source.
"""

import argparse
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

from models.constants import PRECOMPUTE_BITRATES

# ffmpeg encoder names
ENCODER_MAP = {
    "mp3": "libmp3lame",
    "aac": "aac",
    "ogg": "libvorbis",
}


def encode_decode(
    source_path: Path,
    output_path: Path,
    codec: str,
    bitrate: int,
) -> bool:
    """Encode source to lossy codec and decode back to WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = ENCODER_MAP.get(codec)
    if not encoder:
        return False

    ext = {"mp3": "mp3", "aac": "m4a", "ogg": "ogg"}[codec]

    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp_lossy = tmp.name

        # Encode
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(source_path),
             "-c:a", encoder, "-b:a", f"{bitrate}k", tmp_lossy],
            capture_output=True, timeout=60,
            check=True,
        )

        # Decode back to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_lossy, str(output_path)],
            capture_output=True, timeout=60,
            check=True,
        )

        Path(tmp_lossy).unlink(missing_ok=True)
        return True

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        Path(tmp_lossy).unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return False


def process_file(
    source_path: Path,
    source_root: Path,
    output_dir: Path,
    codec: str,
    bitrate: int,
) -> tuple[str, bool]:
    """Process a single file — returns (relative_path, success)."""
    relative = source_path.relative_to(source_root)
    variant_dir = f"{codec}_{bitrate}"
    output_path = output_dir / variant_dir / relative.with_suffix(".wav")

    if output_path.exists():
        return str(relative), True  # Already computed

    success = encode_decode(source_path, output_path, codec, bitrate)
    return str(relative), success


def main():
    parser = argparse.ArgumentParser(description="Pre-compute codec variants for Phase 1")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Source audio directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: datasets/codec_variants)")
    parser.add_argument("--codecs", nargs="+", default=None,
                        help=f"Codecs to compute (default: {list(PRECOMPUTE_BITRATES.keys())})")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    args = parser.parse_args()

    source_root = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("datasets/codec_variants")

    codecs = args.codecs or list(PRECOMPUTE_BITRATES.keys())

    # Find audio files
    extensions = {".wav", ".flac", ".aiff", ".aif"}
    files = sorted(
        p for p in source_root.rglob("*") if p.suffix.lower() in extensions
    )
    print(f"Found {len(files)} audio files in {source_root}")

    # Build work items
    work = []
    for codec in codecs:
        if codec not in PRECOMPUTE_BITRATES:
            print(f"Warning: no precompute bitrates defined for {codec}, skipping")
            continue
        for bitrate in PRECOMPUTE_BITRATES[codec]:
            for f in files:
                work.append((f, source_root, output_dir, codec, bitrate))

    total = len(work)
    print(f"Computing {total} codec variants ({len(codecs)} codecs, {len(files)} files)")

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, *w): w for w in work}
        for future in as_completed(futures):
            rel, success = future.result()
            done += 1
            if not success:
                failed += 1
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] ({failed} failed)")

    print(f"\nDone: {done - failed}/{total} succeeded, {failed} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
