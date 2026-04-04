#!/usr/bin/env python3
"""Analyze audio file quality — shows technical info + perceptual metrics.

Usage:
    python analyze.py track.wav                     # No-reference analysis
    python analyze.py enhanced.wav --ref original.wav  # Compare with reference
    python analyze.py track.flac --all              # Run all metrics (slow)

Metrics:
  Built-in (always available):
    - Sample rate, bit depth, channels, duration
    - Dynamic range (crest factor), RMS level, peak level
    - Spectral centroid, bandwidth, rolloff
    - Stereo width (if stereo)

  Upscale Potential (always available):
    - Sample rate ceiling (true bandwidth vs nominal SR)
    - Codec artifact detection (MP3/AAC cutoff, pre-echo, SBR)
    - Bit depth headroom (effective vs declared)
    - Spectral rolloff vs Nyquist gap

  No-reference (need optional packages):
    - Audiobox Aesthetics (PQ, CE, PC, CU)
    - PAM (perceptual clarity)
    - MuQ-Eval (music quality MOS)

  Reference-based (--ref required):
    - SI-SNR, SDR
    - CDPAM (perceptual distance)
    - ViSQOL (MOS)
    - Chroma / MFCC / Onset preservation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# ── Colors ────────────────────────────────────────────────────────────────────

C_RED = "\033[0;31m"
C_GREEN = "\033[0;32m"
C_YELLOW = "\033[1;33m"
C_BLUE = "\033[0;34m"
C_CYAN = "\033[0;36m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_NC = "\033[0m"


# ── Built-in analysis (no dependencies beyond numpy/soundfile) ────────────────

def analyze_file_info(path: str) -> dict:
    """Basic audio file info."""
    info = sf.info(path)
    return {
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_s": info.duration,
        "frames": info.frames,
    }


def analyze_levels(path: str) -> dict:
    """Analyze audio levels and dynamics."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    # Mono mix for analysis
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    rms = np.sqrt(np.mean(mono ** 2))
    peak = np.max(np.abs(mono))
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = 20 * np.log10(peak + 1e-10)
    crest_factor_db = peak_db - rms_db

    # Dynamic range estimate (difference between loud and quiet sections)
    frame_size = int(sr * 0.4)  # 400ms frames
    hop = frame_size // 2
    frame_rms = []
    for i in range(0, len(mono) - frame_size, hop):
        frame = mono[i:i + frame_size]
        fr = np.sqrt(np.mean(frame ** 2))
        if fr > 1e-6:  # skip silence
            frame_rms.append(20 * np.log10(fr + 1e-10))

    if frame_rms:
        frame_rms = np.array(frame_rms)
        dr = float(np.percentile(frame_rms, 95) - np.percentile(frame_rms, 10))
    else:
        dr = 0.0

    result = {
        "rms_db": float(rms_db),
        "peak_db": float(peak_db),
        "crest_factor_db": float(crest_factor_db),
        "dynamic_range_db": dr,
    }

    # Stereo width (if stereo)
    if data.shape[1] >= 2:
        mid = (data[:, 0] + data[:, 1]) / 2
        side = (data[:, 0] - data[:, 1]) / 2
        mid_energy = np.mean(mid ** 2)
        side_energy = np.mean(side ** 2)
        width = side_energy / (mid_energy + 1e-10)
        result["stereo_width"] = float(width)
        result["correlation"] = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])

    return result


def analyze_spectrum(path: str) -> dict:
    """Spectral analysis."""
    import librosa

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.95)[0]

    return {
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "nyquist_hz": sr / 2,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    print(f"\n{C_BOLD}{'=' * 60}{C_NC}")
    print(f"{C_BOLD}  {title}{C_NC}")
    print(f"{C_BOLD}{'=' * 60}{C_NC}")


def print_section(title: str):
    print(f"\n  {C_CYAN}{title}{C_NC}")
    print(f"  {'-' * 40}")


def print_metric(name: str, value, unit: str = "", good: bool | None = None):
    if good is True:
        color = C_GREEN
    elif good is False:
        color = C_RED
    else:
        color = C_NC

    if isinstance(value, float):
        val_str = f"{value:.2f}"
    else:
        val_str = str(value)

    print(f"  {name:<28s} {color}{val_str:>10s}{C_NC} {C_DIM}{unit}{C_NC}")


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze audio file quality")
    parser.add_argument("file", help="Audio file to analyze")
    parser.add_argument("--ref", help="Reference file for comparison metrics")
    parser.add_argument("--all", action="store_true",
                        help="Run all metrics including slow ones (PAM, MuQ-Eval)")
    parser.add_argument("--no-spectrum", action="store_true",
                        help="Skip spectral analysis (faster)")
    args = parser.parse_args()

    path = args.file
    if not Path(path).exists():
        print(f"{C_RED}File not found: {path}{C_NC}")
        sys.exit(1)

    print_header(f"Audio Analysis: {Path(path).name}")

    # File info
    print_section("File Info")
    info = analyze_file_info(path)
    print_metric("Format", f"{info['format']} / {info['subtype']}")
    print_metric("Sample Rate", info["sample_rate"], "Hz")
    print_metric("Channels", info["channels"])
    print_metric("Duration", format_duration(info["duration_s"]))
    print_metric("Samples", f"{info['frames']:,}")

    # Levels & dynamics
    print_section("Levels & Dynamics")
    levels = analyze_levels(path)
    print_metric("Peak Level", levels["peak_db"], "dBFS")
    print_metric("RMS Level", levels["rms_db"], "dBFS")
    print_metric("Crest Factor", levels["crest_factor_db"], "dB",
                 good=levels["crest_factor_db"] > 10)
    print_metric("Dynamic Range", levels["dynamic_range_db"], "dB",
                 good=levels["dynamic_range_db"] > 8)

    if "stereo_width" in levels:
        print_section("Stereo")
        print_metric("Stereo Width", levels["stereo_width"])
        print_metric("L/R Correlation", levels["correlation"],
                     good=0.3 < levels["correlation"] < 0.95)

    # Spectral
    if not args.no_spectrum:
        print_section("Spectrum")
        spec = analyze_spectrum(path)
        print_metric("Spectral Centroid", spec["spectral_centroid_hz"], "Hz")
        print_metric("Spectral Bandwidth", spec["spectral_bandwidth_hz"], "Hz")
        print_metric("Spectral Rolloff (95%)", spec["spectral_rolloff_hz"], "Hz")
        print_metric("Nyquist", spec["nyquist_hz"], "Hz")

    # Upscale potential
    print_section("Upscale Potential")
    from metrics.upscale_potential import analyze_upscale_potential
    up = analyze_upscale_potential(path)

    score = up["composite_score"]
    if score >= 70:
        score_good = False   # needs work
    elif score >= 40:
        score_good = None    # moderate
    else:
        score_good = True    # already hi-res

    print_metric("Composite Score", f"{score}/100", "", good=score_good)
    print_metric("Summary", up["summary"])

    src = up["sample_rate_ceiling"]
    print_metric("  Effective Bandwidth", src["effective_sr"] // 2, "Hz")
    print_metric("  Bandwidth Utilization",
                 f"{src['bandwidth_utilization'] * 100:.1f}", "%")

    cod = up["codec_artifacts"]
    if cod["codec_artifacts_detected"]:
        print_metric("  Codec Artifacts",
                     f"{cod['likely_codec'].upper()} ({cod['confidence'] * 100:.0f}%)",
                     good=False)
        if cod["cutoff_hz"]:
            print_metric("    Hard Cutoff", int(cod["cutoff_hz"]), "Hz")
        if cod["pre_echo_events"] > 0:
            print_metric("    Pre-echo Events", cod["pre_echo_events"])
        if cod["sbr_detected"]:
            print_metric("    SBR Detected", "yes")
    else:
        print_metric("  Codec Artifacts", "none detected", good=True)

    bd = up["bit_depth_headroom"]
    if bd["headroom_db"] > 0:
        print_metric("  Effective Bit Depth",
                     f"{bd['effective_bit_depth']}-bit in {bd['declared_bit_depth']}-bit",
                     f"(+{bd['headroom_db']:.0f}dB headroom)")
    else:
        print_metric("  Bit Depth", f"{bd['declared_bit_depth']}-bit",
                     "(fully used)", good=True)

    gp = up["spectral_gap"]
    if gp["gap_ratio"] > 0.15:
        print_metric("  Spectral Gap", f"{int(gp['gap_hz'])}",
                     f"Hz below Nyquist ({gp['gap_ratio'] * 100:.0f}% unused)",
                     good=False)
    else:
        print_metric("  Spectral Gap", "minimal", good=True)

    # No-reference perceptual metrics
    print_section("Perceptual Quality (no-reference)")
    from metrics.evaluate import AudioMetrics
    metrics = AudioMetrics()

    # Audiobox (fast, most useful)
    ab = metrics.audiobox_aesthetics(path)
    if isinstance(ab.score, dict):
        for k, v in ab.score.items():
            labels = {"PQ": "Production Quality", "CE": "Enjoyment",
                      "PC": "Complexity", "CU": "Usefulness"}
            print_metric(f"  Audiobox {labels.get(k, k)}", v, "/10",
                         good=v > 6 if k == "PQ" else None)
    else:
        print_metric("Audiobox Aesthetics", "not installed", "",
                     good=None)

    if args.all:
        pam = metrics.pam_score(path)
        print_metric("PAM (clarity)", pam.score, "/1.0",
                     good=pam.score > 0.5 if not np.isnan(pam.score) else None)

        muq = metrics.muq_eval(path)
        print_metric("MuQ-Eval (music MOS)", muq.score, "/5.0",
                     good=muq.score > 3.5 if not np.isnan(muq.score) else None)

    # Reference-based metrics
    if args.ref:
        if not Path(args.ref).exists():
            print(f"\n{C_RED}Reference file not found: {args.ref}{C_NC}")
        else:
            print_section("Reference Comparison")
            ref = args.ref

            sisnr = metrics.si_snr(ref, path)
            print_metric("SI-SNR", sisnr.score, "dB", good=sisnr.score > 15)

            sdr = metrics.sdr(ref, path)
            print_metric("SDR", sdr.score, "dB", good=sdr.score > 15)

            cdpam = metrics.cdpam_score(ref, path)
            if not np.isnan(cdpam.score):
                print_metric("CDPAM (distance)", cdpam.score, "",
                             good=cdpam.score < 0.3)

            if args.all:
                visqol = metrics.visqol(ref, path)
                if not np.isnan(visqol.score):
                    print_metric("ViSQOL", visqol.score, "MOS",
                                 good=visqol.score > 3.5)

            print_section("Content Preservation")
            chroma = metrics.chroma_similarity(ref, path)
            print_metric("Chroma (melody)", chroma.score, "",
                         good=chroma.score > 0.95)
            mfcc = metrics.mfcc_similarity(ref, path)
            print_metric("MFCC (timbre)", mfcc.score, "",
                         good=mfcc.score > 0.90)
            onset = metrics.onset_f1(ref, path)
            print_metric("Onset F1 (rhythm)", onset.score, "",
                         good=onset.score > 0.90)

    print(f"\n{C_BOLD}{'=' * 60}{C_NC}")


if __name__ == "__main__":
    main()
