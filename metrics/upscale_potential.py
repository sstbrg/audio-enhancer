"""Upscale potential assessment — how much a file would benefit from super-resolution.

Analyzes four dimensions of "room for improvement":

  1. Sample Rate Ceiling  — true bandwidth vs nominal sample rate
  2. Codec Artifact Detection — MP3/AAC signatures (pre-echo, hard cutoffs, SBR)
  3. Bit Depth Headroom — effective bit depth vs declared container depth
  4. Spectral Rolloff vs Nyquist Gap — headroom above the energy rolloff point

Returns a composite score (0–100) where higher = more benefit from upscaling.

Usage:
    result = analyze_upscale_potential("track.wav")
    print(result["composite_score"])  # e.g. 72

    # CLI:
    python -m metrics.upscale_potential track.wav
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

# FFT parameters for spectral analysis
FFT_N = 4096

# Spread sampling: number of evenly-spaced FFT frames to draw from the whole file.
# This gives a representative view of the file's bandwidth regardless of which
# section happens to be quiet (e.g. a soft intro shouldn't skew results).
SPREAD_SAMPLE_FRAMES = 20

# Bit-depth analysis: cap at this many seconds to keep it fast
BIT_DEPTH_MAX_SECONDS = 10.0

# Sample-rate ceiling detection
# Noise floor: a frequency bin is "active" when its mean power (across all sampled
# frames) is above this fraction of the spectrum peak.
NOISE_FLOOR_RATIO = 1e-4          # −40 dB below spectrum peak
# Smoothing window (bins) applied to the mean power spectrum before ceiling detection
ENERGY_SMOOTH_BINS = 20
# Tolerance: if the detected ceiling is within this fraction of Nyquist, treat as full
CEILING_TOLERANCE = 0.05          # 5 % of Nyquist

# Codec artifact detection
# Hard-cutoff: flag when energy drops by at least this many dB across a narrow band
CODEC_CUTOFF_DROP_DB = 20.0
# Candidate codec cutoff frequencies (Hz) for the hard-cutoff scan
CODEC_CUTOFF_CANDIDATES_HZ = [11000, 15000, 16000, 18000, 19000, 20000, 22000]
# Width of the band (Hz) used to measure energy on each side of a candidate cutoff
CODEC_CUTOFF_BAND_HZ = 500
# Pass-band must have at least this much energy (fraction of spectrum peak) for the
# cutoff test to be meaningful
CODEC_PASSBAND_MIN_RATIO = 1e-6
# Pre-echo: analysis frame length in ms
PRE_ECHO_WINDOW_MS = 20.0
# Pre-echo: frame is suspicious when its energy exceeds this fraction of the
# following transient frame's energy
PRE_ECHO_RATIO_THRESHOLD = 0.15
# How many pre-echo events must be found before flagging the file
PRE_ECHO_MIN_EVENTS = 2
# SBR (Spectral Band Replication): flag when correlation between the band below and
# the band above the cutoff exceeds this value
SBR_CORRELATION_THRESHOLD = 0.80

# Bit-depth headroom
# Candidate container depths to probe (ascending order)
CANDIDATE_BIT_DEPTHS = [8, 16, 20, 24, 32]
# A depth is considered "insufficient" (signal spills into next depth) when the
# quantisation residual power / total power exceeds this threshold
BIT_DEPTH_RESIDUAL_THRESHOLD = 1e-9

# Spectral rolloff vs Nyquist gap
# Fraction of cumulative spectral energy used to locate the rolloff frequency
ROLLOFF_PERCENT = 0.99   # 99 % catches the true high-frequency content

# Composite score weights (must sum to 1.0)
WEIGHT_SR_CEILING = 0.35
WEIGHT_CODEC = 0.25
WEIGHT_BIT_DEPTH = 0.15
WEIGHT_GAP = 0.25


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_info(path: str) -> sf.SoundFileInfo:
    return sf.info(path)


def _spread_power_spectrum(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute a representative mean power spectrum by sampling evenly across the file.

    Returns (freqs_hz, mean_power, sample_rate).
    Sampling evenly avoids bias from quiet intros / outros.
    """
    info = _load_info(path)
    sr = info.samplerate
    total_frames = info.frames
    freqs = np.fft.rfftfreq(FFT_N, d=1.0 / sr)

    # Positions across the whole file, each at least FFT_N apart
    max_start = max(0, total_frames - FFT_N)
    positions = np.linspace(0, max_start, SPREAD_SAMPLE_FRAMES, dtype=int)
    window = np.hanning(FFT_N)

    accumulated = np.zeros(len(freqs))
    count = 0

    with sf.SoundFile(path) as f:
        for pos in positions:
            f.seek(int(pos))
            raw = f.read(FFT_N, dtype="float32", always_2d=True)
            if len(raw) < FFT_N:
                pad = np.zeros((FFT_N - len(raw), raw.shape[1]), dtype="float32")
                raw = np.vstack([raw, pad])
            mono_frame = raw.mean(axis=1) * window
            accumulated += np.abs(np.fft.rfft(mono_frame)) ** 2
            count += 1

    mean_power = accumulated / max(count, 1)
    return freqs, mean_power, sr


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply a simple moving-average smoothing."""
    if window < 2 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _load_mono_capped(path: str, max_seconds: float) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono, capped to max_seconds."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    cap = int(sr * max_seconds)
    return mono[:cap], sr


# ---------------------------------------------------------------------------
# 1. Sample Rate Ceiling
# ---------------------------------------------------------------------------

def _sample_rate_ceiling(freqs: np.ndarray, power: np.ndarray, sr: int) -> dict:
    """Detect the true bandwidth vs the nominal sample rate.

    Returns:
        effective_sr          — estimated true bandwidth * 2 (Hz), even number
        nominal_sr            — sample rate reported by the file
        bandwidth_utilization — fraction of Nyquist that is genuinely used (0–1)
    """
    nyquist = sr / 2.0
    smooth_power = _smooth(power, ENERGY_SMOOTH_BINS)
    peak_power = smooth_power.max()

    if peak_power == 0:
        return {
            "effective_sr": sr,
            "nominal_sr": sr,
            "bandwidth_utilization": 1.0,
        }

    threshold = peak_power * NOISE_FLOOR_RATIO
    active_indices = np.where(smooth_power >= threshold)[0]

    if len(active_indices) == 0:
        ceiling_hz = 0.0
    else:
        ceiling_hz = float(freqs[active_indices[-1]])

    # If within tolerance of Nyquist, treat as full bandwidth
    if ceiling_hz >= nyquist * (1.0 - CEILING_TOLERANCE):
        ceiling_hz = nyquist

    effective_sr = int(ceiling_hz * 2)
    bandwidth_utilization = min(ceiling_hz / nyquist, 1.0)

    return {
        "effective_sr": effective_sr,
        "nominal_sr": sr,
        "bandwidth_utilization": round(bandwidth_utilization, 4),
    }


# ---------------------------------------------------------------------------
# 2. Codec Artifact Detection
# ---------------------------------------------------------------------------

def _codec_artifacts(
    freqs: np.ndarray,
    power: np.ndarray,
    sr: int,
    path: str,
) -> dict:
    """Detect signs of lossy compression (MP3/AAC).

    Returns:
        codec_artifacts_detected  — bool
        likely_codec              — "mp3" | "aac" | "none" | "unknown"
        confidence                — 0.0–1.0
        cutoff_hz                 — detected hard cutoff (Hz), or None
        pre_echo_events           — number of suspicious pre-echo events found
        sbr_detected              — bool (spectral band replication signature)
    """
    nyquist = sr / 2.0
    evidence = []
    cutoff_hz = None
    sbr_detected = False

    # ── Hard frequency cutoff ────────────────────────────────────────────────
    bin_width_hz = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    band_bins = max(1, int(CODEC_CUTOFF_BAND_HZ / bin_width_hz))
    peak_power = power.max()

    best_drop_db = 0.0
    for candidate_hz in CODEC_CUTOFF_CANDIDATES_HZ:
        if candidate_hz >= nyquist:
            continue
        idx = int(candidate_hz / bin_width_hz)
        if idx < band_bins or idx + band_bins >= len(power):
            continue

        below_power = power[idx - band_bins : idx].mean()
        above_power = power[idx : idx + band_bins].mean()

        if below_power < CODEC_PASSBAND_MIN_RATIO * peak_power:
            continue  # Pass band is essentially silent — not a meaningful cutoff

        if above_power < 1e-30:
            above_power = 1e-30
        drop_db = 10.0 * np.log10(below_power / above_power)

        if drop_db > CODEC_CUTOFF_DROP_DB and drop_db > best_drop_db:
            best_drop_db = drop_db
            cutoff_hz = float(candidate_hz)

    if cutoff_hz is not None:
        # Confidence proportional to sharpness of the drop
        conf = min(0.9, (best_drop_db - CODEC_CUTOFF_DROP_DB) / 30.0 + 0.4)
        evidence.append(("cutoff", conf))

    # ── Spectral Band Replication (SBR) ─────────────────────────────────────
    # AAC-HE patches the band above the core cutoff by replicating the content
    # below it, yielding high cross-band correlation.
    if cutoff_hz is not None:
        cut_idx = int(cutoff_hz / bin_width_hz)
        band_len = min(cut_idx, len(power) - cut_idx)
        if band_len > 16:
            below_band = power[cut_idx - band_len : cut_idx]
            above_band = power[cut_idx : cut_idx + band_len]
            below_norm = below_band / (below_band.max() + 1e-30)
            above_norm = above_band / (above_band.max() + 1e-30)
            corr = float(np.corrcoef(below_norm, above_norm)[0, 1])
            if corr > SBR_CORRELATION_THRESHOLD:
                sbr_detected = True
                evidence.append(("sbr", 0.5))

    # ── Pre-echo detection ───────────────────────────────────────────────────
    # Pre-echo: energy leaks into silence just before a transient onset.
    # We load the first 30 seconds for this (pre-echo analysis needs temporal data).
    pre_echo_events = 0
    try:
        mono_pre, sr_pre = _load_mono_capped(path, max_seconds=30.0)
        win_samples = int(PRE_ECHO_WINDOW_MS / 1000.0 * sr_pre)
        hop_samples = max(1, win_samples // 2)

        ste = np.array([
            float(np.mean(mono_pre[i : i + win_samples] ** 2))
            for i in range(0, len(mono_pre) - win_samples, hop_samples)
        ])

        if len(ste) > 4:
            ste_db = 10.0 * np.log10(ste + 1e-30)
            diffs = np.diff(ste_db)
            transient_indices = np.where(diffs > 10.0)[0]
            for t_idx in transient_indices:
                if t_idx == 0:
                    continue
                if ste[t_idx] > PRE_ECHO_RATIO_THRESHOLD * ste[t_idx + 1]:
                    pre_echo_events += 1
    except Exception:
        pass  # Pre-echo analysis is best-effort

    if pre_echo_events >= PRE_ECHO_MIN_EVENTS:
        evidence.append(("pre_echo", min(0.7, pre_echo_events * 0.1)))

    # ── Aggregate evidence ───────────────────────────────────────────────────
    if not evidence:
        return {
            "codec_artifacts_detected": False,
            "likely_codec": "none",
            "confidence": 0.0,
            "cutoff_hz": None,
            "pre_echo_events": pre_echo_events,
            "sbr_detected": sbr_detected,
        }

    # Independent evidence fusion: P(any) = 1 − ∏(1 − Pᵢ)
    combined = float(1.0 - np.prod([1.0 - c for _, c in evidence]))
    combined = round(min(combined, 0.99), 3)

    if sbr_detected:
        likely_codec = "aac"
    elif cutoff_hz is not None:
        likely_codec = "mp3"
    elif pre_echo_events >= PRE_ECHO_MIN_EVENTS:
        likely_codec = "mp3"
    else:
        likely_codec = "unknown"

    return {
        "codec_artifacts_detected": True,
        "likely_codec": likely_codec,
        "confidence": combined,
        "cutoff_hz": cutoff_hz,
        "pre_echo_events": pre_echo_events,
        "sbr_detected": sbr_detected,
    }


# ---------------------------------------------------------------------------
# 3. Bit Depth Headroom
# ---------------------------------------------------------------------------

def _bit_depth_headroom(path: str) -> dict:
    """Detect effective bit depth vs declared container depth.

    Probes LSB patterns by quantising the signal to each candidate depth and
    measuring how much residual energy exists — if the residual is near zero,
    the file doesn't actually use those lower bits.

    Returns:
        effective_bit_depth  — lowest depth that captures the full signal
        declared_bit_depth   — depth declared in the container subtype
        headroom_db          — extra dynamic range available above effective depth
    """
    info = _load_info(path)
    subtype = info.subtype  # e.g. "PCM_16", "PCM_24", "FLOAT"

    declared_bit_depth = 16  # sensible default
    for depth in CANDIDATE_BIT_DEPTHS:
        if str(depth) in subtype:
            declared_bit_depth = depth
            break
    if "FLOAT" in subtype or "DOUBLE" in subtype:
        declared_bit_depth = 32

    mono, _sr = _load_mono_capped(path, max_seconds=BIT_DEPTH_MAX_SECONDS)

    if mono.size == 0:
        return {
            "effective_bit_depth": declared_bit_depth,
            "declared_bit_depth": declared_bit_depth,
            "headroom_db": 0.0,
        }

    total_power = float(np.mean(mono ** 2))
    if total_power == 0:
        return {
            "effective_bit_depth": 1,
            "declared_bit_depth": declared_bit_depth,
            "headroom_db": float(6.02 * (declared_bit_depth - 1)),
        }

    effective_depth = declared_bit_depth

    for depth in CANDIDATE_BIT_DEPTHS:
        if depth > declared_bit_depth:
            break
        step = 2.0 ** (-(depth - 1))   # quantisation step for full-scale ±1 signal
        quantised = np.round(mono / step) * step
        residual_power = float(np.mean((mono - quantised) ** 2))

        if residual_power / (total_power + 1e-30) < BIT_DEPTH_RESIDUAL_THRESHOLD:
            # Signal fits within this depth — lower bits are empty
            effective_depth = depth
            break
    else:
        effective_depth = declared_bit_depth

    # Each bit ≈ 6.02 dB of dynamic range
    headroom_db = float(6.02 * (declared_bit_depth - effective_depth))

    return {
        "effective_bit_depth": effective_depth,
        "declared_bit_depth": declared_bit_depth,
        "headroom_db": round(headroom_db, 1),
    }


# ---------------------------------------------------------------------------
# 4. Spectral Rolloff vs Nyquist Gap
# ---------------------------------------------------------------------------

def _spectral_gap(freqs: np.ndarray, power: np.ndarray, sr: int) -> dict:
    """Spectral rolloff vs Nyquist gap.

    Uses ROLLOFF_PERCENT (99 %) of cumulative spectral energy so that even
    low-energy but present high-frequency content registers correctly.

    Returns:
        rolloff_hz  — frequency below which ROLLOFF_PERCENT of energy lies
        nyquist_hz  — sr / 2
        gap_hz      — nyquist_hz - rolloff_hz
        gap_ratio   — gap_hz / nyquist_hz  (0 = full bandwidth, 1 = all energy at DC)
    """
    nyquist = sr / 2.0
    total_energy = power.sum()

    if total_energy == 0:
        return {
            "rolloff_hz": 0.0,
            "nyquist_hz": nyquist,
            "gap_hz": nyquist,
            "gap_ratio": 1.0,
        }

    cumulative = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumulative, ROLLOFF_PERCENT * total_energy)
    rolloff_idx = min(int(rolloff_idx), len(freqs) - 1)
    rolloff_hz = float(freqs[rolloff_idx])

    gap_hz = nyquist - rolloff_hz
    gap_ratio = gap_hz / nyquist

    return {
        "rolloff_hz": round(rolloff_hz, 1),
        "nyquist_hz": nyquist,
        "gap_hz": round(max(gap_hz, 0.0), 1),
        "gap_ratio": round(float(max(gap_ratio, 0.0)), 4),
    }


# ---------------------------------------------------------------------------
# 5. Composite Score
# ---------------------------------------------------------------------------

def _composite_score(sr_ceil: dict, codec: dict, bit_depth: dict, gap: dict) -> int:
    """Combine sub-scores into a single 0–100 upscale potential score.

    Scoring intent:
      - SR ceiling: unused bandwidth = headroom for super-resolution
      - Codec:      artefacts = damage the model can repair
      - Bit depth:  wasted container depth = unused precision
      - Gap:        rolloff far below Nyquist = lots of missing harmonics
    """
    sr_score = 1.0 - sr_ceil["bandwidth_utilization"]
    codec_score = codec["confidence"] if codec["codec_artifacts_detected"] else 0.0

    declared = bit_depth["declared_bit_depth"]
    max_headroom = float(6.02 * (declared - 1)) if declared > 1 else 1.0
    bit_score = min(bit_depth["headroom_db"] / max_headroom, 1.0)

    gap_score = gap["gap_ratio"]

    raw = (
        WEIGHT_SR_CEILING * sr_score
        + WEIGHT_CODEC * codec_score
        + WEIGHT_BIT_DEPTH * bit_score
        + WEIGHT_GAP * gap_score
    )

    return int(round(min(max(raw, 0.0), 1.0) * 100))


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    score: int,
    sr_ceil: dict,
    codec: dict,
    bit_depth: dict,
    gap: dict,
) -> str:
    parts = []

    bw = sr_ceil["bandwidth_utilization"]
    eff_sr = sr_ceil["effective_sr"]
    nom_sr = sr_ceil["nominal_sr"]
    if bw < 0.90:
        parts.append(
            f"true bandwidth ~{eff_sr // 1000}kHz in {nom_sr // 1000}kHz file"
        )
    else:
        parts.append(f"full bandwidth used ({nom_sr // 1000}kHz)")

    if codec["codec_artifacts_detected"]:
        codec_name = codec["likely_codec"].upper()
        cutoff = codec["cutoff_hz"]
        if cutoff:
            parts.append(
                f"{codec_name} artefacts detected (cutoff ~{int(cutoff)}Hz, "
                f"{int(codec['confidence'] * 100)}% confidence)"
            )
        else:
            parts.append(
                f"{codec_name} artefacts detected ({int(codec['confidence'] * 100)}% confidence)"
            )

    if bit_depth["headroom_db"] > 0:
        parts.append(
            f"effective depth {bit_depth['effective_bit_depth']}-bit "
            f"in {bit_depth['declared_bit_depth']}-bit container "
            f"(+{bit_depth['headroom_db']:.0f}dB headroom)"
        )

    if gap["gap_ratio"] > 0.15:
        parts.append(
            f"spectral gap {int(gap['gap_hz'])}Hz below Nyquist "
            f"({int(gap['gap_ratio'] * 100)}% unused)"
        )

    if score >= 70:
        verdict = "High upscale potential"
    elif score >= 40:
        verdict = "Moderate upscale potential"
    else:
        verdict = "Low upscale potential"

    return f"{verdict} (score {score}/100): " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_upscale_potential(audio_path: str) -> dict:
    """Analyse a single audio file and return a comprehensive upscale potential dict.

    Uses spread-sampled FFT frames drawn evenly from across the whole file so
    that a quiet intro or outro does not skew the spectral results.

    Args:
        audio_path: path to any soundfile-supported file (wav, flac, ogg, mp3…)

    Returns a dict with keys:
        sample_rate_ceiling  — {effective_sr, nominal_sr, bandwidth_utilization}
        codec_artifacts      — {codec_artifacts_detected, likely_codec, confidence,
                                 cutoff_hz, pre_echo_events, sbr_detected}
        bit_depth_headroom   — {effective_bit_depth, declared_bit_depth, headroom_db}
        spectral_gap         — {rolloff_hz, nyquist_hz, gap_hz, gap_ratio}
        composite_score      — int 0–100  (higher = more benefit from upscaling)
        summary              — human-readable one-liner
    """
    path = str(audio_path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    freqs, power, sr = _spread_power_spectrum(path)

    sr_ceil = _sample_rate_ceiling(freqs, power, sr)
    codec = _codec_artifacts(freqs, power, sr, path)
    bit_depth = _bit_depth_headroom(path)
    gap = _spectral_gap(freqs, power, sr)
    score = _composite_score(sr_ceil, codec, bit_depth, gap)
    summary = _build_summary(score, sr_ceil, codec, bit_depth, gap)

    return {
        "sample_rate_ceiling": sr_ceil,
        "codec_artifacts": codec,
        "bit_depth_headroom": bit_depth,
        "spectral_gap": gap,
        "composite_score": score,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point:  python -m metrics.upscale_potential path/to/file.wav
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyse upscale potential of an audio file"
    )
    parser.add_argument("file", help="Audio file to analyse")
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted report"
    )
    args = parser.parse_args()

    result = analyze_upscale_potential(args.file)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # ── Formatted report ──────────────────────────────────────────────────────
    C_BOLD = "\033[1m"
    C_GREEN = "\033[0;32m"
    C_YELLOW = "\033[1;33m"
    C_RED = "\033[0;31m"
    C_CYAN = "\033[0;36m"
    C_DIM = "\033[2m"
    C_NC = "\033[0m"

    def _score_color(s: int) -> str:
        if s >= 70:
            return C_RED       # High potential — the file needs work
        if s >= 40:
            return C_YELLOW
        return C_GREEN         # Low potential — already hi-res

    def row(name: str, val: str, unit: str = "") -> None:
        print(f"  {name:<34s}  {val:>12s}  {C_DIM}{unit}{C_NC}")

    print(f"\n{C_BOLD}{'=' * 60}{C_NC}")
    print(f"{C_BOLD}  Upscale Potential: {Path(args.file).name}{C_NC}")
    print(f"{C_BOLD}{'=' * 60}{C_NC}")

    score = result["composite_score"]
    color = _score_color(score)
    print(f"\n  {C_BOLD}Composite Score: {color}{score} / 100{C_NC}")
    print(f"  {result['summary']}\n")

    print(f"  {C_CYAN}Sample Rate Ceiling{C_NC}")
    print(f"  {'-' * 44}")
    src = result["sample_rate_ceiling"]
    row("Nominal sample rate", str(src["nominal_sr"]), "Hz")
    row("Effective bandwidth", str(src["effective_sr"] // 2), "Hz")
    row("Bandwidth utilization", f"{src['bandwidth_utilization'] * 100:.1f}", "%")

    print(f"\n  {C_CYAN}Codec Artifacts{C_NC}")
    print(f"  {'-' * 44}")
    cod = result["codec_artifacts"]
    row("Artifacts detected", "YES" if cod["codec_artifacts_detected"] else "no")
    row("Likely codec", cod["likely_codec"])
    row("Confidence", f"{cod['confidence'] * 100:.0f}", "%")
    if cod["cutoff_hz"]:
        row("Hard cutoff", str(int(cod["cutoff_hz"])), "Hz")
    row("Pre-echo events", str(cod["pre_echo_events"]))
    row("SBR detected", "yes" if cod["sbr_detected"] else "no")

    print(f"\n  {C_CYAN}Bit Depth Headroom{C_NC}")
    print(f"  {'-' * 44}")
    bd = result["bit_depth_headroom"]
    row("Declared bit depth", str(bd["declared_bit_depth"]), "bit")
    row("Effective bit depth", str(bd["effective_bit_depth"]), "bit")
    row("Headroom", f"{bd['headroom_db']:.1f}", "dB")

    print(f"\n  {C_CYAN}Spectral Gap vs Nyquist{C_NC}")
    print(f"  {'-' * 44}")
    gp = result["spectral_gap"]
    row("Spectral rolloff (99%)", f"{gp['rolloff_hz']:.0f}", "Hz")
    row("Nyquist", f"{gp['nyquist_hz']:.0f}", "Hz")
    row("Gap", f"{gp['gap_hz']:.0f}", "Hz")
    row("Gap ratio", f"{gp['gap_ratio'] * 100:.1f}", "%")

    print(f"\n{C_BOLD}{'=' * 60}{C_NC}\n")


if __name__ == "__main__":
    _cli()
