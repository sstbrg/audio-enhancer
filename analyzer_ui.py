#!/usr/bin/env python3
"""Browser-based audio quality analyzer.

Usage:
    python analyzer_ui.py
    python analyzer_ui.py --lang ru
    # Opens http://localhost:7860 in your browser
"""

import argparse
import json
import os
import warnings
from pathlib import Path

# Suppress TF/CUDA warnings (Essentia uses TF which expects old CUDA)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import gradio as gr
import numpy as np
import soundfile as sf

from models.constants import (
    INPUT_SAMPLE_RATE,
    UPSCALE_SCORE_HIGH,
    UPSCALE_SCORE_MEDIUM,
    UPSCALE_SCORE_LOW,
)


# ── Localization ──────────────────────────────────────────────────────────────

LOCALES_DIR = Path(__file__).parent / "locales"
_current_locale = {}


def load_locale(lang: str = "en") -> dict:
    path = LOCALES_DIR / f"{lang}.json"
    if not path.exists():
        path = LOCALES_DIR / "en.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_all_locales = {
    "English": load_locale("en"),
    "Русский": load_locale("ru"),
}
_current_lang = "English"


def t(key: str) -> str:
    """Translate a key using the current locale."""
    return _all_locales.get(_current_lang, _all_locales["English"]).get(key, key)


def set_lang(lang: str):
    global _current_lang
    _current_lang = lang


# Load default
_current_locale.update(load_locale("en"))


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio from any format (WAV, FLAC, MP3, AAC, OGG, WebM, etc.)."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data, sr
    except Exception:
        pass

    # Fallback for MP3/AAC/OGG/WebM via librosa (uses ffmpeg under the hood)
    try:
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=False)
        if audio.ndim == 1:
            data = audio.reshape(-1, 1)
        else:
            data = audio.T  # (samples, channels)
        return data, sr
    except Exception:
        pass

    # Last resort: ffmpeg to temp wav
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", str(INPUT_SAMPLE_RATE), "-ac", "2", tmp.name],
                       capture_output=True, check=True)
        data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
        os.unlink(tmp.name)
        return data, sr


def analyze_file_info(path: str) -> dict:
    try:
        info = sf.info(path)
        fmt = f"{info.format} / {info.subtype}"
        sr = info.samplerate
        channels = info.channels
        duration = info.duration
        bit_depth_map = {
            "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
            "FLOAT": 32, "DOUBLE": 64,
        }
        bit_depth = bit_depth_map.get(info.subtype, "?")
    except Exception:
        # MP3/AAC fallback
        import librosa
        y, sr = librosa.load(path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)
        channels = 1 if y.ndim == 1 else y.shape[0]
        ext = Path(path).suffix.lower()
        fmt = ext.replace(".", "").upper()
        bit_depth = "lossy"

    return {
        "filename": Path(path).name,
        "format": fmt,
        "sample_rate": sr,
        "bit_depth": bit_depth,
        "channels": channels,
        "duration_s": duration,
    }


def analyze_levels(path: str) -> dict:
    data, sr = _load_audio(path)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    rms = np.sqrt(np.mean(mono ** 2))
    peak = np.max(np.abs(mono))
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = 20 * np.log10(peak + 1e-10)
    crest_factor_db = peak_db - rms_db

    frame_size = int(sr * 0.4)
    hop = frame_size // 2
    frame_rms = []
    for i in range(0, len(mono) - frame_size, hop):
        frame = mono[i:i + frame_size]
        fr = np.sqrt(np.mean(frame ** 2))
        if fr > 1e-6:
            frame_rms.append(20 * np.log10(fr + 1e-10))

    dr = float(np.percentile(frame_rms, 95) - np.percentile(frame_rms, 10)) if frame_rms else 0.0

    # Clipping / saturation detection
    clip_threshold = 0.99
    clipped_samples = np.sum(np.abs(mono) >= clip_threshold)
    clip_pct = clipped_samples / len(mono) * 100

    # Inter-sample peak detection (samples that would clip after DAC reconstruction)
    # Simple check: consecutive near-max samples indicate true clipping vs transient peaks
    consecutive_clips = 0
    max_consecutive = 0
    for s in np.abs(mono):
        if s >= clip_threshold:
            consecutive_clips += 1
            max_consecutive = max(max_consecutive, consecutive_clips)
        else:
            consecutive_clips = 0

    result = {
        "peak_db": round(float(peak_db), 2),
        "rms_db": round(float(rms_db), 2),
        "crest_factor_db": round(float(crest_factor_db), 2),
        "dynamic_range_db": round(dr, 2),
        "clipped_samples": int(clipped_samples),
        "clip_pct": round(float(clip_pct), 4),
        "max_consecutive_clips": int(max_consecutive),
    }

    if data.shape[1] >= 2:
        mid = (data[:, 0] + data[:, 1]) / 2
        side = (data[:, 0] - data[:, 1]) / 2
        mid_energy = np.mean(mid ** 2)
        side_energy = np.mean(side ** 2)
        result["stereo_width"] = round(float(side_energy / (mid_energy + 1e-10)), 4)
        result["lr_correlation"] = round(float(np.corrcoef(data[:, 0], data[:, 1])[0, 1]), 4)

    return result


def analyze_spectrum(path: str) -> dict:
    import librosa
    data, sr = _load_audio(path)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.95)[0]

    return {
        "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
        "spectral_bandwidth_hz": round(float(np.mean(bandwidth)), 1),
        "spectral_rolloff_95_hz": round(float(np.mean(rolloff)), 1),
        "nyquist_hz": sr / 2,
    }


def _analyze_upscale_potential_stub(path: str) -> dict:
    """Fallback stub when metrics.upscale_potential is not yet available."""
    from models.constants import OUTPUT_SAMPLE_RATE

    data, sr = _load_audio(path)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    target_sr = OUTPUT_SAMPLE_RATE

    import librosa
    S = np.abs(librosa.stft(mono, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    mean_magnitude = S.mean(axis=1)

    threshold = mean_magnitude.max() * 0.01
    above = np.where(mean_magnitude > threshold)[0]
    spectral_ceiling = freqs[above[-1]] if len(above) > 0 else 0
    nyquist = sr / 2
    nyquist_ratio = spectral_ceiling / nyquist if nyquist > 0 else 0
    rolloff_hz = round(float(spectral_ceiling))
    gap_hz = max(0.0, float(nyquist - spectral_ceiling))
    gap_ratio = round(float(gap_hz / nyquist if nyquist > 0 else 0), 3)

    try:
        info = sf.info(path)
        bit_map = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}
        declared_bit_depth = bit_map.get(info.subtype, 16)
    except Exception:
        declared_bit_depth = 16

    noise_floor_db = float(20 * np.log10(np.percentile(np.abs(mono) + 1e-10, 1) + 1e-10))
    effective_bit_depth = max(8, min(24, round((noise_floor_db + 6) / -6)))
    headroom_db = round(declared_bit_depth * 6.02, 1)

    # Simple score: 0-100 based on available improvement potential
    sr_score = min(40, max(0, (target_sr - sr) / target_sr * 40))
    bw_score = (1.0 - nyquist_ratio) * 30
    bit_score = max(0, (24 - declared_bit_depth) / 8 * 20)
    raw_score = sr_score + bw_score + bit_score
    upscale_potential_score = round(min(100.0, max(0.0, raw_score)), 1)

    return {
        "effective_sr": round(float(spectral_ceiling * 2)),
        "nominal_sr": sr,
        "bandwidth_utilization": round(float(nyquist_ratio), 3),
        "codec_artifacts_detected": nyquist_ratio < 0.85,
        "likely_codec": "unknown",
        "confidence": 0.0,
        "effective_bit_depth": effective_bit_depth,
        "declared_bit_depth": declared_bit_depth,
        "headroom_db": headroom_db,
        "rolloff_hz": rolloff_hz,
        "nyquist_hz": round(float(nyquist)),
        "gap_hz": round(gap_hz),
        "gap_ratio": gap_ratio,
        "upscale_potential_score": upscale_potential_score,
    }


def analyze_upscale_potential(path: str) -> dict:
    """Assess how much this file could benefit from enhancement.

    Tries to import from metrics.upscale_potential (Anton's module).
    Falls back to the built-in stub when the module is not available.

    Always returns a flat dict compatible with the renderer:
        upscale_potential_score, effective_sr, nominal_sr,
        bandwidth_utilization, codec_artifacts_detected, likely_codec,
        confidence, effective_bit_depth, declared_bit_depth, headroom_db,
        rolloff_hz, nyquist_hz, gap_hz, gap_ratio
    """
    try:
        from metrics.upscale_potential import analyze_upscale_potential as _real
        raw = _real(path)
        # Flatten Anton's nested structure into the flat dict the renderer expects
        src = raw.get("sample_rate_ceiling", {})
        cod = raw.get("codec_artifacts", {})
        bd = raw.get("bit_depth_headroom", {})
        gap = raw.get("spectral_gap", {})
        return {
            "upscale_potential_score": float(raw.get("composite_score", 0)),
            "effective_sr": src.get("effective_sr", 0),
            "nominal_sr": src.get("nominal_sr", 0),
            "bandwidth_utilization": src.get("bandwidth_utilization", 0.0),
            "codec_artifacts_detected": cod.get("codec_artifacts_detected", False),
            "likely_codec": cod.get("likely_codec", "unknown"),
            "confidence": cod.get("confidence", 0.0),
            "effective_bit_depth": bd.get("effective_bit_depth", 0),
            "declared_bit_depth": bd.get("declared_bit_depth", 0),
            "headroom_db": bd.get("headroom_db", 0.0),
            "rolloff_hz": round(float(gap.get("rolloff_hz", 0))),
            "nyquist_hz": round(float(gap.get("nyquist_hz", 0))),
            "gap_hz": round(float(gap.get("gap_hz", 0))),
            "gap_ratio": gap.get("gap_ratio", 0.0),
        }
    except ImportError:
        return _analyze_upscale_potential_stub(path)


def generate_waveform_plot(path: str):
    """Generate waveform + spectrogram matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, sr = _load_audio(path)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    # Limit to 30 seconds for display
    max_samples = sr * 30
    if len(mono) > max_samples:
        mono = mono[:max_samples]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor="#1a1a2e")

    # Waveform
    time = np.arange(len(mono)) / sr
    ax1.plot(time, mono, color="#00d4ff", linewidth=0.3, alpha=0.8)
    ax1.set_ylabel("Amplitude", color="#aaa")
    ax1.set_title("Waveform", color="#fff", fontsize=12)
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="#888")
    ax1.set_xlim(0, time[-1])
    ax1.axhline(y=0, color="#333", linewidth=0.5)

    # Spectrogram
    ax2.specgram(mono, NFFT=4096, Fs=sr, noverlap=2048,
                 cmap="magma", vmin=-80, vmax=0)
    ax2.set_ylabel("Freq (Hz)", color="#aaa")
    ax2.set_xlabel("Time (s)", color="#aaa")
    ax2.set_title("Spectrogram", color="#fff", fontsize=12)
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="#888")
    ax2.set_ylim(0, min(sr / 2, 24000))

    plt.tight_layout()
    return fig


def quality_badge(value: float, thresholds: tuple) -> str:
    """Return colored HTML badge based on thresholds (bad, ok, good)."""
    bad, good = thresholds
    if value >= good:
        return f'<span class="badge-good">{value:.2f} ✓</span>'
    elif value >= bad:
        return f'<span class="badge-ok">{value:.2f} ~</span>'
    else:
        return f'<span class="badge-bad">{value:.2f} ✗</span>'


def _convert_if_needed(path: str) -> str:
    """Convert unsupported formats (webm, etc.) to wav via ffmpeg."""
    import subprocess, tempfile
    ext = Path(path).suffix.lower()
    if ext in (".wav", ".flac", ".aiff", ".aif"):
        return path
    try:
        # Try soundfile first
        sf.info(path)
        return path
    except Exception:
        pass
    # Convert via ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", str(INPUT_SAMPLE_RATE), "-ac", "2", tmp.name],
                   capture_output=True, check=True)
    return tmp.name


def _row(label_key: str, value: str, desc_key: str = "") -> str:
    """Generate a table row with optional description, using locale keys."""
    html = f'<tr><td class="label">{t(label_key)}</td><td class="value">{value}</td></tr>'
    if desc_key:
        html += f'<tr><td colspan="2" class="desc">{t(desc_key)}</td></tr>'
    return html


def _row_badge(label_key: str, value: float, thresholds: tuple, unit: str = "", desc_key: str = "") -> str:
    """Generate a table row with quality badge + description."""
    html = f'<tr><td class="label">{t(label_key)}</td><td class="value">{quality_badge(value, thresholds)} {unit}</td></tr>'
    if desc_key:
        html += f'<tr><td colspan="2" class="desc">{t(desc_key)}</td></tr>'
    return html


def _section(title_key: str) -> str:
    return f'<h3 class="section">{t(title_key)}</h3><table>'


def _section_ref(title_key: str) -> str:
    return f'<h3 class="ref">{t(title_key)}</h3><table>'


def _section_loss(title_key: str) -> str:
    return f'<h3 class="loss">{t(title_key)}</h3><table>'


def analyze(audio_file, ref_file, lang="English"):
    """Main analysis function called by Gradio."""
    set_lang(lang)
    if audio_file is None:
        return t("analyze_upload"), None

    path = _convert_if_needed(audio_file)

    info = analyze_file_info(path)
    duration_m = int(info["duration_s"]) // 60
    duration_s = int(info["duration_s"]) % 60
    levels = analyze_levels(path)

    try:
        spec = analyze_spectrum(path)
    except Exception:
        spec = {}

    # Build HTML report
    html = '<div class="report">'

    # File info
    html += _section("section_file_info")
    html += _row("metric_format", f"<b>{info['filename']}</b> &mdash; {info['format']}")
    html += _row("metric_sample_rate", f"{info['sample_rate']:,} Hz")
    html += _row("metric_bit_depth", f"{info['bit_depth']}-bit")
    html += _row("metric_channels", str(info["channels"]))
    html += _row("metric_duration", f"{duration_m}:{duration_s:02d}")
    html += "</table>"

    # Upscale potential
    try:
        up = analyze_upscale_potential(path)
        score = float(up.get("upscale_potential_score", 0))

        # Score color: red (high potential) → orange → green (already optimal)
        if score >= 60:
            score_class = "badge-bad"
        elif score >= 30:
            score_class = "badge-ok"
        else:
            score_class = "badge-good"

        # Summary text keyed by score range
        if score >= 60:
            summary_key = "summary_high"
        elif score >= 30:
            summary_key = "summary_medium"
        elif score >= 10:
            summary_key = "summary_low"
        else:
            summary_key = "summary_none"

        html += _section("section_upscale_potential")

        # Prominent score display
        html += (
            f'<tr><td class="label"><b>{t("metric_upscale_score")}</b></td>'
            f'<td class="value"><span class="{score_class}" style="font-size:1.4em;">'
            f'{score:.0f} / 100</span></td></tr>'
            f'<tr><td colspan="2" class="desc">{t("desc_upscale_score")}</td></tr>'
        )

        # Summary sentence
        html += (
            f'<tr><td colspan="2" class="desc" style="font-style:italic; color:#666;">'
            f'{t(summary_key)}</td></tr>'
        )

        # Sample rate
        nominal_sr = up.get("nominal_sr", 0)
        effective_sr = up.get("effective_sr", 0)
        html += _row(
            "metric_effective_sr",
            f"{effective_sr:,} Hz",
            "desc_effective_sr",
        )
        html += _row(
            "metric_nominal_sr",
            f"{nominal_sr:,} Hz",
        )

        # Bandwidth utilization
        bw = up.get("bandwidth_utilization", 0)
        bw_class = "badge-good" if bw >= 0.85 else ("badge-ok" if bw >= 0.6 else "badge-bad")
        html += (
            f'<tr><td class="label">{t("metric_bandwidth_utilization")}</td>'
            f'<td class="value"><span class="{bw_class}">{bw:.0%}</span></td></tr>'
            f'<tr><td colspan="2" class="desc">{t("desc_bandwidth_utilization")}</td></tr>'
        )

        # Codec artifacts
        artifacts = up.get("codec_artifacts_detected", False)
        confidence = up.get("confidence", 0.0)
        likely_codec = up.get("likely_codec", "unknown")
        if artifacts:
            codec_text = t("codec_detected").format(confidence=confidence)
            codec_class = "badge-bad"
        else:
            codec_text = t("codec_not_detected")
            codec_class = "badge-good"
        html += (
            f'<tr><td class="label">{t("metric_codec_artifacts")}</td>'
            f'<td class="value"><span class="{codec_class}">{codec_text}</span></td></tr>'
            f'<tr><td colspan="2" class="desc">{t("desc_codec_artifacts")}</td></tr>'
        )
        if artifacts and likely_codec and likely_codec != "unknown":
            html += (
                f'<tr><td class="label">{t("metric_likely_codec")}</td>'
                f'<td class="value">{likely_codec}</td></tr>'
                f'<tr><td class="label">{t("metric_codec_confidence")}</td>'
                f'<td class="value">{confidence:.0%}</td></tr>'
            )

        # Bit depth
        effective_bd = up.get("effective_bit_depth", 0)
        declared_bd = up.get("declared_bit_depth", 0)
        headroom_db = up.get("headroom_db", 0)
        html += _row(
            "metric_effective_bit_depth",
            f"{effective_bd}-bit",
            "desc_effective_bit_depth",
        )
        html += _row(
            "metric_declared_bit_depth",
            f"{declared_bd}-bit",
        )
        html += _row(
            "metric_headroom_db",
            f"{headroom_db:.1f} dB",
            "desc_headroom_db",
        )

        # Spectral gap
        rolloff_hz = up.get("rolloff_hz", 0)
        nyquist_hz = up.get("nyquist_hz", 0)
        gap_hz = up.get("gap_hz", 0)
        gap_ratio = up.get("gap_ratio", 0)
        gap_class = "badge-bad" if gap_ratio >= 0.4 else ("badge-ok" if gap_ratio >= 0.15 else "badge-good")
        html += _row("metric_rolloff_hz", f"{rolloff_hz:,} Hz")
        html += _row("metric_nyquist_hz", f"{nyquist_hz:,} Hz")
        html += (
            f'<tr><td class="label">{t("metric_gap_hz")}</td>'
            f'<td class="value">{gap_hz:,} Hz</td></tr>'
        )
        html += (
            f'<tr><td class="label">{t("metric_gap_ratio")}</td>'
            f'<td class="value"><span class="{gap_class}">{gap_ratio:.0%}</span></td></tr>'
            f'<tr><td colspan="2" class="desc">{t("desc_gap_ratio")}</td></tr>'
        )

        html += "</table>"
    except Exception as e:
        html += f'<p class="muted">{t("upscale_not_available")}: {e}</p>'

    # Levels & dynamics
    html += _section("section_levels")
    html += _row("metric_peak_level", f"{levels['peak_db']:.1f} dBFS", "desc_peak_level")
    html += _row("metric_rms_level", f"{levels['rms_db']:.1f} dBFS", "desc_rms_level")
    html += _row_badge("metric_crest_factor", levels["crest_factor_db"], (6, 10), "dB", "desc_crest_factor")
    html += _row_badge("metric_dynamic_range", levels["dynamic_range_db"], (6, 10), "dB", "desc_dynamic_range")

    # Clipping
    clip = levels["clip_pct"]
    if clip == 0:
        clip_badge = f'<span class="badge-good">0% {t("clip_clean")}</span>'
    elif clip < 0.1:
        clip_badge = f'<span class="badge-ok">{clip:.3f}% {t("clip_mild")}</span>'
    elif clip < 0.5:
        clip_badge = f'<span class="badge-ok">{clip:.3f}% {t("clip_clipped")}</span>'
    else:
        clip_badge = f'<span class="badge-bad">{clip:.3f}% {t("clip_heavy")}</span>'
    html += f'<tr><td class="label">{t("metric_clipping")}</td><td class="value">{clip_badge} ({levels["clipped_samples"]:,} samples)</td></tr>'
    html += f'<tr><td colspan="2" class="desc">{t("desc_clipping")}</td></tr>'

    mcc = levels["max_consecutive_clips"]
    mcc_badge = '<span class="badge-good">0</span>' if mcc == 0 else f'<span class="badge-bad">{mcc}</span>'
    html += f'<tr><td class="label">{t("metric_max_consecutive_clips")}</td><td class="value">{mcc_badge}</td></tr>'
    html += f'<tr><td colspan="2" class="desc">{t("desc_max_consecutive_clips")}</td></tr>'

    if "stereo_width" in levels:
        html += _row("metric_stereo_width", f"{levels['stereo_width']:.4f}", "desc_stereo_width")
        html += _row_badge("metric_lr_correlation", levels["lr_correlation"], (0.3, 0.5), "", "desc_lr_correlation")

    html += "</table>"

    # Spectrum
    if spec:
        html += _section("section_spectrum")
        html += _row("metric_spectral_centroid", f"{spec['spectral_centroid_hz']:,.0f} Hz", "desc_spectral_centroid")
        html += _row("metric_spectral_bandwidth", f"{spec['spectral_bandwidth_hz']:,.0f} Hz", "desc_spectral_bandwidth")
        html += _row("metric_spectral_rolloff", f"{spec['spectral_rolloff_95_hz']:,.0f} Hz", "desc_spectral_rolloff")
        html += _row("metric_nyquist", f"{spec['nyquist_hz']:,.0f} Hz", "desc_nyquist")
        html += "</table>"

    # Perceptual metrics (Audiobox)
    try:
        from metrics.evaluate import AudioMetrics
        m = AudioMetrics()
        wav_path = _convert_if_needed(path)
        ab = m.audiobox_aesthetics(wav_path)
        if isinstance(ab.score, dict):
            html += _section("section_perceptual")
            ab_keys = {"PQ": "metric_production_quality", "CE": "metric_enjoyment",
                        "PC": "metric_complexity", "CU": "metric_usefulness"}
            ab_descs = {"PQ": "desc_production_quality", "CE": "desc_enjoyment",
                        "PC": "desc_complexity", "CU": "desc_usefulness"}
            for k, v in ab.score.items():
                badge = quality_badge(v, (4, 6)) if k == "PQ" else f'<span class="value">{v:.2f}</span>'
                html += f'<tr><td class="label">{t(ab_keys.get(k, k))}</td><td class="value">{badge} / 10</td></tr>'
                html += f'<tr><td colspan="2" class="desc">{t(ab_descs.get(k, ""))}</td></tr>'
            html += "</table>"
    except Exception as e:
        html += f'<p style="color: #888;">{t("audiobox_not_available")}: {e}</p>'

    # Music intelligence (genre, mood, instruments, key, BPM)
    try:
        from metrics.music_analysis import MusicAnalyzer
        ma = MusicAnalyzer()
        music_info = ma.analyze(path)

        if music_info:
            html += _section("section_music_intelligence")

            if "key" in music_info:
                conf = music_info.get("key_confidence", 0)
                html += _row("metric_key", f"{music_info['key']} ({conf:.0%})", "desc_key")

            if "bpm" in music_info:
                html += _row("metric_bpm", str(music_info["bpm"]), "desc_bpm")

            if "clap_genre" in music_info and music_info["clap_genre"]:
                genre_html = ", ".join(
                    f'<span class="badge-ok">{g}</span> {p:.0%}' if i == 0 else f'{g} {p:.0%}'
                    for i, (g, p) in enumerate(music_info["clap_genre"].items())
                )
                html += _row("metric_genre", genre_html, "desc_genre")

            if "clap_mood" in music_info and music_info["clap_mood"]:
                mood_html = ", ".join(
                    f'<span class="badge-ok">{m}</span> {p:.0%}' if i == 0 else f'{m} {p:.0%}'
                    for i, (m, p) in enumerate(music_info["clap_mood"].items())
                )
                html += _row("metric_mood", mood_html, "desc_mood")

            if "mert_instruments" in music_info and music_info["mert_instruments"]:
                inst_html = ", ".join(
                    f'<span class="badge-good">{inst}</span> {prob:.0%}'
                    for inst, prob in music_info["mert_instruments"].items()
                )
                html += _row("metric_instruments", inst_html, "desc_instruments")

            html += "</table>"
    except Exception as e:
        html += f'<p class="muted">Music analysis: {e}</p>'

    # Before vs After comparison (when reference provided)
    if ref_file is not None:
        ref_file = _convert_if_needed(ref_file)

        # Side-by-side metrics comparison
        try:
            ref_info = analyze_file_info(ref_file)
            ref_levels = analyze_levels(ref_file)
            try:
                ref_spec = analyze_spectrum(ref_file)
            except Exception:
                ref_spec = {}

            html += f'<h3 class="ref">{t("section_comparison")}</h3>'
            html += f'<p class="desc">{t("desc_comparison")}</p>'
            html += '<table>'
            html += f'<tr><td class="label"><b>{t("col_metric")}</b></td><td class="value"><b>{t("col_original")}</b></td><td class="value"><b>{t("col_enhanced")}</b></td><td class="value"><b>{t("col_change")}</b></td></tr>'

            # Sample rate
            html += f'<tr><td class="label">{t("metric_sample_rate")}</td><td class="value">{ref_info["sample_rate"]:,} Hz</td><td class="value">{info["sample_rate"]:,} Hz</td>'
            sr_delta = info["sample_rate"] - ref_info["sample_rate"]
            if sr_delta > 0:
                html += f'<td class="badge-good">+{sr_delta:,} Hz</td></tr>'
            elif sr_delta < 0:
                html += f'<td class="badge-bad">{sr_delta:,} Hz</td></tr>'
            else:
                html += f'<td class="muted">{t("unchanged")}</td></tr>'

            # Dynamic range
            html += f'<tr><td class="label">{t("metric_dynamic_range")}</td><td class="value">{ref_levels["dynamic_range_db"]:.1f} dB</td><td class="value">{levels["dynamic_range_db"]:.1f} dB</td>'
            dr_delta = levels["dynamic_range_db"] - ref_levels["dynamic_range_db"]
            if dr_delta > 0.5:
                html += f'<td class="badge-good">+{dr_delta:.1f} dB {t("improved")}</td></tr>'
            elif dr_delta < -0.5:
                html += f'<td class="badge-bad">{dr_delta:.1f} dB {t("degraded")}</td></tr>'
            else:
                html += f'<td class="muted">{t("unchanged")}</td></tr>'

            # Crest factor
            html += f'<tr><td class="label">{t("metric_crest_factor")}</td><td class="value">{ref_levels["crest_factor_db"]:.1f} dB</td><td class="value">{levels["crest_factor_db"]:.1f} dB</td>'
            cf_delta = levels["crest_factor_db"] - ref_levels["crest_factor_db"]
            if cf_delta > 0.5:
                html += f'<td class="badge-good">+{cf_delta:.1f} dB {t("improved")}</td></tr>'
            elif cf_delta < -0.5:
                html += f'<td class="badge-bad">{cf_delta:.1f} dB {t("degraded")}</td></tr>'
            else:
                html += f'<td class="muted">{t("unchanged")}</td></tr>'

            # Clipping
            html += f'<tr><td class="label">{t("metric_clipping")}</td><td class="value">{ref_levels["clip_pct"]:.3f}%</td><td class="value">{levels["clip_pct"]:.3f}%</td>'
            clip_delta = levels["clip_pct"] - ref_levels["clip_pct"]
            if clip_delta < -0.001:
                html += f'<td class="badge-good">{clip_delta:.3f}% {t("improved")}</td></tr>'
            elif clip_delta > 0.001:
                html += f'<td class="badge-bad">+{clip_delta:.3f}% {t("degraded")}</td></tr>'
            else:
                html += f'<td class="muted">{t("unchanged")}</td></tr>'

            # Spectral rolloff
            if spec and ref_spec:
                html += f'<tr><td class="label">{t("metric_spectral_rolloff")}</td><td class="value">{ref_spec["spectral_rolloff_95_hz"]:,.0f} Hz</td><td class="value">{spec["spectral_rolloff_95_hz"]:,.0f} Hz</td>'
                ro_delta = spec["spectral_rolloff_95_hz"] - ref_spec["spectral_rolloff_95_hz"]
                if ro_delta > 500:
                    html += f'<td class="badge-good">+{ro_delta:,.0f} Hz {t("improved")}</td></tr>'
                elif ro_delta < -500:
                    html += f'<td class="badge-bad">{ro_delta:,.0f} Hz {t("degraded")}</td></tr>'
                else:
                    html += f'<td class="muted">{t("unchanged")}</td></tr>'

            # Stereo width
            if "stereo_width" in levels and "stereo_width" in ref_levels:
                html += f'<tr><td class="label">{t("metric_stereo_width")}</td><td class="value">{ref_levels["stereo_width"]:.4f}</td><td class="value">{levels["stereo_width"]:.4f}</td>'
                sw_delta = levels["stereo_width"] - ref_levels["stereo_width"]
                if abs(sw_delta) < 0.01:
                    html += f'<td class="muted">{t("unchanged")}</td></tr>'
                elif sw_delta > 0:
                    html += f'<td class="badge-ok">+{sw_delta:.4f}</td></tr>'
                else:
                    html += f'<td class="badge-ok">{sw_delta:.4f}</td></tr>'

            html += '</table>'
        except Exception as e:
            html += f'<p class="muted">Comparison: {e}</p>'

        # Reference-based metrics
        try:
            from metrics.evaluate import AudioMetrics
            m = AudioMetrics()

            html += _section_ref("section_reference")
            sisnr = m.si_snr(ref_file, path)
            sdr = m.sdr(ref_file, path)
            html += _row_badge("metric_si_snr", sisnr.score, (10, 20), "dB", "desc_si_snr")
            html += _row_badge("metric_sdr", sdr.score, (10, 20), "dB", "desc_sdr")

            chroma = m.chroma_similarity(ref_file, path)
            mfcc = m.mfcc_similarity(ref_file, path)
            onset = m.onset_f1(ref_file, path)
            html += _row_badge("metric_chroma", chroma.score, (0.9, 0.95), "", "desc_chroma")
            html += _row_badge("metric_mfcc", mfcc.score, (0.85, 0.9), "", "desc_mfcc")
            html += _row_badge("metric_onset_f1", onset.score, (0.8, 0.9), "", "desc_onset_f1")
            html += "</table>"

            # Training losses
            try:
                import torch
                from models.losses import MultiResolutionSTFTLoss, MelSpectrogramLoss
                from models.mastering_losses import MasteringLoss

                ref_audio, ref_sr = _load_audio(ref_file)
                enh_audio, enh_sr = _load_audio(path)
                ref_mono = ref_audio.mean(axis=1) if ref_audio.shape[1] > 1 else ref_audio[:, 0]
                enh_mono = enh_audio.mean(axis=1) if enh_audio.shape[1] > 1 else enh_audio[:, 0]
                min_len = min(len(ref_mono), len(enh_mono))
                ref_t = torch.from_numpy(ref_mono[:min_len]).unsqueeze(0).unsqueeze(0)
                enh_t = torch.from_numpy(enh_mono[:min_len]).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    stft_val = MultiResolutionSTFTLoss()(enh_t, ref_t).item()
                    mel_val = MelSpectrogramLoss(sample_rate=ref_sr)(enh_t, ref_t).item()
                    _, mastering_details = MasteringLoss(sample_rate=ref_sr, device="cpu")(enh_t, ref_t)

                html += _section_loss("section_training_losses")
                html += _row("metric_stft_loss", f"{stft_val:.4f}", "desc_stft_loss")
                html += _row("metric_mel_loss", f"{mel_val:.4f}", "desc_mel_loss")

                mastering_key_map = {
                    "perceptual_stft": "metric_perceptual_stft",
                    "stereo": "metric_stereo_image",
                    "dynamics": "metric_dynamics_loss",
                    "encodec": "metric_encodec",
                    "mastering_total": "metric_mastering_total",
                }
                mastering_desc_map = {
                    "perceptual_stft": "desc_perceptual_stft",
                    "stereo": "desc_stereo_image",
                    "dynamics": "desc_dynamics_loss",
                    "encodec": "desc_encodec",
                    "mastering_total": "desc_mastering_total",
                }
                for k, v in mastering_details.items():
                    html += _row(mastering_key_map.get(k, k), f"{v:.4f}", mastering_desc_map.get(k, ""))
                html += "</table>"
            except Exception as e:
                html += f'<p style="color: #888;">{e}</p>'
        except Exception as e:
            html += f'<p style="color: #ff4444;">{e}</p>'

    html += "</div>"

    try:
        fig = generate_waveform_plot(path)
    except Exception:
        fig = None

    return html, fig


AUDIO_FILE_TYPES = [".wav", ".flac", ".mp3", ".ogg", ".aac", ".aiff", ".m4a", ".wma", ".opus", ".webm"]


def enhance(audio_file, checkpoint_file, skip_apollo, skip_audiosr, lang="English"):
    """Enhance audio using the trained model."""
    set_lang(lang)
    if audio_file is None:
        return t("enhance_upload_audio"), None

    path = _convert_if_needed(audio_file)

    if checkpoint_file is None:
        return t("enhance_upload_checkpoint"), None

    try:
        from enhance import AudioEnhancer
        import tempfile

        enhancer = AudioEnhancer(
            config_path="configs/phase0.yaml",
            gan_checkpoint=checkpoint_file,
        )

        stem = Path(audio_file).stem
        out_path = tempfile.NamedTemporaryFile(suffix=f"_{stem}_enhanced.wav", delete=False).name

        enhancer.enhance_file(
            path, out_path,
            use_apollo=not skip_apollo,
            use_audiosr=not skip_audiosr,
            use_gan=True,
        )

        # Force cleanup of model to free GPU memory
        del enhancer
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        info_before = analyze_file_info(audio_file)
        info_after = analyze_file_info(out_path)

        html = f"""
        <div class="report">
        <h3 class="done">{t("enhance_complete")}</h3>
        <table>
            <tr><td class="label"></td>
                <td class="ref" style="font-weight: bold;">{t("enhance_before")}</td>
                <td class="badge-good" style="font-weight: bold;">{t("enhance_after")}</td></tr>
            <tr><td class="label">{t("metric_sample_rate")}</td>
                <td class="value">{info_before['sample_rate']:,} Hz</td>
                <td class="badge-good">{info_after['sample_rate']:,} Hz</td></tr>
            <tr><td class="label">{t("metric_bit_depth")}</td>
                <td class="value">{info_before['bit_depth']}</td>
                <td class="badge-good">{info_after['bit_depth']}</td></tr>
            <tr><td class="label">{t("metric_format")}</td>
                <td class="value">{info_before['format']}</td>
                <td class="badge-good">{info_after['format']}</td></tr>
        </table>
        <p class="tip">{t("enhance_tip")}</p>
        </div>
        """
        return html, out_path

    except Exception as e:
        return f'<p class="error">{t("enhance_error")}: {e}</p>', None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Audio Enhancer & Analyzer",
    theme=gr.themes.Soft(primary_hue="cyan"),
    css="""
    .gradio-container { max-width: 80% !important; margin: 0 auto !important; }
    .report { font-family: 'Segoe UI', sans-serif; padding: 10px; }
    .report h3 { margin-bottom: 5px; }
    .report h3.section { color: #0088cc; }
    .report h3.ref { color: #cc6600; }
    .report h3.loss { color: #cc4488; }
    .report h3.done { color: #00aa66; }
    .report table { width: 100%; border-collapse: collapse; margin-bottom: 15px; }
    .report td.label { color: #666; padding: 4px 8px; }
    .report td.value { padding: 4px 8px; }
    .report td.desc { color: #999; padding: 0 8px 6px; font-size: 0.85em; }
    .report .badge-good { color: #00aa66; font-weight: bold; }
    .report .badge-ok { color: #cc8800; font-weight: bold; }
    .report .badge-bad { color: #cc3333; font-weight: bold; }
    .report .error { color: #cc3333; }
    .report .muted { color: #999; }
    .report .tip { color: #999; margin-top: 10px; }
    """,
) as app:
    with gr.Row():
        gr.Markdown(f"# 🎵 {t('app_title')}")
        lang_selector = gr.Radio(
            choices=["English", "Русский"],
            value="English" if "en" in str(LOCALES_DIR / "en.json") else "Русский",
            label="🌐",
            scale=0,
        )
    with gr.Tabs():
        with gr.TabItem("Enhance / Улучшить"):
            with gr.Row():
                with gr.Column(scale=1):
                    enh_audio = gr.File(label="Audio / Аудио", file_types=AUDIO_FILE_TYPES)
                    enh_checkpoint = gr.File(label="Checkpoint (.pt)", file_types=[".pt"])
                    enh_skip_apollo = gr.Checkbox(label="Skip Apollo", value=True)
                    enh_skip_audiosr = gr.Checkbox(label="Skip AudioSR", value=True)
                    enh_btn = gr.Button("Enhance / Улучшить", variant="primary", size="lg")

                with gr.Column(scale=2):
                    enh_report = gr.HTML()
                    enh_output = gr.File(label="Output")

            enh_btn.click(
                fn=enhance,
                inputs=[enh_audio, enh_checkpoint, enh_skip_apollo, enh_skip_audiosr, lang_selector],
                outputs=[enh_report, enh_output],
            )

        with gr.TabItem("Analyze / Анализ"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.File(label="Audio / Аудио", file_types=AUDIO_FILE_TYPES)
                    ref_input = gr.File(label="Reference (optional)", file_types=AUDIO_FILE_TYPES)
                    analyze_btn = gr.Button("Analyze / Анализ", variant="primary", size="lg")

                with gr.Column(scale=2):
                    report_html = gr.HTML()

            plot_output = gr.Plot(label="Waveform & Spectrogram")

            analyze_btn.click(
                fn=analyze,
                inputs=[audio_input, ref_input, lang_selector],
                outputs=[report_html, plot_output],
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en", choices=["en", "ru"], help="UI language")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    _current_locale.update(load_locale(args.lang))
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=args.port)
