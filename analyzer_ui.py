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
        subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "48000", "-ac", "2", tmp.name],
                       capture_output=True, check=True)
        data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
        import os; os.unlink(tmp.name)
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


def analyze_upscale_potential(path: str) -> dict:
    """Assess how much this file could benefit from enhancement."""
    from models.constants import OUTPUT_SAMPLE_RATE

    data, sr = _load_audio(path)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    target_sr = OUTPUT_SAMPLE_RATE

    result = {}

    # Sample rate headroom
    result["current_sr"] = sr
    result["target_sr"] = target_sr
    result["sr_headroom"] = max(0, target_sr - sr)

    # Spectral ceiling: find where energy drops below noise floor
    import librosa
    S = np.abs(librosa.stft(mono, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    mean_magnitude = S.mean(axis=1)

    # Find the frequency where magnitude drops to 1% of max (effective bandwidth)
    threshold = mean_magnitude.max() * 0.01
    above = np.where(mean_magnitude > threshold)[0]
    spectral_ceiling = freqs[above[-1]] if len(above) > 0 else 0
    result["spectral_ceiling_hz"] = round(float(spectral_ceiling))
    result["nyquist_hz"] = sr / 2

    # Codec artifact detection: sharp spectral cutoff
    # If spectral ceiling is well below Nyquist, likely lossy-encoded
    nyquist_ratio = spectral_ceiling / (sr / 2) if sr > 0 else 0
    result["nyquist_usage"] = round(float(nyquist_ratio), 3)

    # Check for sharp cutoff (codec signature)
    if len(above) > 10:
        top_freqs = freqs[above[-10:]]
        top_mags = mean_magnitude[above[-10:]]
        # Sharp drop = codec (gradual drop = natural)
        mag_gradient = np.diff(top_mags) / (np.diff(top_freqs) + 1e-8)
        sharpness = float(np.abs(mag_gradient).mean())
        result["cutoff_sharpness"] = round(sharpness, 6)
    else:
        result["cutoff_sharpness"] = 0.0

    # Bit depth assessment
    try:
        import soundfile as sf
        info = sf.info(path)
        bit_map = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}
        result["bit_depth"] = bit_map.get(info.subtype, 16)
    except Exception:
        result["bit_depth"] = 16

    # Enhancement verdict
    score = 0
    if sr < target_sr:
        score += 3  # Can upsample
    if nyquist_ratio < 0.85:
        score += 2  # Spectral content well below Nyquist (lossy/downsampled)
    if result["bit_depth"] < 24:
        score += 1  # Can improve dynamic range
    if sr <= 44100:
        score += 1  # CD quality or below

    if score >= 5:
        result["verdict"] = "high"
    elif score >= 3:
        result["verdict"] = "medium"
    elif score >= 1:
        result["verdict"] = "low"
    else:
        result["verdict"] = "none"

    result["score"] = score
    return result


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
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "48000", "-ac", "2", tmp.name],
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
        verdict_colors = {"high": "badge-bad", "medium": "badge-ok", "low": "badge-good", "none": "badge-good"}
        html += _section("section_upscale_potential")
        html += _row("metric_sr_headroom",
                      f"{up['current_sr']:,} Hz → {up['target_sr']:,} Hz (+{up['sr_headroom']:,} Hz)",
                      "desc_sr_headroom")
        html += _row("metric_spectral_ceiling",
                      f"{up['spectral_ceiling_hz']:,} Hz / {up['nyquist_hz']:,.0f} Hz ({up['nyquist_usage']:.0%} used)",
                      "desc_spectral_ceiling")
        html += _row("metric_bit_depth_headroom",
                      f"{up['bit_depth']}-bit → 24-bit",
                      "desc_bit_depth_headroom")
        verdict_class = verdict_colors.get(up["verdict"], "")
        html += f'<tr><td class="label">{t("metric_enhancement_verdict")}</td>'
        html += f'<td class="value"><span class="{verdict_class}">{t("verdict_" + up["verdict"])}</span></td></tr>'
        html += f'<tr><td colspan="2" class="desc">{t("desc_enhancement_verdict")}</td></tr>'
        html += "</table>"
    except Exception as e:
        html += f'<p class="muted">Upscale analysis: {e}</p>'

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

    # Reference comparison
    if ref_file is not None:
        ref_file = _convert_if_needed(ref_file)
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
    app.launch(server_name="0.0.0.0", server_port=args.port)
