#!/usr/bin/env python3
"""Browser-based audio quality analyzer.

Usage:
    python analyzer_ui.py
    # Opens http://localhost:7860 in your browser
"""

import json
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")


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
        return f'<span style="color: #00ff88; font-weight: bold;">{value:.2f} ✓</span>'
    elif value >= bad:
        return f'<span style="color: #ffaa00; font-weight: bold;">{value:.2f} ~</span>'
    else:
        return f'<span style="color: #ff4444; font-weight: bold;">{value:.2f} ✗</span>'


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


def analyze(audio_file, ref_file=None):
    """Main analysis function called by Gradio."""
    if audio_file is None:
        return "Upload an audio file to analyze.", None

    path = audio_file
    path = _convert_if_needed(path)

    # File info
    info = analyze_file_info(path)
    duration_m = int(info["duration_s"]) // 60
    duration_s = int(info["duration_s"]) % 60

    # Levels
    levels = analyze_levels(path)

    # Spectrum
    try:
        spec = analyze_spectrum(path)
    except Exception:
        spec = {}

    # Build HTML report
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; padding: 10px;">

    <h3 style="color: #00d4ff; margin-bottom: 5px;">📄 File Info</h3>
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">
        <tr><td style="color: #aaa; padding: 4px 8px;">File</td>
            <td style="color: #fff; padding: 4px 8px;"><b>{info['filename']}</b></td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Format</td>
            <td style="color: #fff; padding: 4px 8px;">{info['format']}</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Sample Rate</td>
            <td style="color: #fff; padding: 4px 8px;">{info['sample_rate']:,} Hz</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Bit Depth</td>
            <td style="color: #fff; padding: 4px 8px;">{info['bit_depth']}-bit</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Channels</td>
            <td style="color: #fff; padding: 4px 8px;">{info['channels']}</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Duration</td>
            <td style="color: #fff; padding: 4px 8px;">{duration_m}:{duration_s:02d}</td></tr>
    </table>

    <h3 style="color: #00d4ff; margin-bottom: 5px;">📊 Levels &amp; Dynamics</h3>
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">
        <tr><td style="color: #aaa; padding: 4px 8px;">Peak Level</td>
            <td style="color: #fff; padding: 4px 8px;">{levels['peak_db']:.1f} dBFS</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Loudest moment in the track. 0 dBFS = digital maximum. Typical: -1 to -0.1 dBFS for mastered music.</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">RMS Level</td>
            <td style="color: #fff; padding: 4px 8px;">{levels['rms_db']:.1f} dBFS</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Average perceived loudness. Streaming targets: -14 LUFS (Spotify), -16 LUFS (Apple). Loudness war: above -8.</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Crest Factor</td>
            <td style="padding: 4px 8px;">{quality_badge(levels['crest_factor_db'], (6, 10))} dB</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Peak-to-RMS ratio. Higher = more transient punch. Typical: 10-18 dB (well mastered), 4-6 dB (crushed/loudness war).</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Dynamic Range</td>
            <td style="padding: 4px 8px;">{quality_badge(levels['dynamic_range_db'], (6, 10))} dB</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Difference between loud and quiet sections. Typical: 12+ dB (classical/jazz), 8-12 dB (rock/pop), &lt;6 dB (EDM/compressed).</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Clipping</td>
            <td style="padding: 4px 8px;">{"<span style='color: #00ff88; font-weight: bold;'>0% clean</span>" if levels['clip_pct'] == 0 else "<span style='color: #ffaa00; font-weight: bold;'>" + f"{levels['clip_pct']:.3f}% mild</span>" if levels['clip_pct'] < 0.1 else "<span style='color: #ff8800; font-weight: bold;'>" + f"{levels['clip_pct']:.3f}% clipped</span>" if levels['clip_pct'] < 0.5 else "<span style='color: #ff4444; font-weight: bold;'>" + f"{levels['clip_pct']:.3f}% heavy</span>"} ({levels['clipped_samples']:,} samples)</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Samples at digital maximum. 0% = clean headroom, 0.01-0.1% = brief peaks (often acceptable), 0.1-0.5% = mildly clipped, &gt;0.5% = significant clipping.</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Max Consecutive Clips</td>
            <td style="padding: 4px 8px;">{"<span style='color: #00ff88;'>0</span>" if levels['max_consecutive_clips'] == 0 else f"<span style='color: #ff4444;'>{levels['max_consecutive_clips']}</span>"}</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Consecutive clipped samples indicate hard limiting/brick-wall clipping. 0 = clean, 1-3 = intersample peaks, 4+ = audible distortion.</td></tr>
    """

    if "stereo_width" in levels:
        html += f"""
        <tr><td style="color: #aaa; padding: 4px 8px;">Stereo Width</td>
            <td style="color: #fff; padding: 4px 8px;">{levels['stereo_width']:.4f}</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Side-to-mid energy ratio. 0 = mono, 0.1-0.3 = typical mix, &gt;0.5 = very wide/spatial.</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">L/R Correlation</td>
            <td style="padding: 4px 8px;">{quality_badge(levels['lr_correlation'], (0.3, 0.5))}</td></tr>
        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">How similar L and R channels are. Typical: 0.5-0.9. Near 1.0 = mono. Below 0.3 = possible phase issues.</td></tr>
        """

    html += "</table>"

    if spec:
        html += f"""
        <h3 style="color: #00d4ff; margin-bottom: 5px;">🎵 Spectrum</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">
            <tr><td style="color: #aaa; padding: 4px 8px;">Spectral Centroid</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['spectral_centroid_hz']:,.0f} Hz</td></tr>
            <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">"Center of gravity" of the frequency spectrum. Typical: 1000-3000 Hz (pop/rock), 500-1500 Hz (bass-heavy), 3000+ Hz (bright/electronic).</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Spectral Bandwidth</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['spectral_bandwidth_hz']:,.0f} Hz</td></tr>
            <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Spread of frequencies present. Typical: 2000-4000 Hz. Higher = richer harmonics and more complex arrangement.</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Rolloff (95%)</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['spectral_rolloff_95_hz']:,.0f} Hz</td></tr>
            <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Frequency below which 95% of energy sits. Typical: 8-15 kHz (MP3/compressed), 16-20 kHz (CD), 20+ kHz (hi-res).</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Nyquist</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['nyquist_hz']:,.0f} Hz</td></tr>
            <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Maximum representable frequency at this sample rate (sample_rate / 2).</td></tr>
        </table>
        """

    # Perceptual metrics (Audiobox)
    try:
        from metrics.evaluate import AudioMetrics
        m = AudioMetrics()
        # Audiobox needs a wav file (torchcodec can't handle webm/mp3 on some systems)
        wav_path = _convert_if_needed(path)
        ab = m.audiobox_aesthetics(wav_path)
        if isinstance(ab.score, dict):
            descriptions = {
                "PQ": "Clarity, fidelity, frequency balance, and spatial imaging. Typical: 4-6 (amateur), 6-8 (professional), 8+ (studio master).",
                "CE": "How pleasant and engaging the music feels. Typical: 3-5 (background), 5-7 (good), 7+ (highly engaging).",
                "PC": "Arrangement complexity: layers, instruments, rhythmic variation. Typical: 2-4 (simple), 5-7 (full band), 7+ (orchestral/dense).",
                "CU": "How well-suited for professional use (sync, broadcast). Typical: 3-5 (lo-fi/demo), 6-8 (broadcast ready), 8+ (premium).",
            }
            html += """<h3 style="color: #00d4ff; margin-bottom: 5px;">🎧 Perceptual Quality (Audiobox)</h3>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">"""
            labels = {"PQ": "Production Quality", "CE": "Enjoyment",
                      "PC": "Complexity", "CU": "Usefulness"}
            for k, v in ab.score.items():
                badge = quality_badge(v, (4, 6)) if k == "PQ" else f'<span style="color: #fff;">{v:.2f}</span>'
                desc = descriptions.get(k, "")
                html += f"""<tr><td style="color: #aaa; padding: 4px 8px;">{labels.get(k, k)}</td>
                    <td style="padding: 4px 8px;">{badge} / 10</td></tr>
                    <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">{desc}</td></tr>"""
            html += "</table>"
    except Exception as e:
        html += f'<p style="color: #888;">Audiobox not available: {e}</p>'

    # Reference comparison
    if ref_file is not None:
        ref_file = _convert_if_needed(ref_file)
        try:
            from metrics.evaluate import AudioMetrics
            m = AudioMetrics()

            html += """<h3 style="color: #ff8800; margin-bottom: 5px;">🔀 Reference Comparison</h3>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">"""

            sisnr = m.si_snr(ref_file, path)
            sdr = m.sdr(ref_file, path)
            html += f"""
                <tr><td style="color: #aaa; padding: 4px 8px;">SI-SNR</td>
                    <td style="padding: 4px 8px;">{quality_badge(sisnr.score, (10, 20))} dB</td></tr>
                <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">How closely the signal matches the reference, ignoring volume. Typical: 15-25 dB (good), 25+ dB (transparent), &lt;10 dB (significant artifacts).</td></tr>
                <tr><td style="color: #aaa; padding: 4px 8px;">SDR</td>
                    <td style="padding: 4px 8px;">{quality_badge(sdr.score, (10, 20))} dB</td></tr>
                <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Overall signal quality vs distortion. Typical: 15-25 dB (good enhancement), 25+ dB (near-transparent), &lt;10 dB (heavy processing).</td></tr>
            """

            chroma = m.chroma_similarity(ref_file, path)
            mfcc = m.mfcc_similarity(ref_file, path)
            onset = m.onset_f1(ref_file, path)
            html += f"""
                <tr><td style="color: #aaa; padding: 4px 8px;">Chroma (melody)</td>
                    <td style="padding: 4px 8px;">{quality_badge(chroma.score, (0.9, 0.95))}</td></tr>
                <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Are the notes and harmonies preserved? Typical: &gt;0.98 (transparent), 0.90-0.98 (minor coloring), &lt;0.90 (pitch/harmony damage).</td></tr>
                <tr><td style="color: #aaa; padding: 4px 8px;">MFCC (timbre)</td>
                    <td style="padding: 4px 8px;">{quality_badge(mfcc.score, (0.85, 0.9))}</td></tr>
                <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Does the "character" of instruments sound the same? Typical: &gt;0.95 (transparent), 0.85-0.95 (subtle EQ), &lt;0.85 (tonal shift).</td></tr>
                <tr><td style="color: #aaa; padding: 4px 8px;">Onset F1 (rhythm)</td>
                    <td style="padding: 4px 8px;">{quality_badge(onset.score, (0.8, 0.9))}</td></tr>
                <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Are the beats and note attacks in the right place? Typical: &gt;0.95 (transparent), 0.85-0.95 (slight smear), &lt;0.85 (timing damage).</td></tr>
            """
            html += "</table>"

            # Training losses (same as train.py optimizer targets)
            try:
                import torch
                from models.losses import MultiResolutionSTFTLoss, MelSpectrogramLoss
                from models.mastering_losses import MasteringLoss
                from models.constants import OUTPUT_SAMPLE_RATE

                ref_audio, ref_sr = _load_audio(ref_file)
                enh_audio, enh_sr = _load_audio(path)
                ref_mono = ref_audio.mean(axis=1) if ref_audio.shape[1] > 1 else ref_audio[:, 0]
                enh_mono = enh_audio.mean(axis=1) if enh_audio.shape[1] > 1 else enh_audio[:, 0]
                min_len = min(len(ref_mono), len(enh_mono))
                ref_t = torch.from_numpy(ref_mono[:min_len]).unsqueeze(0).unsqueeze(0)
                enh_t = torch.from_numpy(enh_mono[:min_len]).unsqueeze(0).unsqueeze(0)

                stft_loss_fn = MultiResolutionSTFTLoss()
                mel_loss_fn = MelSpectrogramLoss(sample_rate=ref_sr)

                with torch.no_grad():
                    stft_val = stft_loss_fn(enh_t, ref_t).item()
                    mel_val = mel_loss_fn(enh_t, ref_t).item()

                mastering_fn = MasteringLoss(sample_rate=ref_sr, device="cpu")
                with torch.no_grad():
                    mastering_total, mastering_details = mastering_fn(enh_t, ref_t)

                html += """<h3 style="color: #ff4488; margin-bottom: 5px;">🎛️ Training Losses (same as optimizer)</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">"""

                html += f"""
                    <tr><td style="color: #aaa; padding: 4px 8px;">Multi-Res STFT Loss</td>
                        <td style="color: #fff; padding: 4px 8px;">{stft_val:.4f}</td></tr>
                    <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Frequency-domain accuracy across multiple time-frequency resolutions. Lower = closer match.</td></tr>
                    <tr><td style="color: #aaa; padding: 4px 8px;">Mel Spectrogram Loss</td>
                        <td style="color: #fff; padding: 4px 8px;">{mel_val:.4f}</td></tr>
                    <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">Perceptually-weighted spectral difference using mel scale (mimics human hearing). Lower = better.</td></tr>
                """

                loss_descriptions = {
                    "perceptual_stft": ("Perceptual STFT", "A-weighted, mel-scaled STFT. Penalizes artifacts in the 2-5kHz sensitivity range."),
                    "stereo": ("Stereo Image", "Mid/side fidelity + stereo width preservation."),
                    "dynamics": ("Dynamics", "Crest factor + loudness matching. Ensures punch and DR are preserved."),
                    "encodec": ("EnCodec Embedding", "Neural perceptual distance in Meta's learned audio space. Captures timbre + texture."),
                    "mastering_total": ("Mastering Total", "Combined mastering loss (all above weighted and summed)."),
                }
                for k, v in mastering_details.items():
                    name, desc = loss_descriptions.get(k, (k, ""))
                    html += f"""
                        <tr><td style="color: #aaa; padding: 4px 8px;">{name}</td>
                            <td style="color: #fff; padding: 4px 8px;">{v:.4f}</td></tr>
                        <tr><td colspan="2" style="color: #666; padding: 0 8px 6px; font-size: 0.85em;">{desc}</td></tr>
                    """
                html += "</table>"
            except Exception as e:
                html += f'<p style="color: #888;">Training losses not available: {e}</p>'
        except Exception as e:
            html += f'<p style="color: #ff4444;">Reference comparison error: {e}</p>'

    html += "</div>"

    # Generate plots
    try:
        fig = generate_waveform_plot(path)
    except Exception:
        fig = None

    return html, fig


AUDIO_FILE_TYPES = [".wav", ".flac", ".mp3", ".ogg", ".aac", ".aiff", ".m4a", ".wma", ".opus", ".webm"]


def enhance(audio_file, checkpoint_file, skip_apollo, skip_audiosr):
    """Enhance audio using the trained model."""
    if audio_file is None:
        return "Upload an audio file to enhance.", None

    path = _convert_if_needed(audio_file)

    if checkpoint_file is None:
        return "Upload a model checkpoint (.pt file) to enhance audio.", None

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
        <div style="font-family: 'Segoe UI', sans-serif; padding: 10px;">
        <h3 style="color: #00ff88;">Enhancement Complete</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="color: #aaa; padding: 4px 8px;"></td>
                <td style="color: #ff8800; padding: 4px 8px; font-weight: bold;">Before</td>
                <td style="color: #00ff88; padding: 4px 8px; font-weight: bold;">After</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Sample Rate</td>
                <td style="color: #fff; padding: 4px 8px;">{info_before['sample_rate']:,} Hz</td>
                <td style="color: #00ff88; padding: 4px 8px;">{info_after['sample_rate']:,} Hz</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Bit Depth</td>
                <td style="color: #fff; padding: 4px 8px;">{info_before['bit_depth']}</td>
                <td style="color: #00ff88; padding: 4px 8px;">{info_after['bit_depth']}</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Format</td>
                <td style="color: #fff; padding: 4px 8px;">{info_before['format']}</td>
                <td style="color: #00ff88; padding: 4px 8px;">{info_after['format']}</td></tr>
        </table>
        <p style="color: #888; margin-top: 10px;">Tip: Use the Analyze tab to compare before/after quality metrics.</p>
        </div>
        """
        return html, out_path

    except Exception as e:
        return f'<p style="color: #ff4444;">Enhancement error: {e}</p>', None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Audio Enhancer & Analyzer",
    theme=gr.themes.Default(
        primary_hue="cyan",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { background: #1a1a2e !important; }
    .gr-button-primary { background: #00d4ff !important; }
    """,
) as app:
    gr.Markdown("# 🎵 Audio Enhancer & Analyzer")

    with gr.Tabs():
        with gr.TabItem("Enhance"):
            gr.Markdown("Upload audio and a trained checkpoint to enhance it to 96kHz/24-bit.")
            with gr.Row():
                with gr.Column(scale=1):
                    enh_audio = gr.File(label="Audio File", file_types=AUDIO_FILE_TYPES)
                    enh_checkpoint = gr.File(label="Model Checkpoint (.pt)", file_types=[".pt"])
                    enh_skip_apollo = gr.Checkbox(label="Skip Apollo (input is lossless)", value=True)
                    enh_skip_audiosr = gr.Checkbox(label="Skip AudioSR", value=True)
                    enh_btn = gr.Button("Enhance", variant="primary", size="lg")

                with gr.Column(scale=2):
                    enh_report = gr.HTML(label="Enhancement Report")
                    enh_output = gr.File(label="Enhanced Audio")

            enh_btn.click(
                fn=enhance,
                inputs=[enh_audio, enh_checkpoint, enh_skip_apollo, enh_skip_audiosr],
                outputs=[enh_report, enh_output],
            )

        with gr.TabItem("Analyze"):
            gr.Markdown("Upload audio to see technical info, dynamics, spectrum, and perceptual quality metrics.")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.File(label="Audio File", file_types=AUDIO_FILE_TYPES)
                    ref_input = gr.File(label="Reference File (optional)", file_types=AUDIO_FILE_TYPES)
                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=2):
                    report_html = gr.HTML(label="Analysis Report")

            plot_output = gr.Plot(label="Waveform & Spectrogram")

            analyze_btn.click(
                fn=analyze,
                inputs=[audio_input, ref_input],
                outputs=[report_html, plot_output],
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
