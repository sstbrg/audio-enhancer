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


def analyze_file_info(path: str) -> dict:
    info = sf.info(path)
    bit_depth_map = {
        "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
        "FLOAT": 32, "DOUBLE": 64,
    }
    return {
        "filename": Path(path).name,
        "format": f"{info.format} / {info.subtype}",
        "sample_rate": info.samplerate,
        "bit_depth": bit_depth_map.get(info.subtype, "?"),
        "channels": info.channels,
        "duration_s": info.duration,
    }


def analyze_levels(path: str) -> dict:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
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

    result = {
        "peak_db": round(float(peak_db), 2),
        "rms_db": round(float(rms_db), 2),
        "crest_factor_db": round(float(crest_factor_db), 2),
        "dynamic_range_db": round(dr, 2),
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
    data, sr = sf.read(path, dtype="float32", always_2d=True)
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

    data, sr = sf.read(path, dtype="float32", always_2d=True)
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


def analyze(audio_file, ref_file=None):
    """Main analysis function called by Gradio."""
    if audio_file is None:
        return "Upload an audio file to analyze.", None, ""

    path = audio_file

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
        <tr><td style="color: #aaa; padding: 4px 8px;">RMS Level</td>
            <td style="color: #fff; padding: 4px 8px;">{levels['rms_db']:.1f} dBFS</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Crest Factor</td>
            <td style="padding: 4px 8px;">{quality_badge(levels['crest_factor_db'], (6, 10))} dB</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">Dynamic Range</td>
            <td style="padding: 4px 8px;">{quality_badge(levels['dynamic_range_db'], (6, 10))} dB</td></tr>
    """

    if "stereo_width" in levels:
        html += f"""
        <tr><td style="color: #aaa; padding: 4px 8px;">Stereo Width</td>
            <td style="color: #fff; padding: 4px 8px;">{levels['stereo_width']:.4f}</td></tr>
        <tr><td style="color: #aaa; padding: 4px 8px;">L/R Correlation</td>
            <td style="padding: 4px 8px;">{quality_badge(levels['lr_correlation'], (0.3, 0.5))}</td></tr>
        """

    html += "</table>"

    if spec:
        html += f"""
        <h3 style="color: #00d4ff; margin-bottom: 5px;">🎵 Spectrum</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">
            <tr><td style="color: #aaa; padding: 4px 8px;">Spectral Centroid</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['spectral_centroid_hz']:,.0f} Hz</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Spectral Bandwidth</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['spectral_bandwidth_hz']:,.0f} Hz</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Rolloff (95%)</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['spectral_rolloff_95_hz']:,.0f} Hz</td></tr>
            <tr><td style="color: #aaa; padding: 4px 8px;">Nyquist</td>
                <td style="color: #fff; padding: 4px 8px;">{spec['nyquist_hz']:,.0f} Hz</td></tr>
        </table>
        """

    # Perceptual metrics (Audiobox)
    try:
        from metrics.evaluate import AudioMetrics
        m = AudioMetrics()
        ab = m.audiobox_aesthetics(path)
        if isinstance(ab.score, dict):
            html += """<h3 style="color: #00d4ff; margin-bottom: 5px;">🎧 Perceptual Quality (Audiobox)</h3>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">"""
            labels = {"PQ": "Production Quality", "CE": "Enjoyment",
                      "PC": "Complexity", "CU": "Usefulness"}
            for k, v in ab.score.items():
                badge = quality_badge(v, (4, 6)) if k == "PQ" else f'<span style="color: #fff;">{v:.2f}</span>'
                html += f"""<tr><td style="color: #aaa; padding: 4px 8px;">{labels.get(k, k)}</td>
                    <td style="padding: 4px 8px;">{badge} / 10</td></tr>"""
            html += "</table>"
    except Exception as e:
        html += f'<p style="color: #888;">Audiobox not available: {e}</p>'

    # Reference comparison
    if ref_file is not None:
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
                <tr><td style="color: #aaa; padding: 4px 8px;">SDR</td>
                    <td style="padding: 4px 8px;">{quality_badge(sdr.score, (10, 20))} dB</td></tr>
            """

            chroma = m.chroma_similarity(ref_file, path)
            mfcc = m.mfcc_similarity(ref_file, path)
            onset = m.onset_f1(ref_file, path)
            html += f"""
                <tr><td style="color: #aaa; padding: 4px 8px;">Chroma (melody)</td>
                    <td style="padding: 4px 8px;">{quality_badge(chroma.score, (0.9, 0.95))}</td></tr>
                <tr><td style="color: #aaa; padding: 4px 8px;">MFCC (timbre)</td>
                    <td style="padding: 4px 8px;">{quality_badge(mfcc.score, (0.85, 0.9))}</td></tr>
                <tr><td style="color: #aaa; padding: 4px 8px;">Onset F1 (rhythm)</td>
                    <td style="padding: 4px 8px;">{quality_badge(onset.score, (0.8, 0.9))}</td></tr>
            """
            html += "</table>"
        except Exception as e:
            html += f'<p style="color: #ff4444;">Reference comparison error: {e}</p>'

    html += "</div>"

    # Generate plots
    try:
        fig = generate_waveform_plot(path)
    except Exception:
        fig = None

    return html, fig


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Audio Analyzer",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
    ),
) as app:
    gr.Markdown("# 🎵 Audio Quality Analyzer")
    gr.Markdown("Upload an audio file to see technical info, dynamics, spectrum, and perceptual quality metrics.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Audio File", type="filepath")
            ref_input = gr.Audio(label="Reference File (optional)", type="filepath")
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
