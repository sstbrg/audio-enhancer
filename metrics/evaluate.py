"""Audio quality metrics for evaluating enhancement results.

No-reference metrics (no ground truth needed):
  - PAM (Perceptual Audio Metric) — CLAP-based antonym prompt scoring
  - Audiobox Aesthetics (Meta) — Production Quality, Complexity, Enjoyment, Usefulness
  - MuQ-Eval — Music-specific quality, excellent at detecting MP3 artifacts

Reference-based metrics (compare enhanced vs original):
  - ViSQOL — Google's perceptual quality metric (MOS-LQO 1-5)
  - SI-SNR / SDR — Signal-to-distortion ratio
  - CDPAM — Contrastive Deep Perceptual Audio Metric

Usage:
    metrics = AudioMetrics(device="cuda")

    # No-reference (just score a file):
    scores = metrics.evaluate_no_reference("enhanced.wav")

    # Reference-based (compare enhanced vs ground truth):
    scores = metrics.evaluate_with_reference("reference.wav", "enhanced.wav")

    # Full evaluation (both):
    scores = metrics.evaluate_full("reference.wav", "enhanced.wav")
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


@dataclass
class MetricResult:
    """Container for metric results."""
    name: str
    score: float | dict
    higher_is_better: bool
    description: str = ""


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    input_file: str
    enhanced_file: str | None = None
    reference_file: str | None = None
    metrics: list[MetricResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60, "Audio Quality Evaluation Report", "=" * 60]
        lines.append(f"Enhanced:  {self.enhanced_file or self.input_file}")
        if self.reference_file:
            lines.append(f"Reference: {self.reference_file}")
        lines.append("-" * 60)

        for m in self.metrics:
            direction = "^" if m.higher_is_better else "v"
            if isinstance(m.score, dict):
                lines.append(f"\n  {m.name}:")
                for k, v in m.score.items():
                    lines.append(f"    {k}: {v:.4f}")
            else:
                lines.append(f"  {m.name}: {m.score:.4f}  ({direction} better)")
        lines.append("=" * 60)
        return "\n".join(lines)


def _load_audio_tensor(path: str, target_sr: int) -> torch.Tensor:
    """Load audio and resample to target_sr. Returns (samples,) mono float32."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)


class AudioMetrics:
    """Unified interface for all audio quality metrics."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pam = None
        self._audiobox = None
        self._muq = None
        self._visqol = None
        self._cdpam = None

    # ----------------------------------------------------------------
    # No-reference metrics
    # ----------------------------------------------------------------

    def pam_score(self, audio_path: str) -> MetricResult:
        """PAM: Perceptual Audio Metric using CLAP antonym prompts.

        Returns a score in [0, 1] where higher = cleaner/clearer audio.
        Requires: git clone https://github.com/soham97/PAM (add to sys.path)
        """
        try:
            if self._pam is None:
                import sys
                from pathlib import Path as P
                pam_dir = P(__file__).parent.parent / "third_party" / "PAM"
                if pam_dir.exists():
                    sys.path.insert(0, str(pam_dir))
                from PAM import PAM
                self._pam = PAM(use_cuda=(self.device == "cuda"))

            # Load and prepare audio (PAM expects 44100 Hz, 7-second chunks)
            waveform = _load_audio_tensor(audio_path, 44100)
            chunk_len = 44100 * 7  # 7 seconds

            scores = []
            for start in range(0, len(waveform), chunk_len):
                chunk = waveform[start:start + chunk_len]
                if len(chunk) < chunk_len:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_len - len(chunk)))
                chunk = chunk.unsqueeze(0)  # (1, samples)
                if self.device == "cuda":
                    chunk = chunk.cuda()
                # PAM.evaluate expects (batch, samples) and sample_index
                score, _ = self._pam.evaluate(chunk, [0])
                scores.append(score[0])

            avg = float(np.mean(scores))
            return MetricResult("PAM", avg, higher_is_better=True,
                                description="Perceptual clarity score (0-1)")
        except ImportError:
            warnings.warn("PAM not installed. Clone https://github.com/soham97/PAM to third_party/PAM")
            return MetricResult("PAM", float("nan"), higher_is_better=True,
                                description="Not available")

    def audiobox_aesthetics(self, audio_path: str) -> MetricResult:
        """Audiobox Aesthetics: Meta's 4-dimension audio quality predictor.

        Returns dict with CE (enjoyment), CU (usefulness), PC (complexity), PQ (production quality).
        Install: pip install audiobox-aesthetics
        """
        try:
            if self._audiobox is None:
                from audiobox_aesthetics.infer import initialize_predictor
                self._audiobox = initialize_predictor()

            results = self._audiobox.forward([{"path": audio_path}])
            scores = results[0]  # {CE, CU, PC, PQ}
            return MetricResult("Audiobox Aesthetics", scores, higher_is_better=True,
                                description="Production quality dimensions (0-10)")
        except ImportError:
            warnings.warn("audiobox-aesthetics not installed. pip install audiobox-aesthetics")
            return MetricResult("Audiobox Aesthetics", float("nan"), higher_is_better=True,
                                description="Not available")

    def muq_eval(self, audio_path: str) -> MetricResult:
        """MuQ-Eval: Music-specific quality metric.

        Returns MOS-like score (1-5). Excellent at detecting MP3 artifacts.
        Requires: git clone https://github.com/dgtql/MuQ-Eval (add to sys.path)
        """
        try:
            if self._muq is None:
                import sys
                from pathlib import Path as P
                muq_dir = P(__file__).parent.parent / "third_party" / "MuQ-Eval"
                if muq_dir.exists():
                    sys.path.insert(0, str(muq_dir))
                from omegaconf import OmegaConf
                from huggingface_hub import hf_hub_download
                from src.model import MusicQualityModel

                config_path = hf_hub_download("zhudi2825/MuQ-Eval-A1", "config.yaml")
                model_path = hf_hub_download("zhudi2825/MuQ-Eval-A1", "model_state_dict.pt")
                cfg = OmegaConf.load(config_path)
                model = MusicQualityModel(cfg)
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
                model.eval()
                if self.device == "cuda":
                    model = model.cuda()
                self._muq = model

            # MuQ expects 24000 Hz, max 10 seconds
            waveform = _load_audio_tensor(audio_path, 24000)
            max_samples = 24000 * 10
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]

            waveform = waveform.unsqueeze(0)  # (1, samples)
            if self.device == "cuda":
                waveform = waveform.cuda()

            with torch.no_grad():
                scores = self._muq(waveform)
                mi_score = scores["MI"].item()

            return MetricResult("MuQ-Eval", mi_score, higher_is_better=True,
                                description="Music quality MOS (1-5)")
        except (ImportError, Exception) as e:
            warnings.warn(f"MuQ-Eval not available: {e}")
            return MetricResult("MuQ-Eval", float("nan"), higher_is_better=True,
                                description="Not available")

    # ----------------------------------------------------------------
    # Reference-based metrics
    # ----------------------------------------------------------------

    def visqol(self, reference_path: str, degraded_path: str) -> MetricResult:
        """ViSQOL: Google's perceptual quality metric.

        Returns MOS-LQO score (1-5). Used by Apollo for evaluation.
        Install: pip install git+https://github.com/diggerdu/visqol-py.git
        """
        try:
            if self._visqol is None:
                try:
                    # Try official API first
                    from visqol import visqol_lib_py
                    from visqol.pb2 import visqol_config_pb2
                    import os

                    config = visqol_config_pb2.VisqolConfig()
                    config.audio.sample_rate = 48000
                    config.options.use_speech_scoring = False
                    config.options.svr_model_path = os.path.join(
                        os.path.dirname(visqol_lib_py.__file__),
                        "model", "libsvm_nu_svr_model.txt"
                    )
                    api = visqol_lib_py.VisqolApi()
                    api.Create(config)
                    self._visqol = ("official", api)
                except ImportError:
                    # Try community wrapper
                    import visqol_py
                    self._visqol = ("wrapper", visqol_py.ViSQOL(mode=visqol_py.ViSQOLMode.AUDIO))

            # Load at 48kHz for ViSQOL audio mode
            ref = _load_audio_tensor(reference_path, 48000).numpy()
            deg = _load_audio_tensor(degraded_path, 48000).numpy()

            # Align lengths
            min_len = min(len(ref), len(deg))
            ref = ref[:min_len]
            deg = deg[:min_len]

            kind, api = self._visqol
            if kind == "official":
                result = api.Measure(ref, deg)
                score = result.moslqo
            else:
                result = api.measure(reference_path, degraded_path)
                score = result.moslqo

            return MetricResult("ViSQOL", float(score), higher_is_better=True,
                                description="Perceptual MOS-LQO (1-5)")
        except ImportError:
            warnings.warn(
                "ViSQOL not installed. Install with:\n"
                "  pip install git+https://github.com/diggerdu/visqol-py.git"
            )
            return MetricResult("ViSQOL", float("nan"), higher_is_better=True,
                                description="Not available")

    def si_snr(self, reference_path: str, enhanced_path: str) -> MetricResult:
        """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.

        Higher is better. No external dependencies needed.
        """
        # Use consistent sample rate
        ref = _load_audio_tensor(reference_path, 48000)
        enh = _load_audio_tensor(enhanced_path, 48000)

        min_len = min(len(ref), len(enh))
        ref = ref[:min_len]
        enh = enh[:min_len]

        # SI-SNR calculation
        ref = ref - ref.mean()
        enh = enh - enh.mean()

        dot = torch.dot(enh, ref)
        s_target = dot * ref / (torch.dot(ref, ref) + 1e-8)
        e_noise = enh - s_target

        si_snr_val = 10 * torch.log10(
            torch.dot(s_target, s_target) / (torch.dot(e_noise, e_noise) + 1e-8) + 1e-8
        )

        return MetricResult("SI-SNR", float(si_snr_val), higher_is_better=True,
                            description="Scale-Invariant SNR (dB)")

    def sdr(self, reference_path: str, enhanced_path: str) -> MetricResult:
        """Signal-to-Distortion Ratio (SDR) in dB.

        Higher is better. No external dependencies needed.
        """
        ref = _load_audio_tensor(reference_path, 48000)
        enh = _load_audio_tensor(enhanced_path, 48000)

        min_len = min(len(ref), len(enh))
        ref = ref[:min_len]
        enh = enh[:min_len]

        noise = ref - enh
        sdr_val = 10 * torch.log10(
            torch.dot(ref, ref) / (torch.dot(noise, noise) + 1e-8) + 1e-8
        )

        return MetricResult("SDR", float(sdr_val), higher_is_better=True,
                            description="Signal-to-Distortion Ratio (dB)")

    def cdpam_score(self, reference_path: str, enhanced_path: str) -> MetricResult:
        """CDPAM: Contrastive Deep Perceptual Audio Metric.

        Returns perceptual distance (lower = more similar / better).
        Install: pip install cdpam
        """
        try:
            if self._cdpam is None:
                import cdpam
                self._cdpam = cdpam.DPAM()

            import cdpam
            wav_ref = cdpam.load_audio(reference_path)
            wav_enh = cdpam.load_audio(enhanced_path)
            dist = self._cdpam.forward(wav_ref, wav_enh)

            return MetricResult("CDPAM", float(dist), higher_is_better=False,
                                description="Perceptual distance (lower = better)")
        except ImportError:
            warnings.warn("CDPAM not installed. pip install cdpam")
            return MetricResult("CDPAM", float("nan"), higher_is_better=False,
                                description="Not available")

    # ----------------------------------------------------------------
    # Content preservation metrics
    # ----------------------------------------------------------------

    def chroma_similarity(self, reference_path: str, enhanced_path: str) -> MetricResult:
        """Chroma cosine similarity — verifies melody/harmony are preserved.

        Compares pitch class distributions over time between reference and enhanced.
        Score in [0, 1] where 1.0 = identical pitch content.
        No external dependencies beyond librosa.
        """
        import librosa

        ref_audio = _load_audio_tensor(reference_path, 22050).numpy()
        enh_audio = _load_audio_tensor(enhanced_path, 22050).numpy()

        min_len = min(len(ref_audio), len(enh_audio))
        ref_audio = ref_audio[:min_len]
        enh_audio = enh_audio[:min_len]

        chroma_ref = librosa.feature.chroma_stft(y=ref_audio, sr=22050, n_fft=4096, hop_length=512)
        chroma_enh = librosa.feature.chroma_stft(y=enh_audio, sr=22050, n_fft=4096, hop_length=512)

        # Align frame counts
        min_frames = min(chroma_ref.shape[1], chroma_enh.shape[1])
        chroma_ref = chroma_ref[:, :min_frames]
        chroma_enh = chroma_enh[:, :min_frames]

        # Cosine similarity per frame, then average
        dot = np.sum(chroma_ref * chroma_enh, axis=0)
        norm_ref = np.sqrt(np.sum(chroma_ref ** 2, axis=0)) + 1e-8
        norm_enh = np.sqrt(np.sum(chroma_enh ** 2, axis=0)) + 1e-8
        cos_sim = dot / (norm_ref * norm_enh)

        return MetricResult("Chroma Similarity", float(np.mean(cos_sim)),
                            higher_is_better=True,
                            description="Pitch/harmony preservation (0-1)")

    def mfcc_similarity(self, reference_path: str, enhanced_path: str) -> MetricResult:
        """MFCC cosine similarity — verifies timbral content is preserved.

        MFCCs capture the spectral envelope (timbre, instrument identity).
        Score in [0, 1] where 1.0 = identical timbral content.
        """
        import librosa

        ref_audio = _load_audio_tensor(reference_path, 22050).numpy()
        enh_audio = _load_audio_tensor(enhanced_path, 22050).numpy()

        min_len = min(len(ref_audio), len(enh_audio))
        ref_audio = ref_audio[:min_len]
        enh_audio = enh_audio[:min_len]

        mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=22050, n_mfcc=20)
        mfcc_enh = librosa.feature.mfcc(y=enh_audio, sr=22050, n_mfcc=20)

        min_frames = min(mfcc_ref.shape[1], mfcc_enh.shape[1])
        mfcc_ref = mfcc_ref[:, :min_frames]
        mfcc_enh = mfcc_enh[:, :min_frames]

        dot = np.sum(mfcc_ref * mfcc_enh, axis=0)
        norm_ref = np.sqrt(np.sum(mfcc_ref ** 2, axis=0)) + 1e-8
        norm_enh = np.sqrt(np.sum(mfcc_enh ** 2, axis=0)) + 1e-8
        cos_sim = dot / (norm_ref * norm_enh)

        return MetricResult("MFCC Similarity", float(np.mean(cos_sim)),
                            higher_is_better=True,
                            description="Timbral preservation (0-1)")

    def onset_f1(self, reference_path: str, enhanced_path: str,
                 tolerance_ms: float = 50.0) -> MetricResult:
        """Onset detection F1 score — verifies rhythm/timing is preserved.

        Detects note onsets in both files and measures how many match
        within a tolerance window. F1 in [0, 1] where 1.0 = identical timing.
        """
        import librosa

        ref_audio = _load_audio_tensor(reference_path, 22050).numpy()
        enh_audio = _load_audio_tensor(enhanced_path, 22050).numpy()

        onsets_ref = librosa.onset.onset_detect(y=ref_audio, sr=22050, units="time")
        onsets_enh = librosa.onset.onset_detect(y=enh_audio, sr=22050, units="time")

        if len(onsets_ref) == 0 and len(onsets_enh) == 0:
            return MetricResult("Onset F1", 1.0, higher_is_better=True,
                                description="Rhythm preservation (0-1)")
        if len(onsets_ref) == 0 or len(onsets_enh) == 0:
            return MetricResult("Onset F1", 0.0, higher_is_better=True,
                                description="Rhythm preservation (0-1)")

        tol = tolerance_ms / 1000.0
        matched_ref = set()
        matched_enh = set()
        for i, t_ref in enumerate(onsets_ref):
            for j, t_enh in enumerate(onsets_enh):
                if abs(t_ref - t_enh) <= tol and j not in matched_enh:
                    matched_ref.add(i)
                    matched_enh.add(j)
                    break

        tp = len(matched_ref)
        precision = tp / len(onsets_enh) if onsets_enh.size > 0 else 0
        recall = tp / len(onsets_ref) if onsets_ref.size > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return MetricResult("Onset F1", float(f1), higher_is_better=True,
                            description="Rhythm preservation (0-1)")

    def content_preservation(self, reference_path: str, enhanced_path: str) -> dict:
        """Run all content preservation metrics.

        Returns dict of results. A good enhancement should have:
        - Chroma Similarity > 0.95 (melody/harmony intact)
        - MFCC Similarity > 0.90 (timbre intact)
        - Onset F1 > 0.90 (rhythm intact)
        """
        return {
            "chroma": self.chroma_similarity(reference_path, enhanced_path),
            "mfcc": self.mfcc_similarity(reference_path, enhanced_path),
            "onset_f1": self.onset_f1(reference_path, enhanced_path),
        }

    # ----------------------------------------------------------------
    # Composite evaluation methods
    # ----------------------------------------------------------------

    def evaluate_no_reference(self, audio_path: str) -> EvaluationReport:
        """Run all no-reference metrics on a single audio file."""
        report = EvaluationReport(input_file=audio_path)

        report.metrics.append(self.pam_score(audio_path))
        report.metrics.append(self.audiobox_aesthetics(audio_path))
        report.metrics.append(self.muq_eval(audio_path))

        return report

    def evaluate_with_reference(
        self, reference_path: str, enhanced_path: str
    ) -> EvaluationReport:
        """Run all reference-based metrics comparing enhanced vs reference."""
        report = EvaluationReport(
            input_file=enhanced_path,
            enhanced_file=enhanced_path,
            reference_file=reference_path,
        )

        report.metrics.append(self.visqol(reference_path, enhanced_path))
        report.metrics.append(self.si_snr(reference_path, enhanced_path))
        report.metrics.append(self.sdr(reference_path, enhanced_path))
        report.metrics.append(self.cdpam_score(reference_path, enhanced_path))

        return report

    def evaluate_full(
        self, reference_path: str, enhanced_path: str
    ) -> EvaluationReport:
        """Run ALL metrics (no-reference + reference-based)."""
        report = EvaluationReport(
            input_file=enhanced_path,
            enhanced_file=enhanced_path,
            reference_file=reference_path,
        )

        # No-reference on enhanced file
        report.metrics.append(self.pam_score(enhanced_path))
        report.metrics.append(self.audiobox_aesthetics(enhanced_path))
        report.metrics.append(self.muq_eval(enhanced_path))

        # Reference-based
        report.metrics.append(self.visqol(reference_path, enhanced_path))
        report.metrics.append(self.si_snr(reference_path, enhanced_path))
        report.metrics.append(self.sdr(reference_path, enhanced_path))
        report.metrics.append(self.cdpam_score(reference_path, enhanced_path))

        # Content preservation
        report.metrics.append(self.chroma_similarity(reference_path, enhanced_path))
        report.metrics.append(self.mfcc_similarity(reference_path, enhanced_path))
        report.metrics.append(self.onset_f1(reference_path, enhanced_path))

        return report
