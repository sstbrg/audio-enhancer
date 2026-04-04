# Phase 1: Degradation Pipeline Plan

**Author:** Lara (AI Team Lead)
**Date:** 2024-04-04
**Status:** Planning

## Overview

Phase 1 trains the same HiFi-GAN architecture to restore quality from degraded audio.
The model learns to undo codec artifacts, bad EQ, over-compression, stereo damage, and other common real-world quality problems by training on (degraded, clean) pairs.

The degradation pipeline takes clean, high-quality audio and applies randomized chains of realistic degradations. The model then learns to invert those degradations.

## Degradation Categories

### 1. Codec Artifact Simulation

**Goal:** Teach the model to remove lossy compression artifacts (pre-echo, spectral holes, birdie artifacts, bandwidth limiting).

| Codec | Bitrates | Library | Notes |
|-------|----------|---------|-------|
| MP3 (LAME) | 64, 96, 128, 160, 192, 256, 320 kbps | `pydub` + `ffmpeg` | VBR and CBR modes |
| AAC (FDK) | 64, 96, 128, 160, 192, 256 kbps | `ffmpeg` (libfdk_aac) | HE-AAC v1/v2 at low bitrates |
| OGG Vorbis | 64, 96, 128, 160, 192, 256, 320 kbps | `ffmpeg` (libvorbis) | Quality levels -1 to 10 |
| Opus | 32, 48, 64, 96, 128 kbps | `ffmpeg` (libopus) | Very aggressive at low rates |
| WMA | 96, 128, 192 kbps | `ffmpeg` | Legacy format, still common |

**Implementation approach:**
- Encode to lossy format in a temp file via `ffmpeg` subprocess, decode back to WAV.
- For each sample, randomly select a codec and bitrate.
- Weight distribution: favor mid-range bitrates (128-192 kbps) as these are the most common real-world scenario; low bitrates (64-96 kbps) at ~15% probability for aggressive artifact training.
- Double-encode at ~5% probability (simulates transcoding chains: e.g., MP3->AAC).

**Constants to add to `models/constants.py`:**
```python
CODEC_TYPES = ["mp3", "aac", "ogg", "opus", "wma"]
CODEC_BITRATE_RANGES = {
    "mp3": [64, 96, 128, 160, 192, 256, 320],
    "aac": [64, 96, 128, 160, 192, 256],
    "ogg": [64, 96, 128, 160, 192, 256, 320],
    "opus": [32, 48, 64, 96, 128],
    "wma": [96, 128, 192],
}
```

### 2. Bad EQ Simulation

**Goal:** Teach the model to correct frequency imbalances from poor mastering or bad playback equipment.

**Degradation types:**
- **Random parametric EQ:** 2-5 bands, each with random center frequency (60 Hz - 16 kHz, log-distributed), random gain (-12 dB to +12 dB), random Q (0.3 to 8.0).
- **Resonant peaks:** Narrow Q (6-12) boosts at random frequencies, simulating room modes or equipment resonances. Gain: +6 to +15 dB.
- **Scooped mids:** Common bad-EQ pattern. Cut 500 Hz - 2 kHz by 6-12 dB, boost below 150 Hz and above 8 kHz.
- **Excessive bass boost:** Shelf or peak below 200 Hz, +6 to +15 dB. Very common in consumer audio.
- **Harsh treble:** Boost 2-5 kHz by 4-10 dB with moderate Q.
- **Muddy low-mids:** Boost 200-500 Hz by 4-10 dB, simulating proximity effect or bad room acoustics.
- **Bandwidth limiting:** Lowpass at 12-20 kHz (simulating AM radio, phone, old equipment). Highpass at 40-200 Hz.

**Implementation:**
- Use `scipy.signal.iirpeak`, `iirnotch`, `butter` for filter design.
- Alternative: `torchaudio.functional.equalizer_biquad` for GPU-accelerated EQ (preferred for on-the-fly augmentation).
- Chain 1-3 EQ degradation types per sample.

### 3. Dynamic Range Compression Artifacts

**Goal:** Teach the model to restore dynamics lost to over-compression (the loudness war).

**Degradation types:**
- **Over-compression:** Threshold -20 to -6 dBFS, ratio 4:1 to 20:1, fast attack (0.1-5 ms), slow release (50-500 ms). Simulates "squashed" masters.
- **Brick-wall limiting:** Threshold at -1 to -6 dBFS, instant attack, short release. Simulates loudness-maximized masters.
- **Pumping:** Slow attack (20-100 ms) with fast release (10-50 ms) at high ratios. Creates audible gain modulation.
- **Multi-band compression abuse:** Different compression on 3-4 frequency bands with mismatched settings, creating unnatural spectral envelope modulation.

**Implementation:**
- Custom PyTorch compressor: envelope follower (peak/RMS) -> gain computer -> smoothing -> gain application. All differentiable but used only for degradation (not as a loss).
- Consider `pyloudnorm` for LUFS-targeted limiting.
- For simplicity in v1: use `ffmpeg`'s `compand` and `alimiter` filters via subprocess. Migrate to PyTorch-native later for speed.

### 4. Stereo Damage

**Goal:** Teach the model to restore proper stereo imaging.

**Degradation types:**
- **Width narrowing:** Mix stereo toward mono. `out = (1-w)*mono + w*stereo` where w is 0.0 to 0.6 (partial collapse). Full mono at w=0.
- **Phase issues:** Invert one channel's polarity (swap L/R phase). Apply random per-channel delay (0-2 ms) causing comb filtering.
- **Mid/side imbalance:** Boost mid by 3-12 dB relative to side (makes sound flat/mono-ish), or boost side by 3-12 dB (makes sound washy/unfocused).
- **Channel crosstalk:** Mix a fraction (5-30%) of L into R and vice versa.
- **Mono summing artifacts:** Sum to mono, then re-expand with synthetic stereo (e.g., Haas effect with fixed delay), creating unnatural imaging.

**Implementation:**
- All operations are simple matrix transforms on the (L, R) channels.
- Mid/side: `M = (L+R)/2, S = (L-R)/2`, scale M and S independently, reconstruct.
- Channel delay via `torch.roll` or fractional delay via sinc interpolation.

### 5. Clipping and Limiting Artifacts

**Goal:** Teach the model to restore waveform peaks lost to clipping.

**Degradation types:**
- **Hard clipping:** `torch.clamp(x, -threshold, threshold)` where threshold is 0.3 to 0.95. Simulates ADC overload.
- **Soft clipping:** `tanh(gain * x) / tanh(gain)` where gain is 1.5 to 5.0. Simulates analog saturation / tube distortion.
- **Asymmetric clipping:** Different thresholds for positive and negative peaks. Simulates damaged equipment.
- **Intersample peak clipping:** Upsample 4x, clip, downsample. Simulates DAC reconstruction clipping that occurs between samples.

**Implementation:**
- Hard/soft clipping are trivial PyTorch operations.
- Intersample clipping requires `torchaudio.functional.resample` for up/down.

### 6. Sample Rate / Bit Depth Degradation

**Goal:** Teach the model to handle reduced-quality digital audio.

**Degradation types:**
- **Sample rate reduction:** Resample to 22.05, 32, or 44.1 kHz, then back to original. Loses high-frequency content above the reduced Nyquist.
- **Bit depth reduction:** Quantize to 8-bit or 16-bit. `x_q = torch.round(x * (2^(bits-1))) / (2^(bits-1))`. Adds quantization noise.
- **Dithering artifacts:** Add shaped dither noise before requantization (triangular PDF, 1-2 LSB amplitude).
- **Anti-alias filter artifacts:** Use a deliberately poor lowpass filter (low order Butterworth) before downsampling, causing aliasing.

**Implementation:**
- `torchaudio.functional.resample` for SR changes.
- Quantization is a simple rounding operation.
- Poor anti-alias: `scipy.signal.butter(order=2)` lowpass instead of proper Kaiser.

### 7. Noise Floor Issues

**Goal:** Teach the model to reduce background noise from poor recording conditions.

**Degradation types:**
- **White noise floor:** Gaussian noise at -40 to -20 dBFS. Simulates cheap preamps, high-gain recording.
- **Pink noise (1/f):** More energy in low frequencies. Simulates HVAC, room tone. Generate via filtering white noise.
- **Hum (50/60 Hz):** Sine wave at mains frequency + harmonics (100, 150, 200, 250 Hz). Amplitude: -40 to -25 dBFS.
- **Hiss:** High-frequency noise (shaped above 4 kHz). Simulates tape hiss or preamp noise.
- **Digital noise:** Random impulses (clicks, pops) at random intervals. 1-20 per second, amplitude -30 to -10 dBFS.

**Implementation:**
- `torch.randn` for white noise, filter for colored noise.
- Sine generation for hum: `torch.sin(2 * pi * freq * t)`.
- Poisson process for impulse noise timing.

## Degradation Chaining Strategy

Not all degradations should be applied simultaneously. A realistic degradation chain mimics real-world signal paths.

### Chain Architecture

```
Clean Audio
    |
    v
[Source Degradation] -- 40% probability
    One of: SR/bit depth reduction, noise floor
    |
    v
[Processing Degradation] -- 70% probability
    One of: bad EQ, dynamic compression, clipping
    |
    v
[Codec Degradation] -- 60% probability
    One of: MP3, AAC, OGG, Opus encoding
    |
    v
[Spatial Degradation] -- 30% probability (stereo only)
    One of: width narrowing, phase issues, M/S imbalance
    |
    v
Degraded Audio
```

### Chain Rules

1. **Always apply at least one degradation.** If all random gates fail, force-apply codec degradation.
2. **Limit total degradations to 1-3 per sample.** More than 3 makes the task too ambiguous for the model to learn a clear inverse.
3. **Severity schedule:** Start training with mild degradations (higher bitrates, smaller EQ deviations, lower noise). Gradually increase severity over epochs. This is curriculum learning -- helps the model build a stable foundation before tackling extreme cases.
4. **Codec goes last in the chain** (matches real-world: master -> distribute as MP3). Exception: double-encoding, where a second codec pass follows the first.
5. **No contradictory degradations:** Do not apply both "bandwidth limiting to 12 kHz" and "excessive treble boost above 8 kHz" in the same chain.

### Severity Curriculum

| Epoch Range | Severity | Example |
|-------------|----------|---------|
| 0-20 | Mild | MP3 256-320 kbps, EQ +/- 3 dB, noise -40 dBFS |
| 20-60 | Moderate | MP3 128-192 kbps, EQ +/- 6 dB, compression 4:1, noise -30 dBFS |
| 60-120 | Severe | MP3 64-128 kbps, EQ +/- 12 dB, compression 10:1, clipping |
| 120+ | Mixed | Uniform random severity, including multi-degradation chains |

The curriculum schedule should be configurable via `configs/phase1.yaml`.

## Implementation Plan

### New Files

| File | Purpose |
|------|---------|
| `data/degradations.py` | All degradation transforms (classes, composable) |
| `data/degradation_chain.py` | Chain builder, curriculum scheduler, config interface |
| `data/dataset_phase1.py` | Phase 1 dataset: loads clean audio, applies degradation chain |
| `configs/phase1.yaml` | Phase 1 training config (loss weights, degradation params, curriculum) |

### Modifications to Existing Files

| File | Change |
|------|--------|
| `models/constants.py` | Add Phase 1 constants (codec types, bitrates, EQ ranges, noise levels) |
| `train.py` | Add `--phase` flag to select Phase 0 or Phase 1 dataset/config |

### Library Dependencies

| Library | Purpose | Install |
|---------|---------|---------|
| `ffmpeg` | Codec encoding/decoding | System package (already available) |
| `pydub` | Convenient ffmpeg wrapper | `pip install pydub` |
| `scipy` | Filter design (IIR, biquad) | Already installed |
| `pyloudnorm` | LUFS measurement for limiting | `pip install pyloudnorm` |
| `torchaudio` | EQ biquads, resampling | Already installed |

### Architecture of `data/degradations.py`

Each degradation is a callable class following a common interface:

```python
class Degradation:
    """Base class for audio degradations."""
    
    def __init__(self, severity: float = 0.5):
        """severity: 0.0 (mild) to 1.0 (extreme)"""
        self.severity = severity
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply degradation to waveform (channels, samples)."""
        raise NotImplementedError
    
    def set_severity(self, severity: float):
        """Update severity for curriculum learning."""
        self.severity = severity
```

Concrete classes: `CodecDegradation`, `BadEQDegradation`, `DynamicCompressionDegradation`, `StereoDamageDegradation`, `ClippingDegradation`, `SampleRateDegradation`, `NoiseDegradation`.

### Phase 1 Dataset Design

`data/dataset_phase1.py` wraps the existing `AudioSRDataset` loading logic but instead of creating (48kHz, 96kHz) pairs, it creates (degraded, clean) pairs:

```python
class DegradedAudioDataset(Dataset):
    def __init__(self, root_dir, target_sr, segment_length, degradation_chain):
        # Load clean audio files (same scanning as AudioSRDataset)
        # Store degradation chain
        pass
    
    def __getitem__(self, idx):
        # Load clean segment at target_sr
        clean = self._load_segment(idx)
        # Apply degradation chain
        degraded = self.degradation_chain(clean, self.target_sr)
        return degraded, clean
```

### Config Structure (`configs/phase1.yaml`)

```yaml
output:
  sample_rate: 96000
  bit_depth: 24

gan:
  # Same generator/discriminator as Phase 0
  generator: ...
  discriminator: ...
  
  training:
    batch_size: 8
    learning_rate_g: 0.0001  # Lower LR for fine-tuning from Phase 0 checkpoint
    learning_rate_d: 0.0001
    epochs: 200
    segment_length: 16384
    
    # Resume from Phase 0 checkpoint
    pretrained_checkpoint: "checkpoints/phase0/latest.pt"
    
    degradation:
      # Probability of each degradation category
      p_codec: 0.6
      p_eq: 0.4
      p_compression: 0.3
      p_stereo: 0.2
      p_clipping: 0.15
      p_sample_rate: 0.2
      p_noise: 0.25
      
      # Max degradations per chain
      max_chain_length: 3
      
      # Curriculum learning
      curriculum:
        enabled: true
        warmup_epochs: 20       # Mild severity only
        linear_ramp_epochs: 80  # Linearly increase severity
        # After warmup + ramp, full random severity
    
    mastering:
      lambda_perceptual_stft: 45.0
      lambda_stereo: 10.0
      lambda_dynamics: 5.0
      lambda_encodec: 0.01
```

## Training Strategy

### Initialization
- Load generator and discriminator weights from the best Phase 0 checkpoint.
- Reset optimizer state (new task distribution).
- Use lower learning rate (0.0001 vs 0.0002) to preserve Phase 0 knowledge.

### Loss Function Changes
- Same loss stack as Phase 0 (adversarial + feature matching + STFT + mel + mastering).
- Consider increasing `lambda_dynamics` weight since dynamic range restoration is a primary Phase 1 goal.
- Consider adding an explicit "artifact detection" loss that penalizes codec-specific spectral patterns (future research).

### Evaluation Metrics
- All Phase 0 metrics (SI-SNR, SDR, CDPAM, ViSQOL, chroma/MFCC similarity).
- Add codec-specific metrics: measure spectral distortion in the 16-20 kHz band (where MP3 artifacts are most visible).
- Add PESQ/POLQA for speech content if VCTK is in the dataset.
- A/B listening tests with reference: can a listener distinguish restored audio from the original?

### Dataset Composition
- Use the same datasets as Phase 0 (EG-IPT, MUSDB18-HQ, VCTK, MusicNet, etc.).
- Quality sampling weights may need adjustment: all tiers are equally valuable for Phase 1 since we are degrading and restoring, not upsampling.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Model "forgets" Phase 0 SR ability | Mix 20% Phase 0 (SR) pairs into Phase 1 training |
| Codec encoding is slow (disk I/O) | Pre-compute degraded versions for common codecs; use RAM disk for temp files |
| Degradation too severe -> model collapses | Curriculum learning; monitor loss curves; cap severity |
| Stereo degradation requires stereo data | Use MUSDB18-HQ and EG-IPT stereo tracks; skip stereo degradation for mono |
| ffmpeg not available on training instance | Add to `infra/setup-vastai.sh` install list (likely already present) |

## Open Questions

1. **Should we fine-tune or train from scratch?** Recommendation: fine-tune from Phase 0. The generator already understands audio structure and spectral reconstruction. Phase 1 is an extension of that capability.

2. **Should Phase 1 operate at 48kHz->96kHz or at a single sample rate?** Recommendation: operate at 96kHz throughout (degrade 96kHz audio, restore to 96kHz). This way the model can leverage both SR and restoration in Phase 2.

3. **How much data augmentation overlap with Phase 0?** Recommendation: 20% of each batch should be clean Phase 0 SR pairs (48k->96k with no degradation). This prevents catastrophic forgetting.

4. **Pre-computed vs on-the-fly degradation?** Recommendation: on-the-fly for EQ, compression, clipping, noise, stereo (fast tensor ops). Pre-computed for codec artifacts (ffmpeg I/O is slow). Store 3-5 codec variants per clean file.

## Timeline Estimate

This plan does not include time estimates per project convention. The work should be prioritized in this order:

1. Implement core degradation classes in `data/degradations.py`
2. Implement chain builder and curriculum in `data/degradation_chain.py`
3. Build Phase 1 dataset class
4. Create `configs/phase1.yaml`
5. Update `train.py` for Phase 1 mode
6. Test on small subset before full training run
