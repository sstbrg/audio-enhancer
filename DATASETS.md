# Audio Enhancement Model — Training Strategy & Dataset Catalog

## Goal

Train a GAN-based audio restoration/mastering model that:
1. Restores codec-damaged audio (MP3/AAC/OGG → lossless quality)
2. Improves poorly mastered music (bad EQ, crushed dynamics, narrow stereo, noise)
3. Upscales sample rate and bit depth (48 kHz → 192 kHz, 16-bit → 24-bit)
4. Leaves already-good audio untouched

Target deployment: desktop, cloud.

---

## Part I — Dataset Catalog

Comprehensive catalog of open/free datasets suitable for training.
Research date: 2026-04-02.

---

### Tier 1 — Best Candidates (High-Quality Lossless Music, Widely Used)

#### 1. MUSDB18-HQ

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/3338373 |
| **License** | Non-commercial / educational use only (custom) |
| **Sample Rate** | 44.1 kHz |
| **Bit Depth** | 16-bit PCM |
| **Format** | Uncompressed stereo WAV |
| **Size** | ~22.7 GB download / 21.1 GB content |
| **Duration** | ~10 hours |
| **Content** | 150 full-length songs, various genres. Each track has mixture + 4 stems (drums, bass, vocals, other) |
| **Paired LQ/HQ** | No — single quality, but you can downsample to create pairs |
| **How to Download** | Request access on Zenodo; also `pip install musdb` for Python tools |
| **Notes** | Used by AERO and Apollo for audio super-resolution training. The go-to music dataset for source separation research. |

#### 2. MoisesDB

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/10265363 / https://github.com/moises-ai/moises-db |
| **HuggingFace** | `wearemusicai/moisesdb` |
| **License** | CC BY-NC-SA 4.0 |
| **Sample Rate** | 44.1 kHz (configurable via Python API) |
| **Bit Depth** | Not publicly specified (likely 16 or 24-bit) |
| **Format** | WAV |
| **Size** | ~25 GB |
| **Duration** | 14 hours 25 minutes |
| **Content** | 240 unreleased songs from 47 artists, 12 genres, with stems beyond the usual 4 (detailed instrument separation) |
| **How to Download** | `pip install moises-db`, set `MOISESDB_PATH`, or HuggingFace |
| **Notes** | Used alongside MUSDB18-HQ for Apollo training. Richer stem decomposition than MUSDB18. |

#### 3. SonicMaster Dataset (2025 — THE mastering dataset)

| Field | Value |
|---|---|
| **URL** | https://arxiv.org/abs/2508.03448 (paper) |
| **License** | Research (built on Jamendo CC tracks) |
| **Sample Rate** | 44.1 kHz |
| **Format** | WAV |
| **Size** | ~175K audio pairs |
| **Duration** | 25K clips × 7 degradations = 175K pairs |
| **Content** | High-fidelity Jamendo segments spanning 10 genres, each paired with 7 degraded versions using 19 degradation functions |
| **Paired LQ/HQ** | **YES — 175K paired clips with degradation labels** |
| **Degradation types** | EQ (brightness, darkness, boominess, muddiness, mic coloration), dynamics (compression, punch/transient), reverb (small/big/mixed room, IRs), amplitude (clipping, noise), stereo (width collapse) |
| **Notes** | **The single most important dataset for your use case.** Pre-built good→bad pairs with natural language descriptions. Simulates exactly the amateur production problems you want to fix. |

#### 4. MAESTRO (Piano)

| Field | Value |
|---|---|
| **URL** | https://magenta.withgoogle.com/datasets/maestro |
| **License** | CC BY-NC-SA 4.0 |
| **Sample Rate** | 44.1–48 kHz (varies per recording) |
| **Bit Depth** | 16-bit PCM |
| **Format** | Stereo WAV + aligned MIDI |
| **Size** | V3.0.0: 101 GB compressed / 120 GB uncompressed |
| **Duration** | ~199 hours (1,276 performances) |
| **Content** | Virtuosic piano performances from International Piano-e-Competition (Yamaha Disklavier) |
| **How to Download** | `https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip` |
| **Notes** | Massive. ~200 hours of high-quality piano. Excellent for piano-focused training. |

#### 5. Slakh2100 (Synthesized Lakh MIDI)

| Field | Value |
|---|---|
| **URL** | http://www.slakh.com / https://zenodo.org/records/4599666 |
| **License** | CC BY 4.0 |
| **Sample Rate** | 44.1 kHz |
| **Bit Depth** | 16-bit |
| **Format** | FLAC (converts to ~500 GB WAV) |
| **Size** | 105 GB download (FLAC); ~500 GB as WAV |
| **Duration** | 145 hours |
| **Content** | 2,100 synthesized multitrack songs. 187 instrument patches in 34 classes |
| **Notes** | Enormous and diverse. MIDI available — can re-render at 96/192 kHz using better virtual instruments. |

#### 6. MedleyDB / MedleyDB 2.0

| Field | Value |
|---|---|
| **URL** | https://medleydb.weebly.com / https://zenodo.org/records/1715175 |
| **License** | CC BY-NC |
| **Sample Rate** | 44.1 kHz |
| **Bit Depth** | 16-bit |
| **Format** | Stereo WAV (mix + stems) |
| **Duration** | ~7 hours (196 multitracks) |
| **Notes** | High-quality professional recordings. Used in bandwidth extension research. |

---

### Tier 2 — High-Quality Instrument/Voice Datasets

#### 7. EG-IPT (Electric Guitar, 96 kHz!)

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/15205644 |
| **License** | CC BY 4.0 |
| **Sample Rate** | **96 kHz** |
| **Bit Depth** | **24-bit** |
| **Format** | WAV, monophonic |
| **Size** | 23.8 GB / 29.77 GB uncompressed |
| **Duration** | 28 hours 23 minutes |
| **Content** | 52,320 monophonic electric guitar files. 19 techniques. 6 audio sources (5 mics + DI) |
| **Paired LQ/HQ** | No |
| **Notes** | **One of the few datasets truly at hi-res (96 kHz/24-bit).** Single-note performances only. Published 2025. |

#### 8. EGFxSet (Electric Guitar Effects)

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/7044411 |
| **License** | CC BY 4.0 |
| **Sample Rate** | 48 kHz / 24-bit |
| **Duration** | 12 hours 28 minutes |
| **Content** | Clean and effects-processed Stratocaster tones. 12 real hardware effects |
| **Paired LQ/HQ** | Has clean/effected pairs |

#### 9. GTSinger (Singing Voice, 48 kHz/24-bit)

| Field | Value |
|---|---|
| **URL** | https://github.com/AaronZ345/GTSinger |
| **HuggingFace** | `GTSinger/GTSinger` |
| **License** | CC BY-NC-SA 4.0 |
| **Sample Rate** | **48 kHz / 24-bit** |
| **Duration** | 80.59 hours |
| **Content** | 20 singers, 9 languages, 4 vocal ranges, 6 singing techniques |
| **Notes** | NeurIPS 2024 Spotlight. Studio-quality. Largest recorded singing dataset. |

#### 10. VocalSet

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/1193957 |
| **License** | CC BY 4.0 |
| **Sample Rate** | 44.1 kHz / 16-bit |
| **Duration** | 10.1 hours |
| **Content** | 3,560 recordings, 20 professional singers, 17 techniques |

#### 11. NSynth (Individual Notes)

| Field | Value |
|---|---|
| **URL** | https://magenta.withgoogle.com/datasets/nsynth |
| **License** | CC BY 4.0 |
| **Sample Rate** | 16 kHz (LOW) |
| **Duration** | ~340 hours (305,979 notes × 4s) |
| **Notes** | Low sample rate limits hi-res use. But massive scale and instrument diversity. |

#### 12. MAPS (Piano)

| Field | Value |
|---|---|
| **URL** | https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/ |
| **License** | CC (academic) |
| **Sample Rate** | 44.1 kHz / 16-bit |
| **Duration** | ~65 hours |
| **Content** | MIDI-aligned piano sounds. Real and synthesized recordings. |

---

### Tier 3 — General Music (Compressed / Lower Quality)

#### 13. FMA (Free Music Archive)

| Field | Value |
|---|---|
| **URL** | https://github.com/mdeff/fma |
| **License** | Various CC per track |
| **Format** | **MP3 only** |
| **Size** | Full: 879 GB (106K tracks, untrimmed) |
| **Duration** | ~343 days (!) |
| **Content** | 161 genres, extremely diverse |
| **Notes** | MASSIVE but MP3-only. Best for "restore MP3 artifacts" training and augmenting diversity. |

#### 14. MTG-Jamendo

| Field | Value |
|---|---|
| **URL** | https://github.com/MTG/mtg-jamendo-dataset |
| **License** | CC various per track |
| **Format** | **MP3 at 320 kbps** |
| **Duration** | 55,000+ full tracks |
| **Notes** | Large-scale but MP3 only. Score with Audiobox Aesthetics, keep top 20% as "good production" reference. |

#### 15. MusicNet (Classical)

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/5120004 |
| **License** | CC |
| **Sample Rate** | 44.1 kHz / 16-bit WAV |
| **Duration** | ~34 hours (330 recordings) |
| **Content** | Classical chamber music. 10 composers, 11 instruments. 1M+ temporal note labels. |

---

### Tier 4 — Speech Datasets (Standard SR Benchmarks)

#### 16. VCTK 96 kHz (TRUE HI-RES!)

| Field | Value |
|---|---|
| **URL** | https://datashare.ed.ac.uk/handle/10283/2774 |
| **License** | CC BY 4.0 |
| **Sample Rate** | **96 kHz / 24-bit** |
| **Duration** | ~44 hours |
| **Paired LQ/HQ** | **YES — the 48 kHz version is the downsampled LQ version of this** |
| **Notes** | TRUE HI-RES PAIRED DATA. Perfect for 48→96 kHz SR training validation. Speech only but methodology transfers. |

#### 17. VCTK 48 kHz

| Field | Value |
|---|---|
| **URL** | https://datashare.ed.ac.uk/handle/10283/3443 |
| **License** | CC BY 4.0 |
| **Sample Rate** | 48 kHz / 16-bit |
| **Duration** | ~44 hours |
| **Notes** | THE standard benchmark for audio SR papers (AERO, AudioSR, AudioLBM all use it). |

#### 18. DAPS (Device and Produced Speech)

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/4660670 |
| **License** | CC BY-NC 4.0 |
| **Paired LQ/HQ** | **YES — aligned professional vs consumer device recordings** |
| **Duration** | ~4.5 hours per version × 15 versions |

---

### Tier 5 — Multitrack / Production

#### 19. Cambridge-MT Mixing Secrets

| Field | Value |
|---|---|
| **URL** | https://cambridge-mt.com/ms3/mtk/ |
| **License** | Educational use |
| **Size** | ~532 GB (WAV) |
| **Content** | 500+ real multitrack recording sessions. Raw tracks without effects. Some 24-bit. |
| **Notes** | Foundation for DSD100 and MUSDB18. Massive real-world multitrack audio. |

#### 20. AAM (Artificial Audio Multitracks)

| Field | Value |
|---|---|
| **URL** | https://zenodo.org/records/5794629 |
| **License** | CC BY 4.0 |
| **Size** | ~210 GB (mixes ~44 GB, multitracks ~165 GB) |
| **Content** | 3,000 algorithmically composed tracks with MIDI. Re-renderable at any sample rate. |

---

### Tier 6 — MIDI-Based (Re-Renderable at 192 kHz)

#### 21. Lakh MIDI Dataset

| Field | Value |
|---|---|
| **URL** | https://colinraffel.com/projects/lmd/ |
| **License** | Open |
| **Content** | ~170,000 MIDI files |
| **Notes** | No audio — render at 192 kHz using Kontakt/Pianoteq/REAPER for native hi-res ground truth. |

---

### Quick Reference: Priority Download Order

| # | Dataset | SR | Hours | Key value | Download |
|---|---|---|---|---|---|
| 1 | **SonicMaster** | 44.1k | 175K pairs | Pre-built mastering degradation pairs | arxiv paper / check release |
| 2 | **EG-IPT** | **96k/24b** | 28h | Only true 96kHz music | Zenodo direct |
| 3 | **VCTK 96kHz** | **96k/24b** | 44h | True hi-res paired data (speech) | Edinburgh DataShare |
| 4 | **MUSDB18-HQ + MoisesDB** | 44.1k | 24h | What Apollo/AERO trained on | Zenodo + HuggingFace |
| 5 | **GTSinger** | **48k/24b** | 80h | Studio vocals, massive | HuggingFace |
| 6 | **MAESTRO** | 44-48k | 199h | Enormous scale, piano | Google direct |
| 7 | **Slakh2100** | 44.1k | 145h | Multi-instrument, MIDI re-renderable | Zenodo |
| 8 | **FMA Full** | 44.1k MP3 | 8000h+ | Scale + codec artifact training | GitHub |
| 9 | **MedleyDB + MusicNet** | 44.1k | 41h | Professional recordings | Zenodo |

**Total lossless music available**: ~600+ hours
**Total including speech**: ~700+ hours
**MIDI re-renderable at 192 kHz**: ~150+ hours

---

## Part II — Degradation Pipeline

Start with clean, well-mastered audio. Apply random chains of degradations to create realistic "bad production" inputs. Each training pair = (degraded, clean).

### Degradation Categories

**A. Dynamic range destruction (loudness war)**
- Hard limiting: peak limiter at -0.1 dBFS, drive input 6–18 dB
- Over-compression: fast attack (0.1–1ms), slow release (100–500ms), ratio 8:1–20:1, threshold -20 to -6 dB
- Multiband squashing: independent compression per band, random thresholds
- Brick-wall clipping: hard clip at -3 to -0.5 dB

**B. Spectral / EQ damage**
- Boominess: +6–12 dB shelf or bell at 100–250 Hz
- Muddiness: +4–8 dB broad boost at 200–500 Hz
- Harshness: +4–10 dB narrow peak at 2–5 kHz
- No air: low-pass at 12–16 kHz
- Tinny: high-pass at 150–300 Hz
- Random parametric EQ: 3–5 random bells, ±3–12 dB, Q 0.5–4.0
- Microphone coloration: convolve with measured mic IRs

**C. Stereo field damage**
- Mono collapse: width 0.0–0.5
- Excessive widening: side boost 6–15 dB (phasey, thin)
- Channel imbalance: 1–6 dB offset between L/R
- Phase issues: 0–2ms delay on one channel

**D. Noise and artifacts**
- Tape hiss: shaped noise at -40 to -20 dBFS
- Preamp hum: 50/60 Hz + harmonics at -40 to -25 dBFS
- Quantization noise: reduce to 8–12 bit, back to 16/24
- Codec artifacts: MP3 32–128k, OGG 64–128k, AAC 64–128k (encode→decode)

**E. Reverb / room problems**
- Boxy room: RT60 0.2–0.5s
- Bathroom: highly reflective, metallic
- Too wet: RT60 2–5s, 40–80% wet

**F. Amplitude issues**
- Too quiet: normalize to -20 to -12 dBFS
- DC offset: 0.01–0.05 shift
- Intersample peaks

### Chain Strategy

Apply **2–5 degradations** per sample from different categories:

```python
probability_per_category = {
    "dynamics": 0.6,    # most common real-world issue
    "eq": 0.7,          # almost always present in bad masters
    "stereo": 0.3,
    "noise": 0.3,
    "reverb": 0.25,
    "amplitude": 0.2,
    "codec": 0.4,       # the streaming use case
}
```

### Implementation Tools

- **pedalboard** (Spotify): compressor, limiter, reverb, EQ, clipping — fast, GPU-friendly
- **ffmpeg**: codec round-trips (MP3/OGG/AAC encode→decode)
- **pyroomacoustics**: room simulation
- **scipy.signal**: filters, noise generation
- **torchaudio.functional**: on-the-fly transforms in the dataloader

---

## Part III — Loss Functions

### Generator Losses (combined, weighted)

| Loss | Weight | Formula | Purpose |
|---|---|---|---|
| **L1 time-domain** | 1.0 | \|y_pred - y_clean\| | Anchors output, prevents hallucination |
| **Multi-res STFT** | 1.0 | Σ [spectral_convergence + log_mag_L1] over fft_sizes [512, 1024, 2048, 4096] | Frequency content at multiple scales. **Most important loss.** |
| **Mel-spectrogram** | 0.5 | \|mel(y_pred) - mel(y_clean)\| with n_mels=128, fmax=22050 | Perceptually weighted |
| **Adversarial** | λ_adv (ramps 0.1→0.3) | -E[D(y_pred)] (LSGAN or hinge) | Teaches generator what "sounds real" |
| **Feature matching** | 2.0 | Σ_layers \|D_features(y_pred) - D_features(y_clean)\| | Stabilizes GAN training. Does most heavy lifting. |

**Total**: `L_G = 1.0*L_time + 1.0*L_stft + 0.5*L_mel + λ_adv*L_adv + 2.0*L_fm`

### Discriminator

Multi-resolution STFT discriminator (from Apollo/EnCodec/Gull):
- Window sizes: [32, 64, 128, 256, 512, 1024, 2048]
- LSGAN: `L_D = E[(D(y_clean) - 1)²] + E[D(y_pred)²]`

---

## Part IV — Training Schedule

### Phase 1: Reconstruction pre-training (epochs 1–20)

- **No GAN** — generator only with L1 + STFT + mel
- LR: 1e-3, AdamW, weight decay 0.01
- Batch size: 16–32 (RTX 5090 with 3s clips)
- Clip length: 3 seconds (132,300 samples at 44.1 kHz)
- Degradation severity: 50% of max
- **Goal**: stable input-output mapping without mode collapse risk

### Phase 2: GAN fine-tuning (epochs 21–60)

- Enable discriminator, ramp λ_adv from 0.1 to 0.3
- Generator LR: 1e-4, Discriminator LR: 1e-5 (10× lower)
- LR decay: 0.98 every 2 epochs
- Gradient clipping: max norm 5.0
- Degradation: full severity range
- **Goal**: perceptual quality, natural-sounding output

### Phase 3: Hard negative mining (epochs 61–80)

- Increase severe degradations (low bitrate + heavy compression + bad EQ)
- Add real-world low-DR recordings if available
- Freeze discriminator every 3rd epoch
- LR: 5e-5 both networks
- **Goal**: handle worst-case inputs

### Phase 4: Knowledge distillation for mobile (epochs 81–100)

- Freeze Apollo-style teacher
- Train small U-Net (~500K params) to match teacher outputs
- Loss = |U-Net(x) - Apollo(x)|
- No GAN needed
- LR: 3e-4
- **Goal**: fast mobile model that approximates big model quality

### Early Stopping

Validation STFT loss, stop if no improvement for 20 epochs. Track ViSQOL every 5 epochs.

---

## Part V — Data Loading Strategy

### Chunk Sampling
- Random 3-second chunks from random tracks
- 50% overlap (augmentation via position)
- Reject silent chunks (RMS < -60 dBFS)

### On-the-Fly Degradation

```python
class AudioPairDataset:
    def __getitem__(self, idx):
        # 1. Load random 3s chunk from clean dataset
        clean = load_random_chunk(self.tracks[idx], duration=3.0)
        
        # 2. Apply random degradation chain
        degraded = self.degradation_pipeline(clean)
        
        # 3. Normalize both to [-1, 1]
        clean = clean / (clean.abs().max() + 1e-8)
        degraded = degraded / (degraded.abs().max() + 1e-8)
        
        return degraded, clean  # (2, 132300), (2, 132300)
```

Generate degradations on-the-fly — infinite variety, model never sees exact same degradation twice.

### Augmentation
- Random gain: ±6 dB
- Random channel swap (L↔R): 50%
- Random polarity inversion: 10%
- Random time offset within track

### Batch Composition — CRITICAL

| Proportion | Type | Why |
|---|---|---|
| 40% | Codec degradation (MP3/AAC/OGG) | Your streaming use case |
| 30% | Production quality (EQ + dynamics) | Bad mastering restoration |
| 20% | Combined (codec + production) | Worst case |
| **10%** | **Clean → clean (identity)** | **Prevents over-processing good audio** |

The 10% identity pairs are the single most important trick. Without them, the model aggressively processes everything including tracks that already sound great. With them, it learns "if it's already good, leave it alone."

---

## Part VI — Validation & Quality Tracking

### Automated Metrics (every epoch)
- Multi-resolution STFT loss on validation set
- SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
- SDR (Signal-to-Distortion Ratio)

### Perceptual Metrics (every 5 epochs)

| Metric | Type | What it measures |
|---|---|---|
| **PAM** | No-reference | Perceptual clarity via CLAP antonym prompts (0–1) |
| **Audiobox Aesthetics** | No-reference | Production Quality, Complexity, Enjoyment (0–10) |
| **MuQ-Eval** | No-reference | Music-specific MOS, excellent at MP3 artifact detection |
| **ViSQOL** | Reference-based | Google's perceptual MOS (1–5), used by Apollo paper |
| **CDPAM** | Reference-based | Contrastive perceptual distance (lower = better) |

### Listening Tests (every 10 epochs)

Fixed set of 10 degraded clips. Enhance at each checkpoint. Listen. Include:
- 2× heavy MP3 compression (64 kbps)
- 2× loudness war (DR3 style crushing)
- 2× bad EQ + room problems
- 2× combined multi-degradation
- 2× already-good recordings (should pass through unchanged)

Save enhanced outputs at each checkpoint for A/B comparison across training.

---

## Part VII — Hardware & Speed Estimates

### RTX 5090 (your training machine)

| Setting | Estimate |
|---|---|
| Batch size (3s clips, Apollo-style 16.5M params) | 16–32 |
| Training speed | ~2–4 batches/sec |
| Phase 1 (20 epochs × 1000 steps) | ~3–5 hours |
| Phase 2 (40 epochs) | ~8–12 hours |
| Phase 3 (20 epochs) | ~3–5 hours |
| Full training | **~1 day** |
| VRAM usage | ~18–22 GB |

### Deployment Inference

| Target | Format | Size | Per 1s chunk |
|---|---|---|---|
| Desktop/Cloud (RTX) | PyTorch native | ~67 MB | <5 ms |
| Android S23 Ultra (NPU) | TorchScript .ptl | ~67 MB | ~20–50 ms |
| Android S23 Ultra (NNAPI) | ONNX INT8 | ~17 MB | ~5–15 ms |
| Raspberry Pi 5 (CPU) | ONNX INT8 | ~17 MB | ~200–500 ms |
| Pi 5 + Hailo-8L | Hailo HEF | ~17 MB | ~30–80 ms |

---

## Part VIII — Key Gotchas

1. **Identity preservation**: Without clean→clean pairs in training (10% of batches), the model over-processes good audio. This is the #1 mistake.

2. **Phase coherence**: STFT-based models introduce phase artifacts at chunk boundaries. Use overlap-add with 100ms+ crossfade during inference.

3. **Stereo handling**: Process both channels jointly (not independently). Independent processing destroys stereo imaging.

4. **Normalization**: Normalize to [-1, 1] during training. At inference, preserve original gain — normalize in, process, scale back out.

5. **Codec-specific artifacts**: MP3, OGG, and AAC produce different spectral holes. Train on all three. AAC (what Tidal uses) has its own pattern.

6. **The "better than original" trap**: The model might add pleasant harmonics or width that wasn't in the original. Sounds good subjectively but is technically distortion. More L1 weight = faithful restoration, more adversarial weight = creative enhancement. Choose deliberately.

7. **Discriminator collapse**: If D_loss goes to 0, generator can't learn. Reduce discriminator LR or skip D updates periodically.

8. **Pre-compute codec degradations**: On-the-fly ffmpeg encode→decode is slow and bottlenecks the dataloader. Pre-compute codec variants, do EQ/dynamics/stereo on-the-fly.

9. **Dynamic Range Database as weak labels**: dr.loudness-war.info has DR scores for ~200K albums. DR 12+ = excellent mastering, DR < 4 = crushed. Use as quality proxy for real-world tracks.

10. **ONNX export with STFT**: Apollo uses `torch.stft` with complex tensors, which ONNX doesn't support. Use TorchScript (.ptl) for mobile, or replace STFT with Conv1d-based implementation for ONNX export.

---

## References

- AERO (ICASSP 2023): Trained on VCTK + MUSDB18-HQ
- Apollo (ICASSP 2025): Trained on MUSDB18-HQ + MoisesDB
- AudioSR (ICASSP 2024): Evaluated on VCTK, ESC-50, AudioStock
- AudioLBM (NeurIPS 2025): First any-to-192kHz SR, evaluated on VCTK, ESC-50, Song-Describer
- SonicMaster (2025): 19 degradation types, text-conditioned restoration, Jamendo-based
- FLowHigh (ICASSP 2025): Single-step flow matching SR to 48kHz
- AEROMamba (LAMIR 2024): Mamba-based AERO variant, 14× faster inference