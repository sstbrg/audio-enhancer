# Audio Enhancer — Model Architecture

## Overview

```
Input (48kHz mono)
    │
    ▼
┌─────────────────────────────────────────────────┐
│                  GENERATOR                       │
│                                                  │
│  ┌──────────┐                                   │
│  │ Conv1d   │  in_channels=1 → channels=512     │
│  │ (pre)    │  kernel=7, padding=3              │
│  └────┬─────┘                                   │
│       │                                          │
│       ▼                                          │
│  ┌──────────────────┐                           │
│  │ Upsample Block   │  2x (48kHz → 96kHz)      │
│  │ ConvTranspose1d  │  channels=512→256         │
│  │ kernel=4,stride=2│                           │
│  └────┬─────────────┘                           │
│       │                                          │
│       ▼                                          │
│  ┌──────────────────────────────────┐           │
│  │ Multi-Receptive-Field Fusion     │           │
│  │                                   │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │  │ResBlock │ │ResBlock │ │ResBlock │        │
│  │  │kernel=3 │ │kernel=7 │ │kernel=11│        │
│  │  │dil=1,3,5│ │dil=1,3,5│ │dil=1,3,5│       │
│  │  └────┬────┘ └────┬────┘ └────┬────┘       │
│  │       └──────┬─────┘──────┘                  │
│  │              │ (sum & average)                │
│  └──────────────┼───────────────────┘           │
│                 │                                │
│                 ▼                                │
│  ┌──────────────────────┐                       │
│  │ High-Frequency Branch│                       │
│  │ Conv1d k=7 → ReLU    │  Learns residual HF   │
│  │ Conv1d k=7           │  content (harmonics)   │
│  └────────┬─────────────┘                       │
│           │                                      │
│           ▼                                      │
│  ┌──────────────────────┐                       │
│  │ Skip Connection      │                       │
│  │                       │                       │
│  │ input ──interpolate──►(+)◄── HF branch       │
│  │        (2x linear)    │                       │
│  └────────┬──────────────┘                       │
│           │                                      │
│           ▼                                      │
│  ┌──────────┐                                   │
│  │ Conv1d   │  channels=256 → 1                 │
│  │ (post)   │  kernel=7, padding=3              │
│  └────┬─────┘                                   │
│       │                                          │
│       ▼                                          │
│    tanh()                                        │
│                                                  │
└─────────────────────────────────────────────────┘
    │
    ▼
Output (96kHz mono)
```

## Generator Details

- **Parameters:** 11M
- **Input:** 48kHz mono waveform (1, 1, T)
- **Output:** 96kHz mono waveform (1, 1, 2T)
- **Skip connection:** Linear interpolation of input to output length + learned HF residual
- **Activation:** LeakyReLU(0.1) throughout, tanh at output
- **Normalization:** Weight norm on all Conv1d/ConvTranspose1d layers

## Discriminators

```
                    Input (96kHz waveform)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌─────────────────┐  ┌─────────────────────────────────────┐
│ Multi-Period     │  │ Multi-Scale Discriminator (MSD)     │
│ Discriminator    │  │                                     │
│ (MPD)            │  │  Scale 1: raw audio (spectral norm) │
│                  │  │  Scale 2: AvgPool 4x downsample     │
│ Period 2  ──►2D  │  │  Scale 3: AvgPool 16x downsample   │
│ Period 3  ──►2D  │  │                                     │
│ Period 5  ──►2D  │  │  Each scale:                        │
│ Period 7  ──►2D  │  │    Conv1d stack (channels growing)  │
│ Period 11 ──►2D  │  │    → real/fake score                │
│ Period 17 ──►2D  │  │    → feature maps for FM loss       │
│ Period 23 ──►2D  │  │                                     │
│                  │  │                                     │
│ Each period:     │  │                                     │
│  Reshape to 2D   │  │                                     │
│  Conv2d stack    │  │                                     │
│  → score + feats │  │                                     │
└────────┬─────────┘  └──────────────┬──────────────────────┘
         │                           │
         ▼                           ▼
    Real/Fake scores          Real/Fake scores
    Feature maps              Feature maps
```

### MPD Periods: [2, 3, 5, 7, 11, 17, 23]
Extended from HiFi-GAN default [2,3,5,7,11] for 96kHz.
Larger periods capture harmonics in human hearing range at higher sample rates.

### MSD: 3 scales
- Scale 1: Spectral norm (stabilizes on raw high-res audio)
- Scales 2-3: Weight norm on downsampled audio

## Loss Functions

```
Generator Loss = adversarial + feature_matching + spectral + mastering

┌─────────────────────────────────────────────────────────┐
│ Adversarial (from discriminators)                        │
│   L_adv = L_mpd + L_msd                (LSGAN)         │
├─────────────────────────────────────────────────────────┤
│ Feature Matching                                         │
│   L_fm = λ_fm × Σ |D_feat(real) - D_feat(fake)|        │
│   λ_fm = 2.0                                            │
├─────────────────────────────────────────────────────────┤
│ Spectral                                                 │
│   L_stft = λ_stft × MultiResSTFT(ŷ, y)                 │
│     FFT sizes: [512, 1024, 2048, 4096]                  │
│     Spectral convergence + log magnitude per resolution  │
│   λ_stft = 45.0                                         │
│                                                          │
│   L_mel = λ_mel × MelSpecLoss(ŷ, y)                    │
│     n_mels=128, n_fft=4096                              │
│   λ_mel = 45.0                                          │
├─────────────────────────────────────────────────────────┤
│ Mastering Quality                                        │
│                                                          │
│   Perceptual STFT (A-weighted, mel-scaled)  λ=45.0     │
│     → penalizes artifacts in 2-5kHz sensitivity range    │
│                                                          │
│   Stereo Image (mid/side STFT + width)      λ=10.0     │
│     → preserves stereo field (mono training: skipped)    │
│                                                          │
│   Dynamics (crest factor + K-weighted LUFS)  λ=5.0      │
│     → preserves transients and loudness                  │
│                                                          │
│   EnCodec Embedding (perceptual distance)    λ=0.01     │
│     → neural perceptual quality in Meta's space          │
│     → differentiable (gradients flow through encoder)    │
├─────────────────────────────────────────────────────────┤
│ Phase 1 only (zero weight in Phase 0):                   │
│   HF Band Loss (16-24 kHz energy diff)       λ=10.0     │
│     → penalizes codec-damaged high frequencies           │
├─────────────────────────────────────────────────────────┤
│ Validation only (not in backward pass):                  │
│   Audiobox PQ, CLAP — not differentiable                │
└─────────────────────────────────────────────────────────┘
```

## Training Optimization

- **Mixed Precision (AMP):** float16 compute, float32 master weights
- **torch.compile:** Fused kernels for generator + discriminators
- **cudnn.benchmark:** Auto-tuned convolution algorithms
- **Gradient clipping:** max_norm=5.0 on both G and D (GRAD_CLIP_MAX_NORM from constants)
- **DataLoader:** persistent_workers, prefetch_factor=4, num_workers=auto

## Validation Metrics

| Metric | Type | What it measures for SR |
|---|---|---|
| SI-SNR | Reference | Signal fidelity (dB) |
| SDR | Reference | Distortion ratio (dB) |
| CDPAM | Reference | Perceptual distance |
| Audiobox PQ | No-reference | Production quality (0-10) |
| Chroma similarity | Reference | Melody preservation |
| MFCC similarity | Reference | Timbre preservation |
| **HF energy ratio** | SR-specific | Energy above 24kHz (ref vs enh) |
| **Spectral rolloff** | SR-specific | Effective bandwidth (Hz) |

## Data Flow

```
Source audio (various SR/bit depth)
    │
    ├── 96kHz/24b (EG-IPT, VCTK)
    │     → downsample to 48kHz = LR input
    │     → original 96kHz = HR target (REAL harmonics)
    │
    ├── 48kHz/24b (GTSinger)
    │     → keep at 48kHz = LR input
    │     → resample to 96kHz = HR target (24-bit detail)
    │
    └── 44.1kHz/16b (MUSDB, MusicNet, MAESTRO, MoisesDB)
          → resample to 48kHz = LR input
          → resample to 96kHz = HR target (synthetic, for input diversity)
    │
    ▼
Generator: 48kHz → 96kHz
    │
    ▼
Discriminators judge: real or generated 96kHz?
```

## Training Status

### Phase 0 — Super-Resolution (current)

- **Goal:** Upscale 48kHz audio to 96kHz/24-bit via GAN
- **Status:** Epoch 0 complete. Losses stable: d≈4.2, g≈35. Encodec spikes resolved.
- **Checkpoint:** `gdrive:audio-enhancer-datasets/checkpoints/checkpoint_0000.pt`
- **AMP + torch.compile:** Committed and ready; not yet validated in a full training run.
- **`_unwrap_state_dict`:** Done — wired into all checkpoint save paths in `train.py`.

### Phase 1 — Degradation Restoration (implementation in progress)

- **Goal:** Restore lossy-compressed / poorly mastered audio; simultaneously upscale to 96kHz
- **Architecture:** Same 48kHz→96kHz HiFi-GAN generator. Degradation applied at 96kHz, downsampled to 48kHz for model input; clean 96kHz is the target.
- **Fine-tuning:** From Phase 0 checkpoint; optimizer reset; lower LR (0.0001); 5-epoch linear warmup.
- **Mixed batches:** 20% Phase 0 SR pairs per batch (prevents catastrophic forgetting).
- **Loss changes:** `lambda_dynamics=15.0` (3× Phase 0); new `HighFrequencyBandLoss` (16–24 kHz, λ=10.0).
- **Config:** `configs/phase1.yaml`. Recipe: `docs/phase1_finetuning_recipe.md`.
- **Status:** Core files done (degradations.py, degradation_chain.py, dataset_phase1.py, precompute_codecs.py, configs/phase1.yaml, train.py --phase flag). First training run pending.

### Phase 2 — Full Pipeline (planned)

- **Goal:** End-to-end: any audio in → studio-quality out
- **Pipeline:** Apollo → AudioSR → custom GAN (fine-tuned)
- **Status:** Deferred until Phase 1 complete.
