# Audio Enhancer — Architecture Reference

## System Overview

The audio enhancer is a GAN-based audio super-resolution system. The primary goal is upscaling 48kHz audio to 96kHz/24-bit with hi-fi quality. The pipeline is multi-stage:

```
Any audio input
       │
       ▼
[Stage 1] Apollo — Restore lossy compression artifacts (optional)
       │
       ▼
[Stage 2] AudioSR — Neural bandwidth extension to 48kHz (optional)
       │
       ▼
[Stage 3] Custom GAN — Upsample 48kHz → 96kHz with harmonic generation
       │
       ▼
96kHz / 24-bit WAV output
```

Stages 1 and 2 are third-party models; Stage 3 is the custom HiFi-GAN trained in this project.

---

## Generator Architecture

**File:** `models/generator.py`  
**Parameters:** ~11M  
**Input:** mono waveform `(batch, 1, T)` at 48kHz  
**Output:** mono waveform `(batch, 1, 2T)` at 96kHz  

```
Input (48kHz mono)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                      GENERATOR                           │
│                                                          │
│  conv_pre: Conv1d(1 → 512, kernel=7, padding=3)          │
│       │                                                  │
│       ▼                                                  │
│  upsample block (2x): ConvTranspose1d(512 → 256,         │
│                        kernel=4, stride=2)               │
│       │                                                  │
│       ▼                                                  │
│  ┌────────────────────────────────────────┐             │
│  │    Multi-Receptive-Field Fusion        │             │
│  │                                        │             │
│  │  ResBlock(k=3, dil=[1,3,5])            │             │
│  │  ResBlock(k=7, dil=[1,3,5])    ──sum── │             │
│  │  ResBlock(k=11, dil=[1,3,5])    ÷3     │             │
│  └────────────────────────────────────────┘             │
│       │                                                  │
│       ▼                                                  │
│  hf_branch: Conv1d(k=15) → LeakyReLU → Conv1d(k=3)     │
│       │                                                  │
│       ▼ (x = x + hf_branch(x))                          │
│                                                          │
│  conv_post: Conv1d(256 → 1, kernel=7) → tanh()          │
│                                                          │
│  skip: F.interpolate(input, scale=2, mode="linear")      │
│                                                          │
│  output = tanh_output + skip  (residual learning)        │
└─────────────────────────────────────────────────────────┘
       │
       ▼
Output (96kHz mono)
```

### ResBlock

Each ResBlock applies pairs of dilated Conv1d layers with a residual connection:

```
x → LeakyReLU → conv1(dilation=d) → LeakyReLU → conv2(dilation=1) → + x
```

This is repeated for each dilation in the list `[1, 3, 5]`, stacking temporal context.

### Key design decisions

- **Single 2x upsample:** One ConvTranspose1d stage from 48kHz to 96kHz. Simpler than the multi-stage upsampling in original HiFi-GAN (which targeted 22kHz).
- **Residual (skip) learning:** The generator predicts the high-frequency *residual* on top of a linear-interpolated baseline. This means the model only has to learn what is *different* from a simple upsampler, not the entire waveform.
- **High-frequency branch:** A parallel Conv1d branch (`hf_branch`) explicitly targets harmonic generation above 24kHz, added before the output projection.
- **Weight norm:** Applied to all Conv1d and ConvTranspose1d layers during training; removed at inference via `remove_weight_norm()`.
- **Activation:** LeakyReLU(slope=0.1) throughout, tanh at output to keep waveform in `[-1, 1]`.

---

## Discriminator Architecture

**File:** `models/discriminator.py`

Two discriminators run in parallel. Both return real/fake logits and feature maps used in the feature matching loss.

### Multi-Period Discriminator (MPD)

7 independent sub-discriminators, each operating at a different period:

```
Periods: [2, 3, 5, 7, 11, 17, 23]
```

Each sub-discriminator reshapes the 1D waveform into a 2D tensor `(batch, 1, T//period, period)` then applies 5 × Conv2d layers followed by a final scoring Conv2d. This captures periodic harmonic structure at multiple timescales.

Periods extend HiFi-GAN's original `[2,3,5,7,11]` with 17 and 23 to better cover the extended frequency range at 96kHz.

### Multi-Scale Discriminator (MSD)

3 sub-discriminators operating on the waveform at different temporal resolutions:

```
Scale 1: raw audio — spectral norm (stabilizes training on high-res audio)
Scale 2: AvgPool(4, stride=2) downsampled
Scale 3: AvgPool(4, stride=2) applied twice (4x downsampled)
```

Each scale applies 7 × Conv1d layers with growing channels (`128 → 128 → 256 → 512 → 1024 → 1024 → 1024`) then a final scoring Conv1d.

---

## Loss Function Hierarchy

**Files:** `models/losses.py`, `models/mastering_losses.py`

```
Generator Loss = L_adv + λ_fm × L_fm + λ_stft × L_stft + λ_mel × L_mel + L_mastering

Discriminator Loss = L_disc (LSGAN: real→1, fake→0)
```

### Adversarial losses (LSGAN)

```
L_disc = Σ [(1 - D(real))² + D(fake)²]   (discriminator update)
L_adv  = Σ [(1 - D(fake))²]              (generator update)
```

Applied independently to MPD and MSD outputs, then summed.

### Feature matching loss

L1 distance between discriminator intermediate feature maps of real and generated audio:

```
L_fm = Σ_layers |D_feat(real) - D_feat(fake)|₁
λ_fm = 2.0
```

### Multi-resolution STFT loss

4 STFT resolutions: FFT sizes `[512, 1024, 2048, 4096]` with matching hop/window sizes. Each resolution computes spectral convergence + log magnitude L1 loss.

```
λ_stft = 45.0
```

All STFT computations run in float32 (autocast disabled) to avoid float16 overflow.

### Mel spectrogram loss

L1 on log mel spectrograms (n_mels=128, n_fft=4096). Runs in float32.

```
λ_mel = 45.0
```

### Mastering losses (perceptual)

| Loss | Lambda | Description |
|------|--------|-------------|
| `PerceptualSTFTLoss` | 45.0 | A-weighted mel STFT via auraloss; penalizes artifacts in the 2-5kHz hearing sensitivity band |
| `StereoImageLoss` | 10.0 | Mid/side STFT + stereo width ratio matching; skipped for mono |
| `DynamicRangeLoss` | 5.0 | Per-frame crest factor + K-weighted LUFS matching; prevents dynamics compression |
| `EncodecEmbeddingLoss` | 0.01 | MSE in Meta EnCodec embedding space; differentiable perceptual loss |

Validation-only (not differentiable, no gradient flow):

| Loss | Description |
|------|-------------|
| `AudioboxPQLoss` | Meta Audiobox Production Quality predictor; used in validation only |
| `CLAPEmbeddingLoss` | LAION-CLAP semantic similarity; validation only |

The `MasteringLoss` wrapper skips any component that returns NaN or inf to prevent training instability.

---

## Training Pipeline

**File:** `train.py`

### Initialization order

1. Load config YAML
2. Create dataset and DataLoader
3. Instantiate Generator, MPD, MSD, loss functions
4. **Load checkpoint before `torch.compile`** (state dict keys change after compilation)
5. Apply `torch.compile` to Generator, MPD, MSD
6. Run training loop

### Per-step flow

```
for lr_audio, hr_audio in loader:
    lr_audio, hr_audio → GPU (non_blocking=True)

    # Forward (AMP autocast)
    hr_hat = generator(lr_audio)

    # Discriminator update
    loss_d = disc_loss(mpd(real, fake.detach())) + disc_loss(msd(real, fake.detach()))
    scaler_d.scale(loss_d).backward()
    clip_grad_norm_(D params, max_norm=10.0)
    scaler_d.step(optim_d) ; scaler_d.update()

    # Generator update (re-run discriminators for feature maps)
    loss_g = adv + fm + stft + mel + mastering
    scaler_g.scale(loss_g).backward()
    clip_grad_norm_(G params, max_norm=10.0)
    scaler_g.step(optim_g) ; scaler_g.update()
```

### Optimizations

| Optimization | Detail |
|---|---|
| AMP (mixed precision) | float16 forward/backward, float32 master weights, GradScaler for both G and D |
| torch.compile | Fused convolution kernels; applied after checkpoint load |
| cudnn.benchmark | Auto-selects fastest conv algorithms for input shape |
| Gradient clipping | max_norm=10.0 on both G and D |
| DataLoader | persistent_workers, prefetch_factor=4, pin_memory=True |
| Quality sampling | WeightedRandomSampler: 70% hi-res, 20% mid-res, 10% standard |

### Checkpoints

Saved at the end of every epoch (phase0 config: `checkpoint_interval=1`) as `checkpoint_NNNN.pt` and symlinked to `latest.pt`. Checkpoint keys: generator, mpd, msd, optim_g, optim_d, sched_g, sched_d, scaler_g, scaler_d, config.

### LR scheduling

`ExponentialLR(gamma=0.999)` applied to both optimizers after every epoch.

---

## Inference Pipeline

**File:** `enhance.py`  
**Class:** `AudioEnhancer`

```
Input file (any SR, any format)
       │
       ├── [Stage 1] Apollo (optional, --no-apollo to skip)
       │     Restores MP3/AAC lossy artifacts
       │     Resamples input to 44.1kHz (Apollo requirement)
       │
       ├── [Stage 2] AudioSR (optional, skipped if sr >= 48kHz)
       │     Neural bandwidth extension → 48kHz
       │
       ├── [Stage 3] GAN (requires --gan_checkpoint)
       │     48kHz → 96kHz via generator
       │     Falls back to Kaiser resampling if no checkpoint
       │
       └── Save as 24-bit WAV (or FLAC with --format flac)
```

For files longer than 10s (at 48kHz), the GAN processes in overlapping 10s chunks with linear crossfade at boundaries (`overlap=4800` samples).

Each channel is processed separately (generator is mono).

---

## Dataset Pipeline

**File:** `data/dataset.py`  
**Class:** `AudioSRDataset`

### Quality tiers

```
>= 96kHz (hi-res):   EG-IPT, VCTK
   → HR = original at 96kHz (real harmonics above 24kHz)
   → LR = downsampled to 48kHz

>= 48kHz (mid-res):  GTSinger
   → HR = resampled to 96kHz (24-bit detail preserved)
   → LR = downsampled to 48kHz

<= 44.1kHz (standard): MUSDB18-HQ, MusicNet, MAESTRO, MoisesDB
   → LR = resampled to 48kHz
   → HR = resampled to 96kHz (synthetic target for input diversity)
```

The quality distribution is controlled by `WeightedRandomSampler`. Phase 0 uses weights `7:2:1` (hires:midres:standard) so the model spends most training time on real hi-res sources.

### CD-quality degradation (optional)

When `degradation_prob > 0`, the LR input is additionally degraded:

```
LR at 48kHz
   → downsample to 44.1kHz
   → 16-bit quantisation with TPDF dither
   → resample back to 48kHz
```

The HR target is left clean, so the model learns to recover bandwidth and dynamic range lost through CD encoding.

### Segment extraction

Each training example is a random fixed-length segment:
- LR segment: `segment_length` samples at 48kHz (phase0: 16384 ≈ 0.34s)
- HR segment: `segment_length × upsample_factor` samples at 96kHz

Audio shorter than the target segment length is zero-padded.

---

## Validation Metrics

Computed during training at each `checkpoint_interval` epoch, logged to TensorBoard.

| Metric | Type | Interpretation |
|--------|------|----------------|
| SI-SNR | Reference | Signal fidelity in dB; higher = better |
| SDR | Reference | Distortion ratio in dB; higher = better |
| CDPAM | Reference | Perceptual distance; lower = better |
| Audiobox PQ | No-reference | Production quality 0-10 |
| Chroma similarity | Reference | Harmony/melody preservation 0-1 |
| MFCC similarity | Reference | Timbral preservation 0-1 |
| HF energy ratio | SR-specific | Energy above 24kHz: enh/ref ratio |
| Spectral rolloff | SR-specific | Effective bandwidth in Hz |

---

## Configuration Reference

**Files:** `configs/phase0.yaml`, `configs/default.yaml`  
**Constants:** `models/constants.py`

All architecture parameters flow from the config YAML into model constructors. `models/constants.py` defines the defaults used when no config override is present.

Key config sections:

| Section | Keys | Description |
|---------|------|-------------|
| `output.sample_rate` | 96000 | Target sample rate |
| `output.bit_depth` | 24 | Output bit depth |
| `gan.generator.channels` | 512 | Base channel count |
| `gan.generator.upsample_rates` | [2] | Upsampling factor per stage |
| `gan.discriminator.periods` | [2,3,5,7,11,17,23] | MPD periods |
| `gan.discriminator.scales` | 3 | MSD scale count |
| `gan.training.batch_size` | 8 | Batch size |
| `gan.training.lambda_fm/stft/mel` | 2.0/45.0/45.0 | Loss weights |
| `gan.training.mastering.*` | see phase0.yaml | Mastering loss weights |
| `gan.training.quality_sampling` | 7:2:1 | Dataset tier weights |

---

## Training Phases

### Phase 0 (current) — Super-Resolution

Goal: upscale 48kHz → 96kHz/24-bit.  
Status: epoch 0 complete, losses stable (d≈4.2, g≈35).  
Checkpoint: `gdrive:audio-enhancer-datasets/checkpoints/checkpoint_0000.pt`

### Phase 1 (planned) — Degradation Restoration

Goal: restore quality of lossy-compressed / poorly mastered audio.  
Same model architecture, new dataset with degradation pipeline (bad EQ, codec artifacts, stereo damage).  
Plan: `docs/phase1_degradation_plan.md`

### Phase 2 (planned) — Full Pipeline

Goal: end-to-end: any audio in → studio quality out.  
Pipeline: Apollo → AudioSR → custom GAN (fine-tuned on real-world pairs).
