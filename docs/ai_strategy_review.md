# AI Strategy Review: Phase 0 Training Assessment

**Author:** Lara (AI Team Lead)
**Date:** 2024-04-04
**Scope:** Review of train.py, losses.py, mastering_losses.py, generator.py, discriminator.py

## 1. Training Stability and Convergence

### Current State
- Phase 0 epoch 0 complete: d~4.2, g~35
- AMP (mixed precision) and torch.compile committed but first run used them
- NaN issues in mastering losses identified and fixed (autocast disabled around spectral transforms, biquad filters, EnCodec LSTM, logsumexp)

### Assessment

**Discriminator loss (d~4.2):** This is reasonable for early training. With 7 MPD periods + 3 MSD scales = 10 sub-discriminators, each contributing ~0.4 to the total, the discriminator is not collapsing or dominating. The least-squares GAN loss should stay in the 1-10 range per sub-discriminator during healthy training. A value of 4.2 at epoch 0 suggests the discriminator is learning but not yet saturating -- good.

**Generator loss (g~35):** This is a composite of adversarial + FM + STFT + mel + mastering. The mastering losses (perceptual STFT at lambda=45, dynamics at lambda=5, encodec at lambda=0.01) dominate early training. This is by design -- spectral losses provide strong gradient signal before adversarial losses become meaningful. As the generator improves, the spectral losses should decrease and adversarial losses should stabilize. A g~35 at epoch 0 is not concerning.

**Stability concerns:**
- The NaN fixes for AMP are solid. Disabling autocast around STFT, mel spectrogram, biquad filters, and EnCodec is the correct approach. These operations are numerically sensitive and should always run in float32.
- The `_safe_add` pattern in `MasteringLoss.forward()` that skips non-finite values is a good safety net. However, it could mask systematic issues. Recommendation: log how often `_safe_add` skips values -- if it happens regularly, the underlying cause should be investigated.
- Gradient clipping at max_norm=10.0 is reasonable. If gradients are frequently being clipped (especially in early epochs), consider lowering LR rather than relying on clipping.

### Convergence Outlook

The architecture is sound for 2x upsampling (48k->96k). HiFi-GAN has strong track record for waveform generation, and the single 2x upsample stage is a simple enough task that convergence within 50-100 epochs is plausible. Key indicators to watch:

1. **Adversarial losses should stabilize** (not monotonically decrease) by epoch 10-20.
2. **STFT and mel losses should decrease steadily** for the first 30-50 epochs.
3. **Encodec embedding loss** should decrease, confirming perceptual improvement.
4. **Validation SI-SNR** should reach >25 dB for the task to be considered successful (simple 2x upsample of music).

## 2. Loss Function Balance and Effectiveness

### Current Loss Weights

| Loss | Weight | Contribution at Epoch 0 (estimated) |
|------|--------|-------------------------------------|
| Adversarial (MPD + MSD) | 1.0 each | ~2-4 |
| Feature matching | lambda_fm=2.0 | ~4-8 |
| Multi-res STFT | lambda_stft=45.0 | ~5-10 |
| Mel spectrogram | lambda_mel=45.0 | ~5-10 |
| Perceptual STFT (auraloss) | lambda=45.0 | ~5-10 |
| Dynamics | lambda=5.0 | ~0.5-2 |
| EnCodec embedding | lambda=0.01 | ~0.5-2 |
| Stereo image | lambda=10.0 | 0 (mono training) |
| CLAP / Audiobox PQ | 0.0 | disabled |

### Analysis

**Strengths:**
- The multi-resolution STFT approach (both the custom 4-resolution and auraloss perceptual STFT) covers a wide range of time-frequency tradeoffs. This is important for 96kHz output where both fine temporal detail and broad spectral structure matter.
- EnCodec embedding loss is a strong perceptual proxy. At lambda=0.01, it provides a gentle learned-perceptual signal without dominating training. The 48kHz resampling for EnCodec is handled correctly.
- DynamicRangeLoss combining crest factor and LUFS matching prevents the common GAN failure mode of compressing dynamics.

**Concerns:**

1. **Spectral loss dominance.** With lambda_stft=45 + lambda_mel=45 + lambda_perceptual_stft=45, spectral losses contribute ~60-70% of the total generator loss. This is fine early in training (strong reconstruction signal), but may prevent the adversarial loss from driving fine-grained improvements later. Recommendation: schedule a gradual decrease of spectral weights after epoch 50 (e.g., decay by 0.99/epoch), letting adversarial and perceptual losses take over.

2. **Feature matching at lambda_fm=2.0.** Feature matching is crucial for GAN stability. The current weight seems reasonable, but if discriminator loss starts oscillating, increasing lambda_fm to 5-10 can help. Feature matching acts as a regularizer.

3. **Stereo loss is inactive.** The StereoImageLoss correctly returns 0 for mono input. Currently training is mono-only. Recommendation: when stereo datasets are used, ensure the loss activates. Consider training with stereo from the start if stereo data is available -- the model is channel-agnostic but the losses are not.

4. **No explicit high-frequency loss.** For super-resolution, the generated high-frequency content (24-48 kHz) is the primary goal. The current STFT losses treat all frequencies equally (or weight by A-curve, which actually de-emphasizes >10 kHz). Recommendation: add a high-frequency energy ratio loss or a STFT loss restricted to the 24-48 kHz band. This would directly penalize the model for failing to generate content above the input Nyquist.

5. **EnCodec operates at 48kHz.** This means it cannot evaluate the quality of generated content above 24 kHz -- exactly the region we care about most. The EnCodec loss is useful for ensuring overall timbral quality but blind to the SR task's primary output. This is acceptable but worth noting.

### Recommendations

- **Add high-frequency STFT loss:** A new STFT loss computed only on the 24-48 kHz band (above input Nyquist) with moderate weight (lambda=10-20). This directly targets the super-resolution objective.
- **Spectral loss annealing:** After a warmup period (20-30 epochs), begin decaying lambda_stft and lambda_mel by 0.995 per epoch. Let adversarial quality improve.
- **Monitor per-loss TensorBoard curves:** The current logging infrastructure supports this. Watch for any single loss term dominating the gradient.

## 3. Architecture Assessment

### Generator

The HiFi-GAN generator with a single 2x upsample stage is appropriate for this task. Key observations:

- **11M parameters** is lean. For comparison, original HiFi-GAN V1 had ~14M for 22kHz synthesis. Our task is simpler (2x upsample vs full synthesis), so 11M should suffice.
- **Skip connection** (linear interpolation baseline + learned residual) is a strong design choice for SR. The model only needs to learn the high-frequency residual, not the full waveform. This dramatically helps convergence.
- **HF branch** (parallel conv path for harmonics) is a good addition. The 15-kernel conv captures medium-range temporal correlations important for harmonic content.
- **Weight normalization** is standard for HiFi-GAN. The `remove_weight_norm` method is correctly implemented for inference.

**Potential improvements for future epochs:**
- Consider adding a frequency-domain skip connection (STFT of input, zero-pad high frequencies, ISTFT) in addition to the time-domain interpolation skip.
- The single upsample stage means the receptive field before upsampling is at 48kHz resolution. A wider kernel in `conv_pre` (currently 7) or an additional pre-processing block could help capture longer-range dependencies at the input rate.

### Discriminator

The MPD (periods 2,3,5,7,11,17,23) + MSD (3 scales) design is well-suited for 96kHz audio.

- **Extended periods (17, 23)** beyond the original HiFi-GAN (2,3,5,7,11) help capture longer periodic structures present in 96kHz audio. Good choice.
- **MSD with spectral norm on scale 0** is standard and correct.
- **MSD pool size of 4 with stride 2** creates 3 scales at roughly 96k, 48k, and 24k effective sample rates. This covers the full output bandwidth.

**No changes recommended** for the discriminator architecture at this stage.

## 4. Training Infrastructure

### Positive Observations
- AMP with separate grad scalers for G and D is correct.
- `torch.compile` after checkpoint load avoids `_orig_mod` key mismatch.
- Persistent DataLoader workers with prefetch_factor=4 minimizes I/O bottleneck.
- WeightedRandomSampler for quality-aware batch composition is well-designed.
- Checkpoint saves optimizer, scheduler, and scaler state for clean resume.

### Recommendations for Next Run

1. **Increase segment_length.** Currently 16384 samples at 48kHz = ~0.34 seconds. This is very short for music. The discriminator may not learn meaningful structure at this length. Recommendation: increase to 32768 (0.68s) or 65536 (1.36s) if GPU memory permits with AMP.

2. **Learning rate warmup.** The current training starts at full LR (0.0002). Adding a linear warmup over the first 500-1000 steps would improve early stability, especially with AMP and compiled models.

3. **Validation on a held-out set.** Currently validation runs on training data (`_run_validation` uses the training loader). This means validation metrics reflect training data, not generalization. Recommendation: create a small held-out validation split (5-10% of data or a dedicated test set).

4. **Exponential LR decay (gamma=0.999)** per epoch is very gentle -- after 200 epochs, LR decays to 0.0002 * 0.999^200 = 0.000164. This is fine but could be more aggressive after 100 epochs. Consider a step scheduler that halves LR at epochs 100 and 150.

5. **Log gradient norms** for both G and D. This helps diagnose training instabilities before they cause visible loss spikes. Add `torch.nn.utils.clip_grad_norm_` return value to TensorBoard.

## 5. Priority Actions

In order of impact:

1. **Run more epochs** with current setup. Epoch 0 tells us the pipeline works; we need 10-20 epochs to assess convergence trajectory.
2. **Increase segment_length** to 32768+ for more musically meaningful training samples.
3. **Add validation split** to measure generalization, not just training loss.
4. **Add high-frequency STFT loss** to directly optimize the super-resolution objective.
5. **Log gradient norms** for early warning on instability.
6. **Implement spectral loss annealing** schedule for post-warmup training.
