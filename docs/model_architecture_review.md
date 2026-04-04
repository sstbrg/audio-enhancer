# Model Architecture Review

**Reviewer:** Kyle (AI Engineer -- GANs/Architecture)
**Date:** 2026-04-04
**Scope:** Generator, discriminator, losses, training dynamics
**Codebase ref:** `models/generator.py`, `models/discriminator.py`, `models/constants.py`, `models/losses.py`, `models/mastering_losses.py`

---

## 1. Generator Architecture

### 1.1 Parameter count

Measured: **9.98M** parameters (close to the documented 11M).

| Component | Params | Notes |
|-----------|--------|-------|
| `conv_pre` | 4,608 | 1->512ch, kernel 7 |
| `ups[0]` (ConvTranspose1d) | 525,056 | 512->256ch, k=4, s=2 |
| `resblock[0]` (k=3) | 1,182,720 | 256ch, dilations [1,3,5] |
| `resblock[1]` (k=7) | 2,755,584 | 256ch, dilations [1,3,5] |
| `resblock[2]` (k=11) | 4,328,448 | 256ch, dilations [1,3,5] |
| `hf_branch` | 1,180,672 | k=15 + k=3, parallel path |
| `conv_post` | 1,794 | 256->1ch, kernel 7 |

The size is appropriate for 2x upsampling. HiFi-GAN V1 (original) uses ~14M for 256x upsampling (mel->waveform). Our task is simpler (waveform 2x), so ~10M is well-calibrated. The resblocks dominate the parameter budget (83%), which is correct -- they do the heavy lifting of learning harmonic structure.

### 1.2 Skip connection

The skip connection uses `F.interpolate(mode="linear")` to upsample the input, then adds it to the `tanh` output of the learned path:

```python
return tanh(conv_post(x)) + skip
```

**Issue identified:** The output range is unbounded. `tanh` produces [-1, 1], and the skip adds the interpolated input on top. For full-scale audio (peaks near +/-1.0), the output can reach +/-2.0. This is handled at inference time in `enhance.py:300` with `torch.clamp(waveform, -1.0, 1.0)`, but during training the discriminator sees unclamped values while targets are in [-1, 1]. This creates an asymmetry that could confuse the discriminator and waste generator capacity.

**Recommendation (Priority: Medium):** Apply `torch.tanh` to the final combined output, or scale the residual branch:

```python
# Option A: Tanh the combined output (simple, constrains range)
return torch.tanh(x + skip)

# Option B: Scale the residual (preserves gradients better)
return 0.5 * torch.tanh(x) + skip
```

Option B is preferred because it preserves the linear skip path while keeping the learned residual bounded. The 0.5 scaling means the generator can add/subtract at most 0.5 from the interpolated input, which is physically reasonable for bandwidth extension -- the high-frequency content being generated is lower amplitude than the low-frequency carrier.

**Risk if changed mid-training:** This would change the output distribution. If applied after resuming from the existing checkpoint, re-warm the discriminator for a few hundred steps with a lower G learning rate. Or apply only on a fresh training run.

### 1.3 Receptive field

Gradient-based measurement: **74 input samples = 148 output samples at 96kHz = 1.54 ms**.

This is short. For context:
- A single cycle of 440 Hz (concert A) = 2.27 ms at 96kHz
- Musical transients (drum attacks, consonants) = 5-20 ms
- HiFi-GAN V1 original: ~260 output samples

The limited RF means the model may struggle with:
- Low-frequency harmonic generation (it cannot "see" a full cycle of bass)
- Phase-coherent reconstruction across longer timescales
- Capturing musical context for intelligent harmonic extension

**Recommendation (Priority: High for next training run):**
1. Add an additional upsample stage with identity rate to deepen the network. For example, add `upsample_rates: [1, 2]` with a 1x "upsample" that just adds more resblocks. This doubles the resblock count and roughly doubles RF.
2. Alternatively, increase dilation factors to `[1, 3, 5, 7, 11]` to widen RF without adding parameters.
3. The `hf_branch` with kernel 15 adds some RF, but it operates after all resblocks so it only extends the final stage. Consider moving it before `conv_post` and increasing to kernel 31 or 63.

### 1.4 High-frequency branch

The `hf_branch` is a parallel path (k=15 conv -> LeakyReLU -> k=3 conv -> LeakyReLU) added to the main path. This is a good idea in principle -- it gives the model a dedicated pathway for generating high-frequency detail without interfering with the skip connection's low-frequency pass-through.

**Observation:** The branch output is added without any gating or scaling. During early training, this branch's random output adds noise. Consider adding a learnable scaling parameter initialized to 0 (a la ReZero) so it gradually contributes:

```python
self.hf_scale = nn.Parameter(torch.zeros(1))
# In forward:
hf = self.hf_branch(x)
x = x + self.hf_scale * hf
```

This is a minor optimization -- the model will likely learn to suppress noise naturally, but ReZero can speed up early convergence.

### 1.5 ConvTranspose1d configuration

- Kernel 4, stride 2: `k % s == 0` -- no checkerboard artifacts. Correct.
- Padding `(k-s)//2 = 1`: produces exact 2x upsampling (verified for lengths 16384, 24000, 32768, 48000).
- Weight initialization: N(0, 0.01) via `init_weights`. Standard and appropriate.

### 1.6 Activation functions

LeakyReLU with slope 0.1 throughout (from `constants.py`). This is the standard HiFi-GAN choice. No issues.

However, the activation is applied *before* the upsample ConvTranspose1d (`F.leaky_relu(x, slope)` then `self.ups[i](x)`). This is correct -- pre-activation residual blocks are standard in HiFi-GAN and generally outperform post-activation in waveform generation.

### 1.7 Weight normalization

Weight norm is applied to all convolutions in the generator. This is standard for HiFi-GAN and helps with training stability. The `remove_weight_norm` method is correctly implemented for inference.

---

## 2. Discriminator Architecture

### 2.1 Multi-Period Discriminator (MPD)

Periods: `[2, 3, 5, 7, 11, 17, 23]` -- 7 sub-discriminators.

**Coverage analysis at 96kHz:**

| Period | Fundamental | Musical relevance |
|--------|------------|-------------------|
| 2 | 48,000 Hz | Nyquist of input -- critical for detecting aliasing |
| 3 | 32,000 Hz | Upper harmonics |
| 5 | 19,200 Hz | Upper harmonic fundamentals |
| 7 | 13,714 Hz | Mid-high frequency detail |
| 11 | 8,727 Hz | Mid-range presence |
| 17 | 5,647 Hz | Vocal presence, guitar attack |
| 23 | 4,174 Hz | Speech/vocal fundamentals |

**Assessment:** Good coverage. All periods are coprime (verified), ensuring no redundancy. The original HiFi-GAN used [2, 3, 5, 7, 11] for 22.05kHz output. Adding 17 and 23 for 96kHz is well-motivated -- the higher sample rate means the original periods now alias to different frequency regions, and the larger primes capture structure in the 4-6 kHz range that is perceptually critical.

**Potential improvement (Priority: Low):** Add period 29 or 31 to cover the 3-3.3 kHz range (formant region, very perceptually sensitive). However, this adds ~8M parameters per period discriminator, so the cost-benefit should be weighed against training time.

### 2.2 Multi-Scale Discriminator (MSD)

3 scales with AvgPool1d(4, 2, padding=2) between scales:
- Scale 0: 96kHz (spectral norm)
- Scale 1: ~48kHz effective
- Scale 2: ~24kHz effective

**Assessment:** Standard HiFi-GAN MSD. Spectral norm on the first (highest resolution) discriminator is correct -- it prevents the highest-frequency discriminator from dominating.

**Potential improvement (Priority: Medium):** For 96kHz output, consider adding a 4th scale to capture structure at ~12kHz effective rate. The MSD currently bottoms out at 24kHz, missing patterns in the 12-24kHz band that are important for hi-fi audio quality judgment.

### 2.3 D/G Parameter ratio

**D/G ratio: 8.7x** (87.2M discriminator vs 10.0M generator).

This is on the high side. Typical GAN ratios are 2-5x. A very powerful discriminator relative to the generator can lead to:
- Generator unable to "fool" discriminator early in training, causing vanishing G gradients
- Feature matching loss dominating (which is a form of distillation from the discriminator)

However, for audio super-resolution where the task is relatively constrained (the skip connection handles the easy part), a strong discriminator is actually beneficial -- it pushes the generator to produce genuinely realistic high-frequency content rather than noise.

**Recommendation:** Monitor the D loss and G loss curves. If D loss drops to near-zero while G loss plateaus or increases, the discriminator is overpowering. In that case:
1. Reduce D learning rate (try `learning_rate_d: 0.0001`)
2. Train D every 2 steps instead of every step
3. Or reduce MPD periods to [2, 3, 5, 7, 11] (original HiFi-GAN set)

Current epoch 0 losses (d~4.2, g~35) suggest the discriminator is not yet dominating, which is healthy.

### 2.4 Discriminator architecture details

**PeriodDiscriminator:** 5 conv layers (32->128->512->1024->1024->1), stride (3,1) in the period dimension. Standard HiFi-GAN.

**ScaleDiscriminator:** 7 conv layers with grouped convolutions (groups=4 and 16). The grouped convs are efficient and allow capturing independent frequency sub-band features. Kernel size 41 is large, giving the MSD substantial temporal context.

No issues found in either architecture.

---

## 3. Loss Functions

### 3.1 Multi-Resolution STFT Loss

Resolutions: `[512/50/240, 1024/120/600, 2048/240/1200, 4096/480/2400]`

**Issue:** The hop sizes are not powers of 2, which is fine functionally but the ratios are unusual. More importantly, the largest FFT (4096) at 96kHz gives frequency resolution of 23.4 Hz and temporal resolution of 25 ms -- but there is no 8192 or 16384 resolution to capture sub-bass structure (< 12 Hz resolution).

**Recommendation (Priority: Low):** Add a 8192/960/4800 resolution for better low-frequency coverage, matching the perceptual STFT loss which already uses 16384.

### 3.2 Mel Spectrogram Loss

Uses torchaudio's MelSpectrogram with n_fft=4096, hop=480, 128 mels, f_max=48000. This covers the full Nyquist range at 96kHz. The autocast disable and float32 cast are correctly applied (per earlier bug fixes).

No issues.

### 3.3 Feature matching loss

Standard L1 feature matching across all discriminator layers. Detaches real features correctly (`real_feat.detach()`).

No issues.

### 3.4 Mastering losses

- **Perceptual STFT (auraloss):** Mel-scaled + A-weighted, 4 resolutions up to 16384. Good.
- **Stereo image:** Returns 0 for mono input (correct, since training is mono).
- **Dynamics:** Crest factor + LUFS matching. The `clamp(max=1.0)` in logsumexp and float32 cast are good stability fixes.
- **EnCodec embedding:** Resamples to 48kHz, handles mono-to-stereo expansion. The `encoder.train()` + frozen weights for LSTM backward is a correct workaround for PyTorch LSTM limitations.

**Note on EnCodec lambda:** Config has `lambda_encodec: 0.01` -- this is extremely low relative to other losses (perceptual_stft: 45, mel: 45, stft: 45). The encodec embedding MSE operates in a completely different scale than spectral losses, so this may be appropriate, but it raises the question of whether the encodec loss contributes meaningfully at this weight. Worth logging and monitoring.

---

## 4. Training Dynamics Assessment

### 4.1 Current state

From CLAUDE.md: Epoch 0 complete, d~4.2, g~35. This is healthy for early training:
- D loss ~4.2 means the discriminator is learning but not dominant
- G loss ~35 is dominated by spectral reconstruction losses (STFT + mel + mastering), not adversarial loss

### 4.2 Potential stability risks

1. **Skip connection gradient flow:** The linear interpolation skip provides a strong gradient highway to the input. This is good for stability but may cause the learned path (tanh branch) to be undertrained early on, since the skip alone provides reasonable reconstructions.

2. **AdamW betas (0.8, 0.99):** The low beta1=0.8 is standard for GANs (original HiFi-GAN uses 0.8) and provides faster momentum adaptation. Good.

3. **ExponentialLR gamma=0.999:** This decays LR by ~18% over 200 epochs. Conservative but appropriate for a long training run.

4. **Gradient clipping max_norm=10.0:** Applied to both G and D. Standard.

### 4.3 Mode collapse indicators to watch

- D loss oscillating while G loss is flat = mode collapse
- Generated HF content being spectrally "flat" (white noise-like) rather than structured harmonics
- SI-SNR improving but CDPAM stagnating = model learning identity mapping, not generating useful HF

---

## 5. Issues Found (Bugs/Fixes Needed)

### 5.1 Output range asymmetry (generator.py)

**Severity: Medium**
**Location:** `models/generator.py:187`

The generator output `tanh(x) + skip` is unbounded. During training, the discriminator sees values in [-2, 2] for generated audio but [-1, 1] for real audio. This creates a distributional mismatch that the discriminator can exploit trivially (just check if values exceed 1.0).

At inference, `enhance.py:300` clamps to [-1, 1], meaning some generated content is silently discarded.

**Fix: deferred.** Changing this mid-training breaks checkpoint compatibility. Apply on next fresh training run.

### 5.2 Segment length vs receptive field

**Severity: Low**
**Location:** `configs/phase0.yaml:43`

`segment_length: 16384` at 48kHz = 0.34s. The generator RF is 74 samples = 1.5ms. The segment is 220x the RF, so this is fine -- no issue with edge effects.

---

## 6. Actionable Recommendations (Prioritized)

### For the next training run (high priority)

1. **Increase generator receptive field.** Either:
   - Add dilations `[1, 3, 5, 7, 11]` instead of `[1, 3, 5]` (zero param increase, ~2x RF)
   - Or add a second upsample stage at rate 1 for deeper resblocks

2. **Add output normalization.** Change the final output to `0.5 * torch.tanh(x) + skip` or apply tanh to the combined output to match the [-1, 1] range of real audio.

3. **Monitor D/G loss ratio** closely. If D loss drops below 1.0 while G stays above 20, reduce D learning rate or train D every 2 steps.

### For improved quality (medium priority)

4. **Add a 4th MSD scale** for 12kHz effective coverage.

5. **Add ReZero scaling to hf_branch** for faster early convergence.

6. **Consider increasing segment_length to 32768** (0.68s) once AMP memory usage is characterized. Longer segments capture more musical context for the discriminator.

### For future investigation (low priority)

7. **Experiment with Snake activations** (from BigVGAN). Snake activations are sinusoidal and better suited to periodic signals than LeakyReLU. BigVGAN showed significant improvement over HiFi-GAN with Snake.

8. **Anti-aliased upsampling.** Replace the single ConvTranspose1d with a two-step process: nearest-neighbor upsample + learned conv. This avoids spectral copies that ConvTranspose1d can produce.

9. **Add a sub-discriminator specifically for the 24-48kHz band** (the newly generated content). This would provide stronger supervision signal for the exact frequency range the GAN is supposed to create.

---

## 7. Summary

The architecture is well-designed and follows HiFi-GAN best practices adapted for super-resolution. The skip connection is the right choice for bandwidth extension. The discriminator set (MPD + MSD) provides comprehensive coverage of 96kHz audio structure.

The two most impactful improvements for the next run are: (1) increasing the generator receptive field via wider dilations, and (2) fixing the output range mismatch between training and inference. Both can be implemented with minimal code changes in `models/constants.py` and `models/generator.py`.

Overall assessment: **Solid foundation, ready for continued training with minor improvements.**
