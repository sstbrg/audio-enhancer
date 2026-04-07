# Phase 0 Training Analysis: Epoch 0 Results and Optimization Plan

**Author**: Lara (AI Team Lead)  
**Date**: 2026-04-04  
**Status**: Epoch 0 complete, planning continued training

## Epoch 0 Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Generator loss (g) | ~35.0 | Expected for epoch 0 |
| Discriminator loss (d) | ~4.2 | Healthy, D learning but not dominating |
| Encodec spikes | Resolved | lambda_encodec reduced to 0.01 |
| Training time | ~5 hours | RTX 3090, batch_size=8 |
| AMP | Not used in epoch 0 | Committed but untested |
| torch.compile | Not used in epoch 0 | Committed but untested |

### Loss Decomposition (approximate at epoch 0)

The total generator loss of ~35 is composed of:

| Component | Weight | Estimated Contribution |
|-----------|--------|----------------------|
| Adversarial (MPD + MSD) | 1.0 | ~2-4 |
| Feature matching | lambda_fm=2.0 | ~2-6 |
| Multi-res STFT | lambda_stft=45.0 | ~10-15 |
| Mel spectrogram | lambda_mel=45.0 | ~5-10 |
| Perceptual STFT (mastering) | lambda=45.0 | ~3-6 |
| Stereo image | lambda=10.0 | ~1-3 (if stereo) |
| Dynamics | lambda=5.0 | ~0.5-1 |
| Encodec embedding | lambda=0.01 | ~0.01-0.05 |

The spectral losses (STFT + mel + perceptual STFT) dominate at 135x the adversarial scale.
This is intentional and correct for early training: spectral losses provide stable gradients
that guide the generator toward a reasonable baseline before adversarial pressure refines
perceptual quality.

### Discriminator Health

d_loss = 4.2 is the sum of MPD losses across 7 periods (2,3,5,7,11,17,23) plus MSD
losses across 3 scales. At equilibrium, each sub-discriminator's loss converges to
log(4) ~= 1.39. With 10 total sub-discriminators, the theoretical equilibrium is ~13.9.
At 4.2, the discriminators are still learning to distinguish real from generated audio,
which is healthy for epoch 0.

## Issues Found (Pre-Next-Run)

The following bugs must be fixed before the next training run:

| ID | Issue | Impact | Fix Complexity |
|----|-------|--------|---------------|
| #11 | AMP scaler.update() called after skipped G step | Corrupts loss scale | One-liner gate |
| #8 | input_nyquist hardcoded as output_sr // 4 | Wrong HF mask if SR changes | One-liner constant |
| #7 | Temp file leak in _run_validation | Disk fills over epochs | Add try/finally |
| #13 | epoch variable unbound risk | Crash if 0-epoch range | Init sentinel |
| #12 | Validation uses training loader | Metrics unreliable | Medium effort |

## Optimization Recommendations for Continued Training

### 1. Increase Segment Length: 16384 -> 32768 (HIGH IMPACT)

**Current**: segment_length=16384 (~0.17s at 96kHz output, ~0.34s at 48kHz input)  
**Recommended**: segment_length=32768 (~0.34s at 96kHz, ~0.68s at 48kHz)

**Rationale**:
- The lowest STFT resolution (4096 FFT) at hop=2400 gives only 6-7 frames over
  16384 samples. This provides very coarse spectral loss gradients.
- Dynamics loss with K-weighted LUFS needs >0.4s to be meaningful (ITU-R BS.1770
  specifies 0.4s integration windows). At 0.17s output, the dynamics loss is
  essentially measuring short-term energy, not actual dynamic range.
- The default.yaml already specifies 32768, suggesting this was the original intent.
- With AMP (untested but committed), 32768 at batch_size=8 should fit in 24GB VRAM.

**Risk**: Higher memory usage. If OOM, reduce batch_size to 6.

### 2. Enable and Test AMP + torch.compile (HIGH IMPACT)

Both were committed but never tested in a training run. Expected benefits:
- **AMP**: ~40% memory reduction, ~30% throughput increase
- **torch.compile**: ~10-20% throughput increase (after warmup)
- Combined: potentially 50% faster epochs

**Action**: Must test on Vast.ai before committing to a long run. Run 100 steps with
AMP enabled, verify losses match non-AMP values within tolerance, check for NaN/Inf.

### 3. LR Warmup for Resumed Training (MEDIUM IMPACT)

**Current**: ExponentialLR(gamma=0.999), no warmup  
**Recommendation**: Not needed for Phase 0 continuation (warmup matters for cold start).
Important for Phase 1 fine-tuning — add LinearLR warmup over 500 steps when starting
from a pretrained checkpoint.

The current ExponentialLR schedule is appropriate:
- After 200 epochs: LR = 0.0002 * 0.999^200 = 0.000164 (18% reduction)
- Mild decay prevents G/D balance disruption
- Aggressive decay is dangerous in GANs

### 4. Batch Size Exploration (MEDIUM IMPACT, needs profiling)

With AMP enabled, memory usage drops significantly. This opens the possibility of:
- batch_size=12: ~50% more samples per step, better gradient estimates
- batch_size=16: ~100% more samples, but may need LR adjustment

**LR scaling rule**: If batch doubles, LR should increase by sqrt(2) for AdamW.
batch_size=16 with LR=0.00028 would be a reasonable setting.

**Action**: Profile memory on Vast.ai with batch_size=[8, 12, 16] at segment_length=32768.

### 5. Encodec Loss Weight Increase (LOW IMPACT, deferred)

lambda_encodec=0.01 is very conservative. The encodec embedding loss captures high-level
perceptual similarity. As training stabilizes (epochs 5-10), consider increasing to 0.1
and monitoring for instability. Do NOT change for the immediate next run.

### 6. Loss Weight Curriculum (RESEARCH, deferred to Phase 1)

Static weights are fine for Phase 0. For Phase 1, consider gradually increasing
adversarial weight and decreasing spectral weights as the generator output quality
improves. This would require a curriculum scheduler in train.py.

## Recommended Config for Next Run

Changes from epoch 0 config (configs/phase0.yaml):

```yaml
# Only changes from current phase0.yaml:
training:
  segment_length: 32768    # was 16384, doubled for better spectral resolution
  # All other settings unchanged — resume from checkpoint_0000.pt
```

Everything else stays the same. The key principle: **change one variable at a time**.
Segment length is the highest-impact single change with lowest risk.

## Training Plan: Epochs 1-20

1. **Fix critical bugs** (#11, #8, #7, #13) before running
2. **Test AMP + torch.compile** with 100 steps on Vast.ai
3. **Run epoch 1** with segment_length=32768, AMP enabled, resume from checkpoint_0000.pt
4. **Monitor**: d_loss should decrease toward ~8-10 range, g_loss should decrease
5. **If OOM**: reduce batch_size to 6, keep segment_length=32768
6. **After epoch 5**: Review loss curves, evaluate with analyzer, decide on adjustments
7. **After epoch 10**: Profile batch_size=12 if AMP memory headroom permits
8. **After epoch 20**: Evaluate model quality, decide if architecture changes needed

## Open Questions for Research

1. **Would a discriminator feature warmup help?** In the current setup, D features are
   used for feature matching loss from step 0. But early D features are random/noisy.
   A warmup that delays feature matching for 100-500 steps might improve early training.

2. **Should we add frequency-band-specific discriminators?** The current MPD/MSD cover
   the full spectrum. A sub-band discriminator focused on the 24-48kHz range (the range
   the generator must synthesize) could provide more targeted gradients.

3. **WavLM vs EnCodec for perceptual loss**: WavLM features (used in FINALLY, NeurIPS 2024)
   may capture perceptual similarity better than EnCodec embeddings for music signals.
   Worth benchmarking once Phase 0 baseline is established.
