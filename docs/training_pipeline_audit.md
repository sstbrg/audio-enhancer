# Training Pipeline Audit

**Date:** 2026-04-04
**Auditor:** Adam (AI Engineer - Training & Losses)
**Scope:** train.py, models/losses.py, models/mastering_losses.py, models/generator.py, models/discriminator.py

## 1. AMP (Mixed Precision) Safety

### Previously Fixed (commits b5d5a4e, 708ba00)
- `STFTLoss`: `torch.autocast("cuda", enabled=False)` around `torch.stft` and log/norm ops
- `MelSpectrogramLoss`: autocast disabled around MelSpectrogram transform
- `PerceptualSTFTLoss`: autocast disabled for auraloss STFT
- `StereoImageLoss.forward`: autocast disabled for auraloss mid/side STFT
- `DynamicRangeLoss._crest_factor_loss`: explicit `.float()` cast (logsumexp overflow)
- `DynamicRangeLoss._lufs_loss`: explicit `.float()` cast (highpass_biquad + log10)
- `EncodecEmbeddingLoss`: autocast disabled + `.float()` for LSTM encoder

### Fixed in This Audit
- **`StereoImageLoss._stereo_width_loss`**: Was NOT protected from float16. Squaring, mean, and division operations can overflow/underflow in fp16. The epsilon `1e-8` is near the float16 minimum positive value (~6e-8), making it ineffective as a guard. **Fix:** Wrapped in `torch.autocast("cuda", enabled=False)` with explicit `.float()` cast.

### Verified Safe
- Generator forward pass: all ops (Conv1d, ConvTranspose1d, LeakyReLU, tanh, interpolate) are AMP-safe
- Discriminator forward pass: Conv1d/Conv2d, LeakyReLU, AvgPool1d are AMP-safe
- Adversarial losses (LS-GAN `(1-x)^2`): simple arithmetic, safe in fp16
- Feature matching loss (L1): safe in fp16
- AudioboxPQLoss / CLAPEmbeddingLoss: run under `@torch.no_grad()`, validation only

## 2. Loss Function Balance

### Phase 0 Config Weights (phase0.yaml)
| Component | Weight | Typical Magnitude | Weighted Contribution |
|-----------|--------|-------------------|----------------------|
| Adversarial (MPD+MSD) | 1.0 | 2-8 | 2-8 |
| Feature matching | 2.0 | 5-20 | 10-40 |
| Multi-res STFT | 45.0 | 2-10 | 90-450 |
| Mel spectrogram | 45.0 | 2-10 | 90-450 |
| Perceptual STFT (auraloss) | 45.0 | 1-5 | 45-225 |
| Stereo image | 10.0 | 0 (mono training) | 0 |
| Dynamics | 5.0 | 0.5-5 | 2.5-25 |
| EnCodec embedding | 0.01 | 50-200 | 0.5-2 |

### Assessment
- **Spectral losses dominate early training** (STFT + mel + perceptual STFT ~225-1125): This is appropriate for super-resolution where reconstruction fidelity matters first.
- **Adversarial loss is small initially** (~2-8): Correct for GAN training -- adversarial signal should not overwhelm reconstruction early on.
- **EnCodec at 0.01** is well-calibrated. Raw EnCodec MSE is 50-200+, so this keeps it at ~0.5-2 weighted contribution.
- **Dynamics at 5.0** is reasonable. Crest factor is clamped to 100.0, and LUFS has 0.1 multiplier.
- **Feature matching at 2.0** is standard HiFi-GAN value.
- **Reported g=35 at epoch 0 end** confirms reasonable overall scale.
- **No rebalancing needed** at this stage. As training progresses, may want to reduce spectral weights or increase adversarial weight for perceptual quality.

## 3. Gradient Clipping

### Configuration
- `max_norm=10.0` applied to both generator and discriminator
- `scaler.unscale_()` correctly called before clipping (required for AMP)
- Gradient norm now configurable via `grad_clip_norm` in training config (was hardcoded)

### Assessment
- 10.0 is a reasonable default. HiFi-GAN does not use gradient clipping originally, but AMP introduces occasional inf gradients that need clipping.
- **Added gradient norm logging** (`grad_norm/generator`, `grad_norm/discriminator`) to TensorBoard for monitoring. If norms are consistently hitting the clip threshold, consider increasing it. If norms are always well below 10.0, the clip is not constraining training.
- **Added AMP scaler scale logging** (`amp/scale_g`, `amp/scale_d`). If the scale drops significantly (below ~128), it indicates frequent NaN/inf in gradients, suggesting AMP instability.

## 4. torch.compile Compatibility

### Verified Compatible
- **Generator**: Conv1d, ConvTranspose1d, LeakyReLU, tanh, `F.interpolate(scale_factor=...)` -- all standard ops
- **Discriminators**: Conv1d, Conv2d, LeakyReLU, AvgPool1d, `torch.flatten` -- all standard ops
- **weight_norm**: Uses `torch.nn.utils.parametrizations.weight_norm` (parametrize-based, compile-safe)
- **spectral_norm**: Uses `torch.nn.utils.parametrizations.spectral_norm` (compile-safe)
- **Fixed shapes**: DataLoader uses `drop_last=True`, `segment_length` is fixed, upsample factor is deterministic. No data-dependent shape variations that would cause recompilation.
- **PeriodDiscriminator padding**: Conditional on `t % period`, but with fixed segment lengths this is deterministic per period. No graph breaks.

### Not Compiled (by design)
- Loss functions (STFTLoss, MelSpectrogramLoss, MasteringLoss) are NOT compiled. They contain:
  - `torch.autocast("cuda", enabled=False)` context managers
  - `warnings.warn()` calls (graph break trigger)
  - Lazy model loading (EnCodec)
  - These are called outside compiled regions, so this is correct.

### Notes
- Compilation happens AFTER checkpoint load (correct order to avoid `_orig_mod.*` key mismatch)
- Initial compilation adds ~2-5 minutes one-time overhead
- The `xs = 0` integer accumulator in generator resblock summation is promoted correctly by PyTorch

## 5. Other Improvements Made

### Performance
- **MasteringLoss**: Refactored `_safe_add` to avoid per-term `.item()` calls that force CUDA synchronization. Previous code had up to 6 syncs per training step inside the mastering loss. Now uses `torch.isfinite()` on GPU tensors, with `.item()` calls only during logging.

### Robustness
- **train.py final checkpoint**: Fixed potential `NameError` if training loop is empty (when `start_epoch >= epochs`). Now safely falls back to `start_epoch - 1`.

### Observability
- **Gradient norm logging**: Added `grad_norm/generator` and `grad_norm/discriminator` to TensorBoard.
- **AMP scale logging**: Added `amp/scale_g` and `amp/scale_d` to TensorBoard for diagnosing AMP instability.
- **Config-driven grad clipping**: `grad_clip_norm` is now in training config instead of hardcoded, with fallback default of 10.0.

## 6. Files Modified

- `models/mastering_losses.py`: AMP safety for `_stereo_width_loss`, refactored `_safe_add` performance
- `train.py`: gradient norm logging, AMP scale logging, config-driven grad_clip_norm, epoch edge case fix
- `configs/phase0.yaml`: added `grad_clip_norm: 10.0`
- `configs/default.yaml`: added `grad_clip_norm: 10.0`

## 7. Recommendations for Future

1. **Monitor gradient norms** in TensorBoard after next training run. If generator norms frequently hit 10.0, consider increasing to 20.0 or investigating which loss term causes spikes.
2. **Monitor AMP scaler scale** -- if it drops below ~128, there may be remaining fp16-unsafe operations in custom code paths.
3. **Loss schedule**: Consider reducing `lambda_stft` and `lambda_mel` after ~50 epochs when reconstruction is stable, to let adversarial loss drive perceptual quality improvements.
4. **Compile mode**: Current default mode is fine. If training is stable, try `torch.compile(mode="reduce-overhead")` for additional speedup (uses CUDA graphs).
