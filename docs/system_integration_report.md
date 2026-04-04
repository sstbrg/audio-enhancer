# System Integration Report

**Author:** Jason (system-engineer)
**Date:** 2026-04-04
**Scope:** Full cross-subsystem review of audio-enhancer

---

## 1. System Health Assessment

Overall status: **Stable with minor integration gaps fixed**

The project's six subsystems -- training, inference, evaluation, analysis UI, data pipeline, and infrastructure -- are well-structured and coherent. The core training loop (train.py) and model architecture (generator.py, discriminator.py) are correctly wired. Phase 0 epoch 0 completed successfully, confirming end-to-end pipeline viability.

### Subsystem Status

| Subsystem | Status | Notes |
|-----------|--------|-------|
| Training (train.py) | Good | AMP, torch.compile, degradation pipeline all wired |
| Generator (models/generator.py) | Good | 11M params, 2x upsample, skip connection, HF branch |
| Discriminator (models/discriminator.py) | Good | MPD (7 periods) + MSD (3 scales), all from constants |
| Losses (models/losses.py, mastering_losses.py) | Good | NaN-safe autocast guards on all spectral ops |
| Dataset (data/dataset.py) | Good | Quality tiers, weighted sampling, CD degradation |
| Inference (enhance.py) | Fixed | torch.compile key stripping added, stale 192kHz refs fixed |
| Evaluation (evaluate.py, metrics/) | Good | 10 metrics, graceful fallbacks for optional deps |
| Analyzer UI (analyzer_ui.py) | Good | i18n, upscale potential, waveform/spectrogram plots |
| Infra (infra/) | Good | datasets.py dashboard, setup-vastai.sh, rclone integration |

---

## 2. Integration Issues Found and Fixed

### 2.1 CRITICAL: torch.compile checkpoint key mismatch in enhance.py

**File:** `enhance.py:84`

When `torch.compile()` wraps a model in train.py, its `state_dict()` keys get prefixed with `_orig_mod.`. The training script correctly loads before compiling (line 320-325), and uses `strict=False` for the generator. However, `enhance.py` called `load_state_dict()` without stripping this prefix, meaning inference would fail on any checkpoint saved after torch.compile was applied.

**Fix:** Added key prefix stripping in `enhance.py:_load_gan()`:
```python
state_dict = ckpt["generator"]
cleaned = {}
for k, v in state_dict.items():
    cleaned[k.replace("_orig_mod.", "")] = v
self.gan_model.load_state_dict(cleaned)
```

### 2.2 Hardcoded input sample rate in train.py

**File:** `train.py:226`

The dataset's `input_sr` parameter was hardcoded as `48000` instead of using `INPUT_SAMPLE_RATE` from `models/constants.py`. While functionally identical today, this violates the "no magic numbers" rule and could silently break if the constant changes.

**Fix:** Imported and used `INPUT_SAMPLE_RATE` constant.

### 2.3 Stale 192kHz/32-bit references across the codebase

**Files:** `enhance.py`, `prepare_dataset.py`, `configs/default.yaml`

The project pivoted from 192kHz/32-bit to 96kHz/24-bit (Phase 0), but several files still referenced the old target:

- `enhance.py` docstring: "48k->192k" resampling, argparse description "192kHz/32-bit"
- `enhance.py:_run_gan()` docstring: "GAN upsampling to 192kHz"
- `prepare_dataset.py` docstring: "48kHz -> 192kHz GAN", default `--target-sr 192000`
- `configs/default.yaml`: `sample_rate: 192000`, `bit_depth: 32`, `upsample_rates: [2, 2]`, discriminator periods `[2,3,5,7,11]` (missing extended 96kHz periods)

**Fix:** Updated all references to 96kHz/24-bit. Updated default.yaml to match phase0.yaml architecture (single `[2]` upsample, extended discriminator periods `[2,3,5,7,11,17,23]`).

---

## 3. Integration Verification: Config Flow

Verified the end-to-end config flow:

```
phase0.yaml → train.py:load_config()
  → output.sample_rate (96000) → Dataset(target_sr=96000)
  → gan.generator → Generator(upsample_rates=[2], ...)
  → gan.discriminator → MPD(periods=[2,3,5,7,11,17,23]), MSD(scales=3)
  → gan.training → batch_size, lambdas, mastering weights
  → quality_sampling → WeightedRandomSampler
  → degradation → AudioSRDataset degradation params
```

Checkpoint saves config dict, which enhance.py reads to reconstruct the generator architecture -- this is a good design that prevents config/model mismatch at inference time.

---

## 4. Sample Rate Consistency Audit

| Path | Input SR | Output SR | Status |
|------|----------|-----------|--------|
| train.py dataset creation | INPUT_SAMPLE_RATE (48000) | config output.sample_rate (96000) | OK |
| Generator upsample | 48kHz (1ch, mono) | 96kHz via 2x ConvTranspose1d | OK |
| enhance.py: <44.1kHz input | resample to 48kHz | GAN -> 96kHz | OK |
| enhance.py: 48kHz input | direct | GAN -> 96kHz | OK |
| enhance.py: no GAN fallback | Kaiser resample to target | config-driven | OK |
| EnCodec loss | resample to 48kHz | N/A (embedding space) | OK |
| CLAP loss | resample to 48kHz | N/A (embedding space) | OK |
| Validation metrics (train.py) | output_sr (96kHz) | N/A (metrics computed at output SR) | OK |
| evaluate.py SI-SNR/SDR | resample to 48kHz | N/A | OK |
| Audiobox aesthetics | resample to 16kHz | N/A | OK |

---

## 5. Checkpoint Format Forward-Compatibility

Current checkpoint structure:
```python
{
    "epoch": int,
    "generator": state_dict,       # may have _orig_mod. prefix
    "mpd": state_dict,
    "msd": state_dict,
    "optim_g": state_dict,
    "optim_d": state_dict,
    "sched_g": state_dict,
    "sched_d": state_dict,
    "scaler_g": state_dict,
    "scaler_d": state_dict,
    "config": dict,                # full YAML config
}
```

**Forward-compatibility assessment:**

- Good: Config is saved with checkpoint, so enhance.py reconstructs the exact architecture
- Good: `strict=False` on generator load allows adding/removing layers between phases
- Good: Optimizer state incompatibility is caught and gracefully re-initialized
- Good: Scheduler and scaler state are backward-compatible (checks `if "sched_g" in ckpt`)
- Risk: If discriminator architecture changes significantly for Phase 1, `mpd.load_state_dict()` and `msd.load_state_dict()` use `strict=True` -- a resumed Phase 1 training on a Phase 0 checkpoint could crash. Consider adding `strict=False` for discriminator loading too, since discriminator state is less critical to preserve.

---

## 6. Remaining Risks and Recommendations

### High Priority

1. **Validation uses training DataLoader** (`train.py:477`): `_run_validation()` receives the training `loader`, meaning validation metrics are computed on training data, not a held-out set. This doesn't measure generalization. Recommend adding a separate validation split or directory.

2. **`val_samples = 4` hardcoded** (`train.py:62`): This should be a config parameter or constant, not a magic number embedded in the function.

3. **Apollo expects 44.1kHz** (`enhance.py:209`): The inference pipeline resamples to 44.1kHz for Apollo, then AudioSR brings it to 48kHz, then GAN to 96kHz. If Apollo output quality at 44.1kHz is lower than a direct 48kHz path, this intermediate step may degrade quality. Worth A/B testing.

### Medium Priority

4. **No learning rate warmup**: The training loop jumps directly to `learning_rate_g/d = 0.0002` with exponential decay. A short warmup (100-500 steps) would stabilize early training, especially with AMP.

5. **Discriminator checkpoint loading is strict**: As noted in section 5, `mpd.load_state_dict()` and `msd.load_state_dict()` will crash if architecture changes. Add `strict=False` before Phase 1 begins.

6. **`default.yaml` was dangerously stale**: It had 192kHz/32-bit with 4x upsample -- using it with `enhance.py` or `train.py` without `--config phase0.yaml` would create model architecture mismatches. Fixed in this review.

7. **`prepare_dataset.py` uses scipy.resample_poly**: This is a different resampling implementation than the Kaiser-windowed `torchaudio.transforms.Resample` used in `utils/audio.py` and the training pipeline. The inconsistency could introduce subtle spectral differences between prepared datasets and on-the-fly training pairs. Consider unifying on the torchaudio resampler.

### Low Priority

8. **`mel_spectrogram_loss` function still exported** (`models/__init__.py`): The deprecated function wrapper is still in the public API. No current code calls it, but it could confuse future development.

9. **Onset F1 matching algorithm is O(n*m)** (`metrics/evaluate.py:428`): The nested loop for onset matching could be slow for long files with many onsets. A sorted-merge approach would be O(n+m).

10. **CLAP loss resamples to 48kHz but uses `get_audio_embedding_from_data` with numpy** (`mastering_losses.py:398-406`): The CPU roundtrip through numpy is slow and breaks the GPU pipeline. This is fine since CLAP is validation-only, but worth noting.

---

## 7. Priority Items for Next Training Run

1. **Test AMP + torch.compile** -- committed but untested in actual training. The NaN guards are in place (autocast disabled for STFT, mel, encodec, auraloss, dynamics). Should work, but monitor for NaN/inf in first 100 steps.

2. **Add validation split** -- create a small held-out set (e.g., 5% of data or a fixed 50 files) to get meaningful generalization metrics during training.

3. **Move `val_samples` to config** -- add `validation_samples: 4` to the training section of phase0.yaml.

4. **Add discriminator `strict=False`** -- preemptive fix for Phase 1 checkpoint loading.

5. **Verify checkpoint key compatibility** -- after a training run with torch.compile, verify that `enhance.py` can load the checkpoint correctly with the new key stripping code.

---

## 8. Files Modified in This Review

- `/home/stas/audio-enhancer/enhance.py` -- Fixed torch.compile key stripping, updated stale 192kHz references
- `/home/stas/audio-enhancer/train.py` -- Replaced hardcoded 48000 with INPUT_SAMPLE_RATE constant
- `/home/stas/audio-enhancer/prepare_dataset.py` -- Updated default target-sr from 192000 to 96000
- `/home/stas/audio-enhancer/configs/default.yaml` -- Aligned with Phase 0 architecture (96kHz/24-bit, single 2x upsample, extended discriminator periods)
