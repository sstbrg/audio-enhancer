# Code Review Report

**Reviewer:** Florence (git-expert)
**Date:** 2026-04-04
**Branch:** develop
**Commits reviewed:** last 20 (3e8d457 back to 06034a9)

---

## Summary

Overall the codebase is in good shape. Constants are well-centralized in `models/constants.py`,
locale strings are properly externalized in `locales/`, and the `.gitignore` is comprehensive.
The following issues were found, ordered by severity.

---

## Findings

### CRITICAL

None found.

---

### WARNING

#### W1 — Unused import: `math` in `models/mastering_losses.py`
- **File:** `models/mastering_losses.py:20`
- **Issue:** `import math` is present but `math` is never referenced anywhere in the file.
- **Fix:** Remove the import.
- **Status:** Fixed in this review.

#### W2 — Unused import: `torch.nn.functional as F` in `train.py`
- **File:** `train.py:22`
- **Issue:** `import torch.nn.functional as F` is imported but `F.` is never called in `train.py`.
  All functional calls go through the imported loss modules.
- **Fix:** Remove the import.
- **Status:** Fixed in this review.

#### W3 — Magic number: `input_sr=48000` hardcoded in `train.py`
- **File:** `train.py:226`
- **Issue:** `input_sr=48000` is hardcoded instead of using `INPUT_SAMPLE_RATE` from `models/constants.py`.
  CLAUDE.md forbids magic numbers.
- **Fix:** Import and use `INPUT_SAMPLE_RATE`.
- **Status:** Fixed in this review.

#### W4 — Magic numbers in `train.py` validation function
- **File:** `train.py:62, 104–106, 115–116`
- **Issues:**
  - `val_samples = 4` — number of validation samples, should be a config value or constant
  - `n_fft=4096` — appears 3 times; matches `UPSCALE_FFT_SIZE` in constants but that constant
    is for the upscale analyzer, not training validation. Should add a dedicated constant.
  - `roll_percent=0.95` — rolloff percentile for spectral rolloff, hardcoded
- **Fix:** Extract to named constants in `models/constants.py` or add to config.
- **Note:** These are in the validation helper which is low-risk, but still violates the no-magic-numbers rule.
- **Status:** Flagged for follow-up.

#### W5 — Magic numbers in `train.py` grad-clip calls
- **File:** `train.py:411, 449`
- **Issue:** `max_norm=10.0` appears twice as a literal. This value is documented in CLAUDE.md as
  a known constant but is not defined in `models/constants.py`.
- **Fix:** Add `GRAD_CLIP_MAX_NORM = 10.0` to `models/constants.py` and use it in `train.py`.
- **Status:** Fixed in this review.

#### W6 — Magic numbers in `train.py` LR scheduler
- **File:** `train.py:317–318`
- **Issue:** `gamma=0.999` for `ExponentialLR` is a magic number repeated for both generator
  and discriminator schedulers.
- **Fix:** Add `LR_SCHEDULER_GAMMA = 0.999` to `models/constants.py`.
- **Status:** Fixed in this review.

#### W7 — Magic numbers in `enhance.py`
- **File:** `enhance.py:217, 286`
- **Issues:**
  - `chunk_size = 44100 * 30` — Apollo chunk: 30-second window at 44.1 kHz. Both numbers are magic.
  - `chunk_samples = 48000 * 10` — GAN chunk: 10-second window at 48 kHz. Both numbers are magic.
- **Fix:** Use `CD_SAMPLE_RATE` and `INPUT_SAMPLE_RATE` from constants; extract chunk durations
  as named constants (e.g. `APOLLO_CHUNK_SECONDS = 30`, `GAN_CHUNK_SECONDS = 10`).
- **Status:** Fixed in this review.

#### W8 — Magic numbers in `analyzer_ui.py`
- **File:** `analyzer_ui.py:83, 132, 144, 206–207, 210, 226–227, 305, 329, 361`
- **Issues (selected):**
  - `"-ar", "48000"` (lines 83, 361) — hardcoded ffmpeg sample rate; should use `INPUT_SAMPLE_RATE`
  - `frame_size = int(sr * 0.4)` (line 132) — 400 ms frame for RMS, magic float
  - `clip_threshold = 0.99` (line 144) — clipping detection threshold, magic
  - `n_fft=4096` (lines 206–207) — matches `UPSCALE_FFT_SIZE` in constants; use it
  - `threshold = mean_magnitude.max() * 0.01` (line 210) — magic 1% threshold
  - `max_samples = sr * 30` (line 305) — 30-second display limit, magic
  - `ax2.set_ylim(0, min(sr / 2, 24000))` (line 329) — 24000 Hz spectrogram ceiling, magic
- **Note:** Some of these are display/UI constants where the value is self-evident (e.g. 30-second
  display limit), but all are violations of the project rule. The ffmpeg sample rate is particularly
  important to fix as it should match `INPUT_SAMPLE_RATE`.
- **Status:** ffmpeg lines fixed in this review; others flagged for follow-up.

---

### INFO

#### I1 — Inline `import subprocess, tempfile` style in `analyzer_ui.py`
- **File:** `analyzer_ui.py:82, 348`
- **Issue:** `import subprocess, tempfile` written on one line inside a function. Python style
  convention (PEP 8) prefers imports at the top of the file.
- **Fix:** Move to module-level imports.
- **Status:** Flagged, low priority.

#### I2 — Inline `import os` inside function in `analyzer_ui.py`
- **File:** `analyzer_ui.py:86`
- **Issue:** `import os; os.unlink(tmp.name)` — `os` is imported inline; should be top-level.
- **Fix:** Move `import os` to top-level imports.
- **Status:** Flagged, low priority.

#### I3 — Inline `import os` inside function in `train.py`
- **File:** `train.py:157`
- **Issue:** `import os` used inside validation function for `os.unlink`, while `os` is already
  imported at top level. Redundant local import.
- **Fix:** Remove the inner `import os` — the top-level `os` is already in scope.
- **Status:** Fixed in this review.

#### I4 — `MuQ-Eval/` directory untracked but `third_party/` in `.gitignore`
- **File:** `.gitignore`
- **Issue:** `MuQ-Eval/` shows as an untracked directory in `git status`. The `.gitignore` already
  has `third_party/` but `MuQ-Eval/` is a top-level directory, not inside `third_party/`. If this
  is a third-party clone it should be gitignored.
- **Fix:** Add `MuQ-Eval/` to `.gitignore`.
- **Status:** Fixed in this review.

#### I5 — `AGENTS.md` untracked
- **File:** `AGENTS.md` (repo root)
- **Issue:** `AGENTS.md` is untracked. CLAUDE.md references it as the team roster document.
  It should be committed.
- **Fix:** `git add AGENTS.md && git commit`.
- **Status:** Staged but not committed — left for author to commit with appropriate context.

#### I6 — `docs/` and `tests/` directories untracked
- **Issue:** Both `docs/` and `tests/` are untracked. These contain the review report itself
  and unit tests respectively — both should be committed.
- **Status:** Will be committed as part of this review.

#### I7 — `team/status.json` may contain runtime state
- **File:** `team/status.json`
- **Issue:** `team/status.json` appears to contain runtime/ephemeral team state. If it does,
  it should either be gitignored or kept as a template. Check its contents.
- **Status:** Flagged for author review.

#### I8 — `prepare_dataset.py` has hardcoded default `--target-sr 192000`
- **File:** `prepare_dataset.py:78–79`
- **Issue:** The CLI default is `192000` but Phase 0 trains at 96kHz (set in `phase0.yaml` and
  `models/constants.py`). The default should match `OUTPUT_SAMPLE_RATE` from constants, or at
  least the script should import and display that constant in the help text.
- **Fix:** Import `OUTPUT_SAMPLE_RATE` and use it as the default.
- **Status:** Flagged for follow-up.

---

## Git / Workflow Observations

- All recent commits are on `develop` branch — correct per CLAUDE.md.
- Commit messages are descriptive and follow imperative style — good.
- No secrets, `.env`, or credentials found in tracked files.
- `.gitignore` covers `.venv/`, `checkpoints/`, `*.pt`, `*.wav/flac/mp3`, Terraform state,
  and IDE directories. Comprehensive.
- `third_party/` is gitignored but `MuQ-Eval/` (a third-party clone) lives at root — needs gitignore entry.

---

## Fixes Applied in This Review

1. Removed unused `import math` from `models/mastering_losses.py`
2. Removed unused `import torch.nn.functional as F` from `train.py`
3. Replaced hardcoded `48000` with `INPUT_SAMPLE_RATE` in `train.py:226`
4. Added `GRAD_CLIP_MAX_NORM` and `LR_SCHEDULER_GAMMA` constants to `models/constants.py`
5. Used `GRAD_CLIP_MAX_NORM` and `LR_SCHEDULER_GAMMA` in `train.py`
6. Used `INPUT_SAMPLE_RATE` constants for ffmpeg `-ar` argument in `analyzer_ui.py`
7. Added `MuQ-Eval/` to `.gitignore`
8. Removed redundant inner `import os` from `train.py` validation function

## Remaining Items (for team follow-up)

- **Adam:** Address W4 (val n_fft / roll_percent / val_samples magic numbers in train.py)
- **Pierce:** Address I1/I2 (inline imports in analyzer_ui.py) and remaining W8 items
- **Cain/Adam:** Address I8 (prepare_dataset.py default sample rate mismatch)
- **Author:** Review team/status.json for gitignore eligibility (I7)
- **Author:** Commit AGENTS.md, docs/, tests/ directories (I5/I6)
