# Inference Guide — enhance.py

Enhance audio files to 96kHz/24-bit using the trained GAN.

---

## Quick Start

```bash
source .venv/bin/activate

# Enhance a single file
python enhance.py input.mp3 -o output.wav --gan_checkpoint checkpoints/phase0/latest.pt

# Enhance a directory of files
python enhance.py /path/to/music/ -o /path/to/output/ --gan_checkpoint checkpoints/phase0/latest.pt
```

---

## Input Format Requirements

Supported formats: `.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.m4a`, `.wma`, `.aiff`, `.aif`, `.opus`, `.alac`, `.webm`

Any sample rate and channel count is accepted. The pipeline handles resampling and channel conversion internally.

---

## Sample Rate Handling Logic

The pipeline behavior depends on input sample rate:

| Input SR | Apollo | AudioSR | GAN | Notes |
|----------|--------|---------|-----|-------|
| Any (lossy) | Yes | if <48kHz | Yes | Full pipeline |
| 44.1kHz | Yes | Yes | Yes | Most common CD-quality input |
| 48kHz | Yes | Skipped | Yes | Already at GAN input SR |
| 96kHz | Yes | Skipped | No | Already at target SR — GAN skipped. Phase 1: mastering-only mode. |
| >96kHz | Yes | Skipped | No | Above target SR — passes through unchanged. |

AudioSR stage is skipped when `sr >= 48000` (the condition in `_run_audiosr`).
GAN stage is skipped when `sr >= 96000` — at that point the signal is already at target SR.

Stage details:

1. **Apollo** (Stage 1): Restores lossy artifacts. Internally resamples to 44.1kHz (Apollo requirement), then passes back to the pipeline at 44.1kHz.

2. **AudioSR** (Stage 2): Neural bandwidth extension to 48kHz. Writes a temporary WAV file, calls AudioSR, returns 48kHz output.

3. **GAN** (Stage 3): Runs the trained generator. Input is resampled to 48kHz if not already. Each channel is processed independently (generator is mono). Output is clamped to `[-1, 1]`.

If no GAN checkpoint is provided, the pipeline falls back to high-quality Kaiser sinc resampling for the final SR conversion.

---

## Output Format

- **Default:** 24-bit WAV
- **Channels:** same as input
- **Sample rate:** 96kHz (set in config `output.sample_rate`)
- **Filename:** `{original_stem}_enhanced.wav` when processing a directory

To output FLAC:

```bash
python enhance.py input.wav -o output.flac --format flac
```

---

## Command-Line Reference

```
python enhance.py INPUT -o OUTPUT [options]

Positional:
  input                 Input audio file or directory

Required:
  -o, --output          Output file or directory

Options:
  --config              Config file (default: configs/default.yaml)
  --gan_checkpoint      Path to trained GAN .pt checkpoint
  --no-apollo           Skip Apollo lossy restoration stage
  --no-audiosr          Skip AudioSR neural super-resolution stage
  --no-gan              Skip GAN upsampling (use Kaiser resampling instead)
  --format {wav,flac}   Output format (default: wav)
  --device {cuda,cpu}   Compute device (default: auto-detect)
```

---

## Usage Examples

### Lossless input (skip Apollo)

If your input is already lossless (WAV, FLAC, AIFF), you can skip the Apollo stage:

```bash
python enhance.py input.flac -o output.wav \
  --no-apollo \
  --gan_checkpoint checkpoints/phase0/latest.pt
```

### Batch processing a music library

```bash
python enhance.py /home/user/music/ -o /home/user/music_enhanced/ \
  --gan_checkpoint checkpoints/phase0/latest.pt \
  --no-apollo \
  --format flac
```

Files that fail (corrupted, unsupported codec) are skipped with an error message; processing continues.

### GAN-only (bypass Apollo and AudioSR)

Useful for testing the GAN in isolation on already-48kHz audio:

```bash
python enhance.py input_48k.wav -o output.wav \
  --no-apollo \
  --no-audiosr \
  --gan_checkpoint checkpoints/phase0/latest.pt
```

### CPU inference (no GPU)

```bash
python enhance.py input.wav -o output.wav \
  --device cpu \
  --gan_checkpoint checkpoints/phase0/latest.pt
```

CPU inference is slow for long files. For batch processing, a GPU is recommended.

---

## Chunked Processing for Long Files

The GAN processes long audio in overlapping 10-second chunks (at 48kHz input) to manage GPU memory. Chunks overlap by 4800 samples (0.1s) with linear crossfade at the boundaries to avoid discontinuities.

Apollo uses 30-second chunks with 1-second overlap.

This happens automatically — no configuration needed.

---

## Analyzer GUI

For interactive use, the Gradio web UI provides both enhancement and quality analysis:

```bash
python analyzer_ui.py                # English
python analyzer_ui.py --lang ru      # Russian
```

Open `http://localhost:7860`.

- **Enhance tab:** Upload a file, optionally provide a checkpoint, click Enhance.
- **Analyze tab:** Upload a file, run quality metrics (PAM, Audiobox, MuQ-Eval, and reference-based metrics if a reference file is provided).

---

## Loading a Checkpoint Manually

If you need to use the generator in your own code:

```python
import torch
from models.generator import Generator

# NOTE: weights_only=False is required because the checkpoint embeds a config dict.
# Only load checkpoints from trusted sources (your own training runs, official releases).
# Once the checkpoint format is updated to store config separately (JSON sidecar),
# this can be changed to weights_only=True for improved security.
ckpt = torch.load("checkpoints/phase0/latest.pt", map_location="cpu", weights_only=False)
gen_cfg = ckpt["config"]["gan"]["generator"]

generator = Generator(
    channels=gen_cfg["channels"],
    upsample_rates=gen_cfg["upsample_rates"],
    upsample_kernel_sizes=gen_cfg["upsample_kernel_sizes"],
    resblock_kernel_sizes=gen_cfg["resblock_kernel_sizes"],
    resblock_dilation_sizes=gen_cfg["resblock_dilation_sizes"],
)

# If the checkpoint was saved from a torch.compile-wrapped model without
# _unwrap_state_dict, keys will have an _orig_mod. prefix. Strip it:
gen_state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["generator"].items()}
generator.load_state_dict(gen_state)
generator.eval()
generator.remove_weight_norm()  # Fuses weight norm for faster inference
```

Checkpoints saved by the current codebase (after the `_unwrap_state_dict` fix) do not have this prefix and load directly.

Input: `(batch, 1, samples)` float32 tensor at 48kHz  
Output: `(batch, 1, samples * 2)` float32 tensor at 96kHz

---

## Troubleshooting

**Apollo not installed:**
```
WARNING: Apollo not installed. Skipping lossy restoration.
```
Use `--no-apollo` to skip, or install via `pip install look2hear`.

**AudioSR not installed:**
```
WARNING: AudioSR not installed. Skipping neural SR.
```
Use `--no-audiosr` to skip, or install via `pip install audiosr`.

**GAN checkpoint not found / not provided:**
The pipeline falls back to Kaiser sinc resampling for the final SR step. Output quality will be lower than with a trained GAN.

**OOM during GAN:**
The chunked processing is automatic, but for very large files on low-VRAM machines, you can reduce chunk size by editing the `chunk_samples` calculation in `enhance.py:_run_gan`. The default is `INPUT_SAMPLE_RATE * GAN_CHUNK_SECONDS` (10 seconds at 48kHz = 480000 samples). Reducing to 5 seconds should halve peak GPU memory for the GAN stage.

**Output sounds distorted / clipped:**
Check that the input file is not already clipping (`peak > 1.0`). The GAN output is clamped to `[-1, 1]` but pre-existing distortion will pass through.
