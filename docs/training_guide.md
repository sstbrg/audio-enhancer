# Training Guide

GAN-based audio super-resolution: 48kHz → 96kHz/24-bit.

---

## Prerequisites

- Vast.ai RTX 3090 spot instance (~$0.13-0.22/hr)
- SSH key at `~/.ssh/id_ed25519`
- rclone configured with `gdrive:` remote pointing to your Google Drive
- GitHub access to `sstbrg/audio-enhancer`

---

## 1. Provision a Vast.ai Instance

1. Go to vast.ai, filter by RTX 3090, CUDA 12.x, PyTorch template.
2. Note the SSH command: `ssh -i ~/.ssh/id_ed25519 -p PORT root@sshN.vast.ai`
3. SSH in and clone the repo:

```bash
cd /workspace
git clone https://github.com/sstbrg/audio-enhancer.git
cd audio-enhancer
git checkout develop
```

---

## 2. Instance Setup

Run the one-command setup script. Pass `--from-gdrive` to pull datasets from Google Drive (fastest for repeat instances):

```bash
# First time: download from Zenodo (free, no auth needed)
bash infra/setup-vastai.sh

# Subsequent times: pull existing datasets from Google Drive (faster)
bash infra/setup-vastai.sh --from-gdrive
```

The script:
1. Installs system packages (libsndfile1, ffmpeg, tmux, rclone)
2. Creates `.venv` and installs all Python dependencies
3. Downloads and extracts EG-IPT and MUSDB18-HQ datasets
4. Creates `datasets/phase0_combined/` symlinks for training
5. Verifies PyTorch and CUDA are working

If `--from-gdrive` is used, rclone must be configured first (`rclone config`).

---

## 3. Dataset Preparation

### Available datasets

| Dataset | Size | Quality | Notes |
|---------|------|---------|-------|
| EG-IPT | 22GB | 96kHz/24b | Guitar. Real hi-res source. |
| MUSDB18-HQ | 22GB | 44.1kHz | Multitrack music stems. |
| VCTK 96kHz | 20GB | 96kHz | Speech. Real hi-res source. |
| MusicNet | 11GB | 44.1kHz | Classical music. |
| GTSinger | 30GB | 48kHz/24b | Vocals. |
| MAESTRO | 120GB | 44.1kHz | Piano. Large. |
| MoisesDB | 88GB | various | Music stems. Manual request required. |
| MedleyDB | ~30GB | various | Professional recordings. Manual request required. |

### Manage datasets with the dataset manager

```bash
source .venv/bin/activate

# Show dashboard (local + Drive status)
python infra/datasets.py status

# Live dashboard (refreshes every 5s)
python infra/datasets.py watch

# Download all available datasets
python infra/datasets.py download

# Download a specific dataset
python infra/datasets.py download egipt

# Upload local datasets to Google Drive (for sharing between instances)
python infra/datasets.py upload

# Pull all datasets from Google Drive to local
python infra/datasets.py pull
```

### Prepare the combined dataset directory

After downloading, create symlinks for the training data directory:

```bash
mkdir -p datasets/phase0_combined
# Symlink each dataset subdirectory into phase0_combined
ln -sf /workspace/audio-enhancer/datasets/raw/EG-IPT /workspace/audio-enhancer/datasets/phase0_combined/EG-IPT
ln -sf /workspace/audio-enhancer/datasets/raw/musdb18hq /workspace/audio-enhancer/datasets/phase0_combined/musdb18hq
# Add more symlinks as datasets are downloaded
```

The `AudioSRDataset` scans recursively, so nesting structure does not matter.

---

## 4. Configuration

Training is controlled by YAML config files in `configs/`. The active phase uses `configs/phase0.yaml`.

### Key config parameters

```yaml
output:
  sample_rate: 96000        # Target output sample rate
  bit_depth: 24             # Output bit depth

gan:
  generator:
    channels: 512           # Base channel width; reduce to 256 for lower VRAM
    upsample_rates: [2]     # [2] = 2x upsample (48kHz → 96kHz)

  training:
    batch_size: 8           # RTX 3090: 8 is stable at 96kHz; reduce if OOM
    epochs: 200
    segment_length: 16384   # ~0.34s at 48kHz per training sample
    lambda_fm: 2.0          # Feature matching weight
    lambda_stft: 45.0       # Multi-resolution STFT weight
    lambda_mel: 45.0        # Mel spectrogram weight
    checkpoint_interval: 1  # Save checkpoint every N epochs

    quality_sampling:       # Weighted dataset sampling
      weight_hires: 7.0     # 96kHz sources (EG-IPT, VCTK)
      weight_midres: 2.0    # 48kHz sources (GTSinger)
      weight_standard: 1.0  # 44.1kHz sources (MUSDB, etc.)

    mastering:
      lambda_perceptual_stft: 45.0   # A-weighted mel STFT
      lambda_stereo: 10.0            # Mid/side stereo preservation
      lambda_dynamics: 5.0           # Crest factor + LUFS
      lambda_encodec: 0.01           # EnCodec embedding loss
```

All architecture constants without a YAML override come from `models/constants.py`. Never hardcode numbers — add them to constants.py or the config file.

---

## 5. Start Training

### Fresh training run

```bash
cd /workspace/audio-enhancer
source .venv/bin/activate

python train.py \
  --data_dir datasets/phase0_combined \
  --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 \
  --max-hours 5
```

`--max-hours` gracefully stops training after the time limit and saves a final checkpoint. Useful for Vast.ai spot instances.

### Run in tmux (recommended for long runs)

```bash
tmux new -s train
# inside tmux:
source .venv/bin/activate
python train.py --data_dir datasets/phase0_combined --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 --max-hours 5
# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
```

### Resume from checkpoint

```bash
python train.py \
  --data_dir datasets/phase0_combined \
  --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 \
  --resume checkpoints/phase0/latest.pt \
  --max-hours 5
```

The resume logic loads model weights, optimizer states, scheduler states, and AMP scaler states. If the architecture changed since the checkpoint was saved, optimizer state is reinitialized automatically (a warning is printed).

Note: checkpoint loading happens before `torch.compile`. If you try to load a compiled model's checkpoint into an uncompiled model (or vice versa), key names will mismatch. The code handles this by loading before compilation.

---

## 6. Monitor Training

### TensorBoard (via SSH tunnel)

On your local machine:

```bash
ssh -N -L 6006:localhost:6006 -i ~/.ssh/id_ed25519 -p PORT root@sshN.vast.ai
```

Then open `http://localhost:6006` in your browser.

Logged metrics:
- `loss/generator` — total generator loss
- `loss/discriminator` — total discriminator loss
- `loss/stft` — multi-resolution STFT loss
- `loss/mel` — mel spectrogram loss
- `loss/fm` — feature matching loss
- `loss/mastering_total` — combined mastering loss
- `loss/mastering_perceptual_stft` — per-component mastering losses
- `loss/mastering_stereo`, `loss/mastering_dynamics`, `loss/mastering_encodec`

Validation metrics (logged every `checkpoint_interval` epochs):
- `val/si_snr`, `val/sdr` — signal quality
- `val/cdpam` — perceptual distance
- `val/audiobox_pq` — production quality
- `val/chroma_similarity`, `val/mfcc_similarity` — content preservation
- `val/sr_hf_energy_enh`, `val/sr_rolloff_enh_hz` — bandwidth extension metrics

### Expected loss trajectory

| Epoch | Discriminator | Generator | Notes |
|-------|--------------|-----------|-------|
| 0 | ~4.2 | ~35 | Stable; encodec spikes resolved |
| 10+ | decreasing | decreasing | Should see gradual improvement |
| 100+ | ~2-3 | ~15-25 | Model producing plausible HF content |

If generator loss explodes (>100), check gradient clipping is active (`max_norm=10.0`). If NaN appears, check the mastering loss component — the `MasteringLoss` wrapper will log which component produced the NaN and skip it.

---

## 7. Checkpoint Management

Checkpoints are saved to `--checkpoint_dir` as:
- `checkpoint_NNNN.pt` — per-epoch checkpoint (N = epoch number, zero-padded to 4 digits)
- `latest.pt` — always points to the most recent checkpoint

### Upload to Google Drive after training

```bash
python infra/datasets.py upload checkpoints
# Or manually:
rclone copy checkpoints/phase0/latest.pt gdrive:audio-enhancer-datasets/checkpoints/
```

### Download a checkpoint on a new instance

```bash
rclone copy gdrive:audio-enhancer-datasets/checkpoints/ checkpoints/phase0/
```

---

## 8. Evaluate a Checkpoint

Run the CLI evaluator on a test file:

```bash
python evaluate.py --checkpoint checkpoints/phase0/latest.pt --input test.wav
```

Or use the Gradio analyzer UI:

```bash
python analyzer_ui.py
# Open http://localhost:7860
# Use the Enhance tab to apply the model
# Use the Analyze tab to compute quality metrics
```

---

## 9. Shut Down the Instance

The Vast.ai instance has an auto-shutdown cron job: if no `train.py` process is running for 15 minutes, the instance shuts down. To disable this (e.g., during dataset download):

```bash
crontab -l          # view current cron jobs
crontab -r          # remove all (disables auto-shutdown)
```

To manually destroy from the Vast.ai dashboard, or:

```bash
vastai destroy instance INSTANCE_ID
```

Always upload your checkpoints to Google Drive before destroying the instance.

---

## Troubleshooting

**OOM (out of memory):**
- Reduce `batch_size` in config (try 4)
- Reduce `channels` from 512 to 256
- Reduce `segment_length` from 16384 to 8192

**NaN in losses:**
- Check TensorBoard for which loss component went NaN
- EnCodec embedding loss is the most common source; reduce `lambda_encodec` to 0.001
- Perceptual STFT and mel spectrogram losses disable autocast internally (safe)
- Increase gradient clip (or check if generator output diverged)

**Checkpoint resume fails with "key mismatch":**
- If you changed the architecture (added/removed layers), optimizer state will be incompatible — this is expected and handled; a warning is printed and optimizer is reinitialized
- Generator weights load with `strict=False` to allow partial loading

**torch.compile fails:**
- The code catches compilation errors and falls back to eager mode; training continues
- This is expected on some CUDA/PyTorch version combinations

**Dataset not found:**
- Verify `--data_dir` contains `.wav`, `.flac`, `.aiff`, or `.aif` files
- Check symlinks in `datasets/phase0_combined/` point to actual extracted directories
