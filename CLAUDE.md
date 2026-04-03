# Audio Enhancer

GAN-based audio super-resolution model: upscales 48kHz → 192kHz / 32-bit.
Target quality: hi-fi playback, Dire Straits-level mastering.

## Project structure

```
train.py              # Main training script (GAN: generator + MPD + MSD)
enhance.py            # Inference / enhancement script
evaluate.py           # Model evaluation
prepare_dataset.py    # Resample audio collection to 192kHz for training
models/
  generator.py        # HiFi-GAN style generator (48kHz → 192kHz)
  discriminator.py    # Multi-period + multi-scale discriminators
  losses.py           # Adversarial, feature matching, mel, multi-res STFT losses
  mastering_losses.py # Perceptual/stereo/dynamics/encodec losses
data/
  dataset.py          # AudioSRDataset: creates (48kHz, 192kHz) pairs on the fly
metrics/
  evaluate.py         # SI-SNR, SDR, reference-based metrics
configs/
  phase0.yaml         # Quick validation run (200 epochs, batch 4)
  default.yaml        # Default training config
infra/
  setup-vastai.sh     # One-command Vast.ai instance setup
  deploy.sh           # GCP deployment helper (kept for future use)
  main.tf             # Terraform config (GCP, kept for future use)
```

## Training

Training runs on Vast.ai (RTX 3090, spot). Datasets stored on Google Drive for reuse.

```bash
# On Vast.ai instance:
cd /workspace/audio-enhancer
source .venv/bin/activate
python train.py \
  --data_dir datasets/phase0_combined \
  --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 \
  --max-hours 5
```

Checkpoints save every 10 epochs. TensorBoard logs go to `checkpoints/*/logs/`.

## Datasets

Phase 0 uses EG-IPT (96kHz guitar) + MUSDB18-HQ (44.1kHz mixed music).
See DATASETS.md for full catalog and download URLs.

Raw zips stored in Google Drive under `audio-enhancer-datasets/` for reuse across instances.

## Commands

- Always use `.venv` — never install packages globally
- Python 3.11+, PyTorch 2.x, CUDA 12.8
- `source .venv/bin/activate` before any pip/python command

## Infrastructure

- **Compute**: Vast.ai GPU rental (spot RTX 3090, ~$0.13-0.17/hr)
- **Code**: GitHub public repo (sstbrg/audio-enhancer, branch: main)
- **Data**: Google Drive (rclone remote "gdrive:")
- **GCP**: project stoked-mapper-258810 exists but GPU quota denied; Terraform configs kept in infra/ for later
- **Monitoring**: TensorBoard via SSH tunnel (`ssh -N -L 6006:localhost:6006`)

## Key architectural decisions

- Generator: HiFi-GAN style with 2x2 upsample rates (4x total: 48k→192k)
- Discriminators: multi-period (periods 2,3,5,7,11) + multi-scale (3 scales)
- Losses: adversarial + feature matching + multi-res STFT + mel + mastering (perceptual STFT, stereo, dynamics, encodec)
- Gradient clipping: max_norm=10.0 on both generator and discriminators
- Spot instance training: checkpoints every 10 epochs to survive preemption
