# Audio Enhancer

GAN-based audio super-resolution: upscales 48kHz → 96kHz / 24-bit.
Target quality: hi-fi playback, Dire Straits-level mastering.

## Project structure

```
train.py              # Training script (GAN with AMP, torch.compile)
enhance.py            # Inference: any audio → 96kHz/24-bit wav
evaluate.py           # CLI model evaluation
analyze.py            # CLI audio quality analysis
analyzer_ui.py        # Gradio web UI (Enhance + Analyze tabs, i18n EN/RU)
prepare_dataset.py    # Resample audio to target SR for training
models/
  constants.py        # All defaults: sample rates, architecture params, slopes
  generator.py        # HiFi-GAN generator (48kHz → 96kHz, 2x upsample)
  discriminator.py    # Multi-period (2,3,5,7,11,17,23) + multi-scale (3)
  losses.py           # STFT, mel spectrogram, adversarial, feature matching
  mastering_losses.py # Perceptual STFT, stereo, dynamics, encodec, CLAP, audiobox
data/
  dataset.py          # AudioSRDataset: creates (48kHz, 96kHz) pairs on the fly
metrics/
  evaluate.py         # SI-SNR, SDR, CDPAM, ViSQOL, Audiobox, PAM, MuQ-Eval, chroma/MFCC/onset
  music_analysis.py   # Genre, mood, instruments, key, BPM (Essentia, CLAP, MERT)
configs/
  phase0.yaml         # Phase 0: 96kHz target, batch 8, checkpoint every epoch
  default.yaml        # Default config
locales/
  en.json             # English UI strings
  ru.json             # Russian UI strings
infra/
  datasets.py         # Dataset manager: download/upload/pull/status dashboard
  datasets.sh         # Shell version (deprecated, use .py)
  setup-vastai.sh     # One-command Vast.ai instance setup
  deploy.sh           # GCP deployment helper (kept for future use)
  main.tf             # Terraform config (GCP, kept for future use)
third_party/          # (gitignored) PAM, MuQ-Eval clones
```

## Training

Training runs on Vast.ai (RTX 3090 spot, ~$0.13-0.22/hr).

```bash
# On Vast.ai instance:
cd /workspace/audio-enhancer && source .venv/bin/activate
python train.py \
  --data_dir datasets/phase0_combined \
  --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 \
  --max-hours 5
```

Optimizations: AMP (mixed precision), torch.compile, cudnn.benchmark, persistent DataLoader workers.
Checkpoints save every epoch. TensorBoard via `ssh -N -L 6006:localhost:6006`.
Resume: `--resume checkpoints/phase0/latest.pt`

## Current status

- Phase 0 epoch 0 complete, checkpoint saved to Google Drive
- Checkpoint: `gdrive:audio-enhancer-datasets/checkpoints/checkpoint_0000.pt`
- Losses at end of epoch 0: d≈4.2, g≈35 (stable, encodec spikes resolved)
- Training optimizations (AMP, torch.compile) committed but not yet tested in training
- Vast.ai auto-shutdown after 15min idle (cron checks for train.py process)

## Datasets

Stored on Google Drive (`gdrive:audio-enhancer-datasets/`), pulled to Vast.ai instances.
Manage with: `python infra/datasets.py status|download|upload|pull|watch`

On Drive: EG-IPT (22GB), MUSDB18-HQ (22GB), VCTK 96kHz (20GB), MusicNet (11GB)
Downloading: MAESTRO (120GB), GTSinger (30GB) — may need re-download on next instance
Manual: MoisesDB (requested at developer.moises.ai), MedleyDB (requested at medleydb.weebly.com)

rclone config uses custom OAuth client ID (GCP project stoked-mapper-258810).

## Analyzer GUI

```bash
python analyzer_ui.py           # English, http://localhost:7860
python analyzer_ui.py --lang ru  # Russian
```

Two tabs: Enhance (apply model) + Analyze (quality metrics).
All strings from locale files, all styles in CSS. Supports wav/flac/mp3/ogg/webm.

## Commands

- ALWAYS use `.venv` — never install packages globally or with --user
- ALWAYS work on `develop` branch, not `main`
- Python 3.12, PyTorch 2.11, CUDA 12.8/13.0
- No magic numbers — all constants in models/constants.py or configs/

## Infrastructure

- **Compute**: Vast.ai (RTX 3090 spot). SSH: `ssh -i ~/.ssh/id_ed25519 -p PORT root@sshN.vast.ai`
- **Code**: GitHub public repo sstbrg/audio-enhancer, branch: develop
- **Data**: Google Drive via rclone (remote name: `gdrive:`)
- **GCP**: project stoked-mapper-258810, GPU quota denied (can retry after 48h)
- **Monitoring**: TensorBoard via SSH tunnel, `infra/datasets.py watch` for data

## Architecture

- Generator: HiFi-GAN, 11M params, single 2x upsample (48k→96k), skip connection
- Discriminators: MPD (periods 2,3,5,7,11,17,23) + MSD (3 scales, spectral norm on first)
- Training losses: adversarial + feature matching + multi-res STFT + mel + mastering
- Mastering losses: perceptual STFT (auraloss), stereo image, dynamics (K-weighted), encodec embedding
- CLAP + Audiobox: validation only (not differentiable)
- Gradient clipping: max_norm=10.0, LeakyReLU slope=0.1 (from constants)

## Next steps

1. Test AMP + torch.compile training (committed, not yet run)
2. Evaluate epoch 0 checkpoint quality with analyzer GUI
3. Continue training (more epochs, possibly larger batch with AMP)
4. Add 44.1kHz/16-bit input degradation to dataset (simulate CD quality input)
5. Analyzer: add "upscale potential" assessment — detect sample rate ceiling, codec artifacts, bit depth headroom, spectral rolloff vs nyquist gap
6. MAESTRO dataset: re-download on Vast.ai (101/120GB incomplete, auto-retry script in place)
7. MoisesDB: user requested access at developer.moises.ai — download when link arrives
8. MedleyDB: user requested access at medleydb.weebly.com — download when link arrives
9. Phase 1: degradation pipeline (codec artifacts, bad EQ, compression, stereo damage)
10. Vast.ai instance may still be running (auto-shutdown was disabled for MAESTRO download) — check and destroy if done
