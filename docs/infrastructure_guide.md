# Infrastructure Guide

This guide covers the Vast.ai training infrastructure for the audio-enhancer project.

## Architecture overview

| Layer | Tool | Purpose |
|-------|------|---------|
| Compute | Vast.ai RTX 3090 spot | GPU training (~$0.13–0.22/hr) |
| Code | GitHub (sstbrg/audio-enhancer) | Source of truth, branch: develop |
| Data | Google Drive via rclone | Dataset storage (gdrive:audio-enhancer-datasets/) |
| Checkpoints | Google Drive (same remote) | gdrive:audio-enhancer-datasets/checkpoints/ |
| Monitoring | TensorBoard via SSH tunnel | Port 6006 |

## Quick start: spin up a new instance

### 1. Launch instance on Vast.ai

Pick an RTX 3090 spot instance with at least 20GB VRAM and 100GB disk.
SSH key: `~/.ssh/id_ed25519`

```bash
# Connect
ssh -i ~/.ssh/id_ed25519 -p PORT root@sshN.vast.ai
```

### 2. Clone the repo

```bash
git clone https://github.com/sstbrg/audio-enhancer /workspace/audio-enhancer
cd /workspace/audio-enhancer
git checkout develop
```

### 3. Run setup

**Option A — fresh start (no Drive data):**
```bash
bash infra/setup-vastai.sh
```

**Option B — pull datasets and checkpoints from Drive:**
```bash
bash infra/setup-vastai.sh --from-gdrive
```

**Option C — full auto (pull Drive data + start training immediately):**
```bash
bash infra/setup-vastai.sh --from-gdrive --resume
```

The setup script:
- Checks GPU availability and disk space (fails fast if insufficient)
- Creates `.venv` with PyTorch (CUDA 12.8) and project dependencies
- Validates the venv works (CUDA available, correct Python version)
- Checks rclone config if `--from-gdrive` is set
- Downloads or pulls datasets
- Pulls checkpoints from Drive (with `--from-gdrive`)
- Prints the exact training command to run

### 4. Start the automated training workflow

```bash
bash infra/auto_train.sh
```

This script handles the complete lifecycle:
1. Pulls latest code from `develop` branch
2. Pulls datasets from Drive (skip with `--no-pull-data`)
3. Pulls latest checkpoint (auto-resume if it exists)
4. Validates the data directory
5. Starts training with `--max-hours` limit
6. Pushes checkpoints to Drive every 30 minutes (background)
7. Final checkpoint push when training ends
8. Auto-shuts down the instance (skip with `--no-shutdown`)

**Options:**
```bash
bash infra/auto_train.sh --no-pull-data    # skip dataset pull (data already local)
bash infra/auto_train.sh --no-shutdown     # keep instance alive after training
bash infra/auto_train.sh --max-hours=10    # override training duration
```

**Environment variable overrides:**
```bash
MAX_HOURS=10 CHECKPOINT_PUSH_INTERVAL_MIN=15 bash infra/auto_train.sh
```

## Dataset management

All datasets are managed via `infra/datasets.py`.

```bash
source .venv/bin/activate

# Show dashboard: local size, Drive availability, status
python infra/datasets.py status

# Live-refresh dashboard every 5s
python infra/datasets.py watch

# Download from source URLs
python infra/datasets.py download           # all
python infra/datasets.py download egipt     # one dataset

# Upload to Google Drive
python infra/datasets.py upload             # all
python infra/datasets.py upload musdb       # one dataset

# Pull from Google Drive (to local)
python infra/datasets.py pull               # all
python infra/datasets.py pull egipt         # one dataset

# Verify local dataset integrity (size + archive check + file count)
python infra/datasets.py verify             # all
python infra/datasets.py verify maestro     # one dataset
```

### Dataset catalog

| ID | Name | Size | Source |
|----|------|------|--------|
| egipt | EG-IPT 96kHz guitar | 22GB | Zenodo |
| musdb | MUSDB18-HQ 44.1kHz | 22GB | Zenodo |
| vctk | VCTK 96kHz speech | 20GB | Edinburgh |
| gtsinger | GTSinger 48kHz vocals | 5GB | HuggingFace |
| maestro | MAESTRO v3 piano | 101GB | Google |
| musicnet | MusicNet classical | 11GB | Zenodo |
| moisesdb | MoisesDB stems | 88GB | manual |
| medleydb | MedleyDB v2 | 30GB | manual |

Manual datasets require requesting access at the listed URLs and downloading yourself.

### rclone configuration

rclone uses a custom OAuth client (GCP project: stoked-mapper-258810).
Config lives at `~/.config/rclone/rclone.conf`.

To set up rclone on a new machine:
```bash
rclone config  # follow prompts, select "Google Drive", use existing client ID
```

Copy the existing config from your local machine:
```bash
scp ~/.config/rclone/rclone.conf root@sshN.vast.ai:~/.config/rclone/rclone.conf
```

## Manual training

```bash
cd /workspace/audio-enhancer && source .venv/bin/activate

# Fresh start
python train.py \
  --data_dir datasets/phase0_combined \
  --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 \
  --max-hours 5

# Resume from checkpoint
python train.py \
  --data_dir datasets/phase0_combined \
  --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 \
  --max-hours 5 \
  --resume checkpoints/phase0/latest.pt
```

## TensorBoard monitoring

Open a local SSH tunnel and view TensorBoard in your browser:

```bash
# On your local machine:
ssh -i ~/.ssh/id_ed25519 -N -L 6006:localhost:6006 -p PORT root@sshN.vast.ai
# Then open: http://localhost:6006
```

## Checkpoint management

Checkpoints are saved to `checkpoints/phase0/` on the instance and pushed to:
`gdrive:audio-enhancer-datasets/checkpoints/`

`auto_train.sh` pushes every 30 minutes and does a final push on exit.

To manually push:
```bash
rclone copy checkpoints/phase0/ gdrive:audio-enhancer-datasets/checkpoints/ --progress
```

To manually pull on a new instance:
```bash
rclone copy gdrive:audio-enhancer-datasets/checkpoints/ checkpoints/phase0/ --progress
```

## Idle auto-shutdown

Vast.ai instances cost money even when idle. The cron-based auto-shutdown
(configured during setup) terminates the instance if `train.py` has not been
running for 15 minutes.

To disable auto-shutdown temporarily (e.g. for a long dataset download):
```bash
crontab -r   # remove all crons — remember to re-add or just destroy manually
```

## Cost tracking

- RTX 3090 spot: ~$0.13–0.22/hr
- A 5-hour training run costs ~$0.65–$1.10
- Google Drive storage: free tier (15GB free; datasets on paid plan)
- Always destroy instances after use: `vast destroy instance <id>`

## GCP / Terraform (kept for future use)

`infra/main.tf` and `infra/deploy.sh` contain GCP infrastructure definitions.
GCP GPU quota was denied; these are kept in case quota is approved later.
Do not use for active training — use Vast.ai instead.
