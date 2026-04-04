#!/bin/bash
# One-command setup for a fresh Vast.ai instance.
# Usage: ssh into instance, then:
#   bash /workspace/audio-enhancer/infra/setup-vastai.sh [--from-gdrive] [--resume] [--phase=N]
#
# --from-gdrive : pull datasets and checkpoints from Google Drive (requires rclone config)
# --resume      : auto-start training from latest checkpoint after setup
# --phase=N     : training phase (0 or 1, default: 0)
#                 Phase 0: 48kHz->96kHz super-resolution
#                 Phase 1: degradation restoration (fine-tune from Phase 0 checkpoint)
set -euo pipefail

PROJECT_DIR="/workspace/audio-enhancer"
VENV="$PROJECT_DIR/.venv"
RAW_DIR="$PROJECT_DIR/datasets/raw"
EXTRACT_DIR="$PROJECT_DIR/datasets/extracted"
GDRIVE_PATH="audio-enhancer-datasets"
MIN_DISK_GB=50
MIN_CUDA_GB=20

# Parse flags
OPT_GDRIVE=0
OPT_RESUME=0
PHASE=0
for arg in "$@"; do
    case "$arg" in
        --from-gdrive) OPT_GDRIVE=1 ;;
        --resume)      OPT_RESUME=1 ;;
        --phase=*)     PHASE="${arg#--phase=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Phase-dependent paths
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/phase${PHASE}"
CONFIG="$PROJECT_DIR/configs/phase${PHASE}.yaml"
DATA_DIR="$PROJECT_DIR/datasets/phase${PHASE}_combined"

# Google Drive checkpoint path:
# Phase 0 keeps original location for backward compatibility with existing checkpoints.
# Phase N (N>0) uses a subdirectory to separate from Phase 0.
if [ "$PHASE" = "0" ]; then
    GDRIVE_CHECKPOINTS="$GDRIVE_PATH/checkpoints"
else
    GDRIVE_CHECKPOINTS="$GDRIVE_PATH/checkpoints/phase${PHASE}"
fi

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] OK  $*"; }
err()  { echo "[$(date '+%H:%M:%S')] ERR $*" >&2; }
die()  { err "$*"; exit 1; }

echo "=== Audio Enhancer — Vast.ai Setup ==="
log "Project: $PROJECT_DIR"
log "Phase:   $PHASE"
log "Config:  $CONFIG"
log "Data:    $DATA_DIR"
log "Ckpts:   $CHECKPOINT_DIR"
log "Flags:   gdrive=$OPT_GDRIVE resume=$OPT_RESUME"

# ── [0] Health checks ─────────────────────────────────────────────────────────

log "[0/6] Running health checks..."

# GPU check
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Is this a GPU instance?"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_MEM_GB=$(( GPU_MEM_MB / 1024 ))
log "GPU: $GPU_NAME (${GPU_MEM_GB}GB)"
if (( GPU_MEM_GB < MIN_CUDA_GB )); then
    err "Warning: GPU has only ${GPU_MEM_GB}GB VRAM, recommend >= ${MIN_CUDA_GB}GB for batch_size=8"
fi

# Disk space check
DISK_FREE_GB=$(df -BG "$PROJECT_DIR" | tail -1 | awk '{print $4}' | tr -d 'G')
log "Disk free: ${DISK_FREE_GB}GB"
if (( DISK_FREE_GB < MIN_DISK_GB )); then
    die "Insufficient disk space: ${DISK_FREE_GB}GB free, need at least ${MIN_DISK_GB}GB"
fi

ok "Health checks passed"

# ── [1] System deps ───────────────────────────────────────────────────────────

log "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq libsndfile1 ffmpeg tmux unzip rclone bc > /dev/null 2>&1
ok "System packages installed"

# ── [2] Python venv ───────────────────────────────────────────────────────────

log "[2/6] Setting up Python environment..."
cd "$PROJECT_DIR"

# Detect SSL cert path — pytorch docker images use /opt/conda, standard Ubuntu uses /etc/ssl
SSL_CERT=""
for cert_path in /opt/conda/ssl/cert.pem /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-bundle.crt; do
    if [ -f "$cert_path" ]; then
        SSL_CERT="$cert_path"
        break
    fi
done
if [ -n "$SSL_CERT" ]; then
    export SSL_CERT_FILE="$SSL_CERT"
    log "SSL cert: $SSL_CERT"
fi

# Create venv only if it doesn't exist or is broken
if [ -f "$VENV/bin/python" ]; then
    if "$VENV/bin/python" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        ok "Existing venv looks good, skipping creation"
    else
        log "Existing venv has wrong Python version, recreating..."
        rm -rf "$VENV"
    fi
fi

if [ ! -f "$VENV/bin/python" ]; then
    # Use --system-site-packages if PyTorch is already installed system-wide (e.g. pytorch docker image)
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        python3 -m venv "$VENV" --system-site-packages
        log "Created venv with system-site-packages (reusing pre-installed PyTorch)"
    else
        python3 -m venv "$VENV"
        log "Created new venv at $VENV"
    fi
fi

source "$VENV/bin/activate"
pip install --upgrade pip -q

# Install PyTorch (CUDA 12.8) only if not already available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "Installing PyTorch with CUDA 12.8..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 -q 2>&1 | tail -3
else
    ok "PyTorch already installed with CUDA ($(python -c 'import torch; print(torch.__version__)'))"
fi

# Install project requirements
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt" -q 2>&1 | tail -3
fi
# Core deps that may not be in requirements.txt
pip install numpy soundfile librosa scipy tqdm pyyaml tensorboard auraloss encodec -q 2>&1 | tail -1

# Validate venv
python -c "
import torch, torchaudio, numpy, soundfile, yaml
print(f'  PyTorch {torch.__version__}')
print(f'  torchaudio {torchaudio.__version__}')
cuda_ok = torch.cuda.is_available()
print(f'  CUDA: {cuda_ok}')
if cuda_ok:
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  VRAM: {mem_gb:.1f}GB')
assert cuda_ok, 'CUDA not available after setup!'
"
ok "Python environment validated"

# ── [3] rclone config check ───────────────────────────────────────────────────

if (( OPT_GDRIVE )); then
    log "[3/6] Checking rclone config..."
    if [ ! -f ~/.config/rclone/rclone.conf ]; then
        die "rclone not configured. Run 'rclone config' to set up gdrive remote first."
    fi
    if ! rclone lsf "gdrive:$GDRIVE_PATH/" --max-depth 1 &>/dev/null; then
        die "Cannot access gdrive:$GDRIVE_PATH/ — check rclone config and Drive permissions."
    fi
    ok "rclone gdrive access confirmed"
else
    log "[3/6] Skipping rclone check (--from-gdrive not set)"
fi

# ── [4] Datasets ──────────────────────────────────────────────────────────────

log "[4/6] Setting up datasets..."
mkdir -p "$RAW_DIR" "$EXTRACT_DIR" "$DATA_DIR"

if (( OPT_GDRIVE )); then
    log "  Pulling datasets from Google Drive..."
    python "$PROJECT_DIR/infra/datasets.py" pull
elif [ "$PHASE" = "0" ]; then
    log "  Downloading Phase 0 core datasets from Zenodo..."
    cd "$RAW_DIR"
    [ -f EG-IPT.zip ] || wget -q --show-progress -O EG-IPT.zip \
        "https://zenodo.org/records/15205644/files/EG-IPT.zip?download=1" &
    [ -f musdb18hq.zip ] || wget -q --show-progress -O musdb18hq.zip \
        "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1" &
    wait
else
    log "  Phase ${PHASE}: datasets must be pulled from Google Drive (--from-gdrive required)"
    log "  Re-run with: bash $0 --phase=${PHASE} --from-gdrive"
    die "Phase ${PHASE} datasets not available without --from-gdrive"
fi

# Extract datasets (Phase 0 only — Phase 1 datasets arrive pre-extracted via Drive)
if [ "$PHASE" = "0" ]; then
    if [ ! -d "$EXTRACT_DIR/EG-IPT" ] && [ -f "$RAW_DIR/EG-IPT.zip" ]; then
        log "  Extracting EG-IPT..."
        unzip -q "$RAW_DIR/EG-IPT.zip" -d "$EXTRACT_DIR/EG-IPT"
        ok "  EG-IPT extracted"
    else
        [ -d "$EXTRACT_DIR/EG-IPT" ] && ok "  EG-IPT already extracted"
    fi

    if [ ! -d "$EXTRACT_DIR/musdb18hq" ] && [ -f "$RAW_DIR/musdb18hq.zip" ]; then
        log "  Extracting MUSDB18-HQ..."
        unzip -q "$RAW_DIR/musdb18hq.zip" -d "$EXTRACT_DIR/musdb18hq"
        ok "  MUSDB18-HQ extracted"
    else
        [ -d "$EXTRACT_DIR/musdb18hq" ] && ok "  MUSDB18-HQ already extracted"
    fi

    # Symlinks for training
    for name in EG-IPT musdb18hq; do
        if [ -d "$EXTRACT_DIR/$name" ] && [ ! -L "$DATA_DIR/$name" ]; then
            ln -sf "$EXTRACT_DIR/$name" "$DATA_DIR/$name"
            log "  Linked $name into phase${PHASE}_combined"
        fi
    done
fi

ok "Datasets ready"

# ── [5] Checkpoints ───────────────────────────────────────────────────────────

log "[5/6] Checking checkpoints..."
mkdir -p "$CHECKPOINT_DIR"

if (( OPT_GDRIVE )); then
    log "  Pulling checkpoints from Google Drive..."
    if rclone lsf "gdrive:$GDRIVE_CHECKPOINTS/" --max-depth 1 &>/dev/null; then
        rclone copy "gdrive:$GDRIVE_CHECKPOINTS/" "$CHECKPOINT_DIR/" --progress
        ok "  Checkpoints pulled"
        # Show what we have
        if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
            CKPT_SIZE=$(du -sh "$CHECKPOINT_DIR/latest.pt" | cut -f1)
            ok "  latest.pt ($CKPT_SIZE) ready for resume"
        fi
    else
        log "  No checkpoints on Drive yet (first run)"
    fi
else
    log "  Skipping checkpoint pull (--from-gdrive not set)"
fi

ok "Checkpoint setup done"

# ── [6] Final validation ──────────────────────────────────────────────────────

log "[6/6] Final validation..."
source "$VENV/bin/activate"
python -c "
import torch
import sys
from pathlib import Path

project = Path('$PROJECT_DIR')
phase = '$PHASE'
ok = True

# Check CUDA
if not torch.cuda.is_available():
    print('  FAIL: CUDA not available')
    ok = False
else:
    print(f'  PASS: CUDA — {torch.cuda.get_device_name(0)}')

# Check data dir has content
data_dir = project / f'datasets/phase{phase}_combined'
if data_dir.exists():
    entries = list(data_dir.iterdir())
    print(f'  PASS: data dir has {len(entries)} entries')
else:
    print(f'  WARN: datasets/phase{phase}_combined not found — training will need data')

# Check config
config = project / f'configs/phase{phase}.yaml'
if config.exists():
    print(f'  PASS: config {config.name} found')
else:
    print(f'  FAIL: config {config} missing')
    ok = False

# Check for latest checkpoint
ckpt = project / f'checkpoints/phase{phase}/latest.pt'
if ckpt.exists():
    print(f'  PASS: latest.pt checkpoint found — training will resume')
else:
    if phase == '0':
        print('  INFO: No latest.pt — training will start from scratch')
    else:
        print(f'  WARN: No Phase {phase} latest.pt — expected pretrained checkpoint for fine-tuning')

if not ok:
    sys.exit(1)
"

echo ""
echo "=== Setup complete (Phase ${PHASE}) ==="
echo ""
echo "To start training manually:"
echo "  cd $PROJECT_DIR && source .venv/bin/activate"
RESUME_FLAG=""
if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
    RESUME_FLAG=" --resume $CHECKPOINT_DIR/latest.pt"
fi
echo "  python train.py \\"
echo "    --data_dir $DATA_DIR \\"
echo "    --config $CONFIG \\"
echo "    --checkpoint_dir $CHECKPOINT_DIR \\"
echo "    --max-hours 5$RESUME_FLAG"
echo ""
echo "Or run the full automated workflow:"
echo "  bash $PROJECT_DIR/infra/auto_train.sh --phase=${PHASE}"

# ── Auto-resume training if requested ─────────────────────────────────────────

if (( OPT_RESUME )); then
    log "Auto-resume requested, launching training in tmux session 'train'..."
    tmux new-session -d -s train \
        "cd $PROJECT_DIR && source .venv/bin/activate && bash infra/auto_train.sh --phase=${PHASE} 2>&1 | tee /tmp/train.log"
    log "Training started in tmux session 'train'. Attach with: tmux attach -t train"
fi
