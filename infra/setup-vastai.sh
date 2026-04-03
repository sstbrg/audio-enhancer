#!/bin/bash
# One-command setup for a fresh Vast.ai instance.
# Usage: ssh into instance, then:
#   bash /workspace/audio-enhancer/infra/setup-vastai.sh [--from-gdrive]
#
# --from-gdrive: download datasets from Google Drive (requires rclone config)
# Without flag: download from Zenodo (first time only)
set -euo pipefail

PROJECT_DIR="/workspace/audio-enhancer"
RAW_DIR="$PROJECT_DIR/datasets/raw"
EXTRACT_DIR="$PROJECT_DIR/datasets/extracted"
GDRIVE_PATH="audio-enhancer-datasets"

echo "=== Audio Enhancer — Vast.ai Setup ==="

# System deps
echo "[1/5] Installing system packages..."
apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg tmux unzip rclone > /dev/null 2>&1

# Python deps
echo "[2/5] Setting up Python environment..."
cd "$PROJECT_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 -q 2>&1 | tail -1
pip install -r requirements.txt -q 2>&1 | tail -1
pip install numpy soundfile librosa scipy tqdm pyyaml tensorboard gdown -q 2>&1 | tail -1

# Download datasets
echo "[3/5] Getting datasets..."
mkdir -p "$RAW_DIR" "$EXTRACT_DIR"

if [ "${1:-}" = "--from-gdrive" ]; then
    echo "  Downloading from Google Drive..."
    if [ ! -f ~/.config/rclone/rclone.conf ]; then
        echo "  ERROR: rclone not configured. Run 'rclone config' first."
        exit 1
    fi
    rclone copy "gdrive:$GDRIVE_PATH/" "$RAW_DIR/" --progress
else
    echo "  Downloading from Zenodo (first time)..."
    cd "$RAW_DIR"
    [ -f EG-IPT.zip ] || wget -q --show-progress -O EG-IPT.zip \
        "https://zenodo.org/records/15205644/files/EG-IPT.zip?download=1" &
    [ -f musdb18hq.zip ] || wget -q --show-progress -O musdb18hq.zip \
        "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1" &
    wait
fi

echo "  Raw datasets:"
ls -lh "$RAW_DIR/"

# Extract
echo "[4/5] Extracting..."
if [ ! -d "$EXTRACT_DIR/EG-IPT" ]; then
    unzip -q "$RAW_DIR/EG-IPT.zip" -d "$EXTRACT_DIR/EG-IPT"
    echo "  Extracted EG-IPT"
else
    echo "  EG-IPT already extracted"
fi

if [ ! -d "$EXTRACT_DIR/musdb18hq" ]; then
    unzip -q "$RAW_DIR/musdb18hq.zip" -d "$EXTRACT_DIR/musdb18hq"
    echo "  Extracted MUSDB18-HQ"
else
    echo "  MUSDB18-HQ already extracted"
fi

# Symlink for training
mkdir -p "$PROJECT_DIR/datasets/phase0_combined"
ln -sf "$EXTRACT_DIR/EG-IPT" "$PROJECT_DIR/datasets/phase0_combined/EG-IPT"
ln -sf "$EXTRACT_DIR/musdb18hq" "$PROJECT_DIR/datasets/phase0_combined/musdb18hq"

# Verify
echo "[5/5] Verifying..."
source "$PROJECT_DIR/.venv/bin/activate"
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
echo ""
echo "=== Setup complete ==="
echo "To train:"
echo "  cd $PROJECT_DIR && source .venv/bin/activate"
echo "  python train.py --data_dir datasets/phase0_combined --config configs/phase0.yaml --checkpoint_dir checkpoints/phase0 --max-hours 5"
