#!/bin/bash
# Startup script for the audio-enhancer training VM.
# Runs once on first boot to set up the environment.
set -euo pipefail

MARKER="/opt/audio-enhancer-setup-done"
if [ -f "$MARKER" ]; then
  echo "Setup already completed, skipping."
  exit 0
fi

echo "=== Audio Enhancer VM Setup ==="

# Wait for NVIDIA driver
echo "Waiting for GPU driver..."
for i in $(seq 1 30); do
  nvidia-smi && break
  sleep 10
done

# System packages
apt-get update
apt-get install -y --no-install-recommends libsndfile1 ffmpeg tmux htop

# Create project directory
PROJECT_DIR="/home/stas/audio-enhancer"
mkdir -p "$PROJECT_DIR"

# Clone the repo (will be replaced by actual repo URL after GitHub push)
# For now, code is synced via GCS
BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/data-bucket" \
  -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

if [ -n "$BUCKET" ]; then
  echo "Downloading code from gs://$BUCKET/code/ ..."
  gsutil -m rsync -r "gs://$BUCKET/code/" "$PROJECT_DIR/"

  echo "Downloading dataset from gs://$BUCKET/datasets/ ..."
  mkdir -p "$PROJECT_DIR/datasets/phase0_combined"
  gsutil -m rsync -r "gs://$BUCKET/datasets/" "$PROJECT_DIR/datasets/phase0_combined/"
fi

# Python environment
cd "$PROJECT_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Fix ownership
chown -R stas:stas "$PROJECT_DIR"

touch "$MARKER"
echo "=== Setup complete ==="
