#!/bin/bash
# Startup script for the audio-enhancer training VM.
# Runs once on first boot to set up the environment.
set -euo pipefail

MARKER="/opt/audio-enhancer-setup-done"
if [ -f "$MARKER" ]; then
  echo "Setup already completed, skipping."
  exit 0
fi

exec > >(tee /var/log/audio-enhancer-setup.log) 2>&1
echo "=== Audio Enhancer VM Setup ==="

# Wait for NVIDIA driver
echo "Waiting for GPU driver..."
for i in $(seq 1 30); do
  nvidia-smi && break
  sleep 10
done

# System packages
apt-get update
apt-get install -y --no-install-recommends libsndfile1 ffmpeg tmux htop git

# Create project directory
PROJECT_DIR="/home/stas/audio-enhancer"

# Clone from GitHub
echo "Cloning repo..."
git clone https://github.com/sstbrg/audio-enhancer.git "$PROJECT_DIR" || true

# Download dataset from GCS
BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/data-bucket" \
  -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

if [ -n "$BUCKET" ]; then
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

# Auto-shutdown cron: power off if idle (no python training process) for 15 min
cat > /usr/local/bin/auto-shutdown.sh <<'SHUTDOWN'
#!/bin/bash
if ! pgrep -f "python.*train.py" > /dev/null; then
  IDLE_FILE="/tmp/gpu-idle-since"
  if [ ! -f "$IDLE_FILE" ]; then
    date +%s > "$IDLE_FILE"
  else
    IDLE_SINCE=$(cat "$IDLE_FILE")
    NOW=$(date +%s)
    if (( NOW - IDLE_SINCE > 900 )); then
      logger "audio-enhancer: No training running for 15min, shutting down"
      shutdown -h now
    fi
  fi
else
  rm -f /tmp/gpu-idle-since
fi
SHUTDOWN
chmod +x /usr/local/bin/auto-shutdown.sh
echo "*/5 * * * * root /usr/local/bin/auto-shutdown.sh" > /etc/cron.d/auto-shutdown

touch "$MARKER"
echo "=== Setup complete ==="
