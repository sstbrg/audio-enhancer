#!/bin/bash
# Startup script for the audio-enhancer training VM.
# Runs once on first boot to set up the environment.
# Data lives on a persistent disk at /mnt/data that survives VM preemption.
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
apt-get install -y --no-install-recommends libsndfile1 ffmpeg tmux htop git rclone

# Detect the SSH user from instance metadata
SSH_USER=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/ssh-keys" \
  -H "Metadata-Flavor: Google" 2>/dev/null | head -1 | cut -d: -f1)
SSH_USER="${SSH_USER:-stas}"
PROJECT_DIR="/home/${SSH_USER}/audio-enhancer"
DATA_MOUNT="/mnt/data"
DATA_DEVICE="/dev/disk/by-id/google-audio-enhancer-data"

# ── Mount persistent data disk ────────────────────────────────────────────────

echo "Mounting persistent data disk..."
mkdir -p "$DATA_MOUNT"

# Format only if no filesystem exists (first time)
if ! blkid "$DATA_DEVICE" &>/dev/null; then
  echo "New disk detected, formatting as ext4..."
  mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0 "$DATA_DEVICE"
fi

mount -o discard,defaults "$DATA_DEVICE" "$DATA_MOUNT"

# Add to fstab for remounts
if ! grep -q "audio-enhancer-data" /etc/fstab; then
  echo "$DATA_DEVICE $DATA_MOUNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab
fi

# Create directory structure on data disk
mkdir -p "$DATA_MOUNT/datasets/phase0_combined"
mkdir -p "$DATA_MOUNT/checkpoints"
mkdir -p "$DATA_MOUNT/rclone-config"

chown -R "${SSH_USER}:${SSH_USER}" "$DATA_MOUNT"

# ── Clone repo ────────────────────────────────────────────────────────────────

echo "Cloning repo..."
git clone https://github.com/sstbrg/audio-enhancer.git "$PROJECT_DIR" 2>/dev/null || {
  cd "$PROJECT_DIR" && git pull origin develop || true
}
cd "$PROJECT_DIR"
git checkout develop 2>/dev/null || true

# Symlink data directories to persistent disk
ln -sfn "$DATA_MOUNT/datasets" "$PROJECT_DIR/datasets"
ln -sfn "$DATA_MOUNT/checkpoints" "$PROJECT_DIR/checkpoints"

# ── Python environment ────────────────────────────────────────────────────────

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ── rclone setup ──────────────────────────────────────────────────────────────
# rclone config is stored on the persistent disk so it survives VM recreation.
# If no config exists yet, the user must run: rclone config
# and set up a "gdrive" remote pointing to audio-enhancer-datasets/.

RCLONE_CONF_DIR="/home/${SSH_USER}/.config/rclone"
mkdir -p "$RCLONE_CONF_DIR"

# Symlink rclone config to persistent disk
if [ -f "$DATA_MOUNT/rclone-config/rclone.conf" ]; then
  ln -sf "$DATA_MOUNT/rclone-config/rclone.conf" "$RCLONE_CONF_DIR/rclone.conf"
  echo "rclone config restored from persistent disk"
else
  # First time: create a placeholder. User needs to run 'rclone config' manually.
  echo "# Run 'rclone config' to set up gdrive remote" > "$DATA_MOUNT/rclone-config/rclone.conf"
  ln -sf "$DATA_MOUNT/rclone-config/rclone.conf" "$RCLONE_CONF_DIR/rclone.conf"
  echo "WARNING: rclone not configured yet. Run 'rclone config' after SSH-ing in."
fi

# ── Pull datasets from Google Drive (if rclone is configured) ─────────────────

GDRIVE_PATH="audio-enhancer-datasets"
if rclone lsf "gdrive:$GDRIVE_PATH/" --max-depth 1 &>/dev/null; then
  echo "Pulling datasets from Google Drive..."

  # Pull each dataset directory
  for ds in EG-IPT MUSDB18-HQ VCTK-96kHz MusicNet; do
    if rclone lsf "gdrive:$GDRIVE_PATH/$ds/" --max-depth 1 &>/dev/null; then
      echo "  Syncing $ds..."
      rclone sync "gdrive:$GDRIVE_PATH/$ds/" "$DATA_MOUNT/datasets/phase0_combined/$ds/" \
        --transfers=8 --checkers=16 --progress 2>&1 | tail -1 || true
    fi
  done

  # Pull checkpoints
  echo "Pulling checkpoints from Google Drive..."
  rclone copy "gdrive:$GDRIVE_PATH/checkpoints/" "$DATA_MOUNT/checkpoints/" \
    --progress 2>&1 | tail -1 || true

  echo "Google Drive sync complete"
else
  echo "WARNING: Cannot access gdrive:$GDRIVE_PATH/ — rclone not configured or no access."
  echo "Run 'rclone config' and then 'python infra/datasets.py pull' manually."
fi

# ── Fix ownership ─────────────────────────────────────────────────────────────

chown -R "${SSH_USER}:${SSH_USER}" "$PROJECT_DIR"
chown -R "${SSH_USER}:${SSH_USER}" "$DATA_MOUNT"
chown -R "${SSH_USER}:${SSH_USER}" "$RCLONE_CONF_DIR"

# ── Auto-shutdown cron: power off if idle for 15 min ──────────────────────────

cat > /usr/local/bin/auto-shutdown.sh <<'SHUTDOWN'
#!/bin/bash
# Don't shut down while training or while rclone is syncing
if pgrep -f "python.*train.py" > /dev/null || pgrep -f "rclone" > /dev/null; then
  rm -f /tmp/gpu-idle-since
  exit 0
fi
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
SHUTDOWN
chmod +x /usr/local/bin/auto-shutdown.sh
echo "*/5 * * * * root /usr/local/bin/auto-shutdown.sh" > /etc/cron.d/auto-shutdown

touch "$MARKER"
echo "=== Setup complete ==="
