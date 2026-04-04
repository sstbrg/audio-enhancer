#!/bin/bash
# Full automated training workflow for Vast.ai instances.
#
# Workflow:
#   1. Pull latest code from git (develop branch)
#   2. Pull datasets from Google Drive
#   3. Pull latest checkpoint from Drive (for resume)
#   4. Start training with auto-resume
#   5. Push checkpoints to Drive every CHECKPOINT_PUSH_INTERVAL_MIN minutes
#   6. Auto-shutdown when training completes or hits max hours
#
# Usage:
#   bash infra/auto_train.sh [--no-pull-data] [--no-shutdown] [--max-hours N] [--phase=N]
#
# --phase=N : training phase (0 or 1, default: 0)
#             Phase 0: 48kHz->96kHz super-resolution
#             Phase 1: degradation restoration (fine-tune from Phase 0 checkpoint)
#
# Environment variables (override defaults):
#   MAX_HOURS             — max training hours (default: 5)
#   CHECKPOINT_PUSH_INTERVAL_MIN — how often to push checkpoints to Drive (default: 30)
#   IDLE_SHUTDOWN_MIN     — minutes of idle (no train.py) before shutdown (default: 15)
set -euo pipefail

PROJECT_DIR="/workspace/audio-enhancer"
VENV="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
TRAIN_LOG="$LOG_DIR/train_$(date '+%Y%m%d_%H%M%S').log"

# Configurable via env vars
MAX_HOURS="${MAX_HOURS:-5}"
CHECKPOINT_PUSH_INTERVAL_MIN="${CHECKPOINT_PUSH_INTERVAL_MIN:-30}"
IDLE_SHUTDOWN_MIN="${IDLE_SHUTDOWN_MIN:-15}"

# Parse flags
OPT_NO_PULL_DATA=0
OPT_NO_SHUTDOWN=0
PHASE=0
for arg in "$@"; do
    case "$arg" in
        --no-pull-data)  OPT_NO_PULL_DATA=1 ;;
        --no-shutdown)   OPT_NO_SHUTDOWN=1 ;;
        --phase=*)       PHASE="${arg#--phase=}" ;;
        --max-hours)     shift; MAX_HOURS="$1" ;;
        --max-hours=*)   MAX_HOURS="${arg#--max-hours=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Phase-dependent paths
CONFIG="$PROJECT_DIR/configs/phase${PHASE}.yaml"
DATA_DIR="$PROJECT_DIR/datasets/phase${PHASE}_combined"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/phase${PHASE}"

# Google Drive checkpoint path:
# Phase 0 keeps original location for backward compatibility with existing checkpoints.
# Phase N (N>0) uses a subdirectory to separate from Phase 0.
if [ "$PHASE" = "0" ]; then
    GDRIVE_CHECKPOINTS="audio-enhancer-datasets/checkpoints"
else
    GDRIVE_CHECKPOINTS="audio-enhancer-datasets/checkpoints/phase${PHASE}"
fi

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TRAIN_LOG"; }
ok()   { log "OK  $*"; }
err()  { log "ERR $*" >&2; }
die()  { err "$*"; exit 1; }

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
log "=== Auto Train (Phase ${PHASE}) — $(date) ==="
log "Config:  $CONFIG"
log "Data:    $DATA_DIR"
log "Ckpts:   $CHECKPOINT_DIR"
log "MAX_HOURS=$MAX_HOURS  PUSH_INTERVAL=${CHECKPOINT_PUSH_INTERVAL_MIN}min  IDLE_SHUTDOWN=${IDLE_SHUTDOWN_MIN}min"
log "Log: $TRAIN_LOG"

# ── Activate venv ─────────────────────────────────────────────────────────────

if [ ! -f "$VENV/bin/activate" ]; then
    die "venv not found at $VENV — run setup-vastai.sh first"
fi
source "$VENV/bin/activate"
log "venv: activated"

# ── [1] Pull latest code ──────────────────────────────────────────────────────

log "[1/5] Pulling latest code..."
cd "$PROJECT_DIR"
if git remote get-url origin &>/dev/null; then
    git fetch origin develop --quiet
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/develop)
    if [ "$LOCAL" != "$REMOTE" ]; then
        log "  Updating to latest develop (was $LOCAL, now $REMOTE)"
        git pull origin develop --quiet
        ok "  Code updated"
        # Reinstall requirements if they changed
        if git diff HEAD@{1} HEAD -- requirements.txt | grep -q '^[+-]'; then
            log "  requirements.txt changed, reinstalling..."
            pip install -r requirements.txt -q 2>&1 | tail -3
        fi
    else
        ok "  Already up to date ($LOCAL)"
    fi
else
    log "  No git remote configured, skipping pull"
fi

# ── [2] Pull datasets ─────────────────────────────────────────────────────────

if (( OPT_NO_PULL_DATA )); then
    log "[2/5] Skipping dataset pull (--no-pull-data)"
else
    log "[2/5] Pulling datasets from Drive..."
    if rclone lsf "gdrive:audio-enhancer-datasets/" --max-depth 1 &>/dev/null 2>&1; then
        python "$PROJECT_DIR/infra/datasets.py" pull 2>&1 | tee -a "$TRAIN_LOG"
        ok "Datasets ready"
    else
        log "  rclone/Drive not accessible — using existing local data"
    fi
fi

# ── [3] Pull latest checkpoint ────────────────────────────────────────────────

log "[3/5] Checking for checkpoint on Drive..."
RESUME_FLAG=""
if rclone lsf "gdrive:$GDRIVE_CHECKPOINTS/" --max-depth 1 &>/dev/null 2>&1; then
    log "  Pulling checkpoints..."
    rclone copy "gdrive:$GDRIVE_CHECKPOINTS/" "$CHECKPOINT_DIR/" \
        --progress --retries 5 2>&1 | tee -a "$TRAIN_LOG" || true
    if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
        CKPT_SIZE=$(du -sh "$CHECKPOINT_DIR/latest.pt" | cut -f1)
        ok "  latest.pt ($CKPT_SIZE) — will resume from this checkpoint"
        RESUME_FLAG="--resume $CHECKPOINT_DIR/latest.pt"
    fi
else
    log "  Drive not accessible, checking local checkpoints..."
    if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
        ok "  Found local latest.pt — will resume"
        RESUME_FLAG="--resume $CHECKPOINT_DIR/latest.pt"
    else
        log "  No checkpoint found — starting from scratch"
    fi
fi

# ── [4] Validate data dir ─────────────────────────────────────────────────────

log "[4/5] Validating data dir..."
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    die "Data dir $DATA_DIR is empty or missing. Run datasets.py pull first."
fi
FILE_COUNT=$(find "$DATA_DIR" -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.mp3" \) 2>/dev/null | wc -l)
log "  Found $FILE_COUNT audio files in $DATA_DIR"
if (( FILE_COUNT == 0 )); then
    die "No audio files found in $DATA_DIR"
fi
ok "Data dir OK"

# ── [5] Start training ────────────────────────────────────────────────────────

log "[5/5] Starting training..."
log "  Config: $CONFIG"
log "  Data: $DATA_DIR"
log "  Checkpoints: $CHECKPOINT_DIR"
log "  Resume flag: ${RESUME_FLAG:-none}"
log "  Max hours: $MAX_HOURS"

TRAIN_CMD=(
    python "$PROJECT_DIR/train.py"
    --data_dir "$DATA_DIR"
    --config "$CONFIG"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --max-hours "$MAX_HOURS"
)
if [ -n "$RESUME_FLAG" ]; then
    # shellcheck disable=SC2206
    TRAIN_CMD+=($RESUME_FLAG)
fi

log "  Command: ${TRAIN_CMD[*]}"

# ── Periodic checkpoint push in background ────────────────────────────────────

push_checkpoints() {
    local interval_s=$(( CHECKPOINT_PUSH_INTERVAL_MIN * 60 ))
    while true; do
        sleep "$interval_s"
        if ls "$CHECKPOINT_DIR"/*.pt &>/dev/null 2>&1; then
            log "[push] Pushing checkpoints to Drive..."
            rclone copy "$CHECKPOINT_DIR/" "gdrive:$GDRIVE_CHECKPOINTS/" \
                --retries 3 --quiet 2>>"$TRAIN_LOG" && \
                log "[push] Checkpoints pushed OK" || \
                log "[push] Checkpoint push failed (will retry next interval)"
        fi
    done
}

if rclone lsf "gdrive:audio-enhancer-datasets/" --max-depth 1 &>/dev/null 2>&1; then
    log "Starting periodic checkpoint push every ${CHECKPOINT_PUSH_INTERVAL_MIN}min..."
    push_checkpoints &
    PUSH_PID=$!
else
    PUSH_PID=""
    log "Drive not accessible — checkpoint auto-push disabled"
fi

# ── Run training ──────────────────────────────────────────────────────────────

TRAIN_EXIT=0
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$TRAIN_LOG" || TRAIN_EXIT=$?

# Stop checkpoint push loop
if [ -n "$PUSH_PID" ]; then
    kill "$PUSH_PID" 2>/dev/null || true
fi

# Final checkpoint push
if rclone lsf "gdrive:audio-enhancer-datasets/" --max-depth 1 &>/dev/null 2>&1; then
    if ls "$CHECKPOINT_DIR"/*.pt &>/dev/null 2>&1; then
        log "Final checkpoint push..."
        rclone copy "$CHECKPOINT_DIR/" "gdrive:$GDRIVE_CHECKPOINTS/" \
            --progress --retries 5 2>&1 | tee -a "$TRAIN_LOG"
        ok "Final checkpoints pushed to Drive"
    fi
fi

if [ "$TRAIN_EXIT" -eq 0 ]; then
    log "Training completed successfully (exit 0)"
else
    err "Training exited with code $TRAIN_EXIT"
fi

# ── Auto-shutdown ─────────────────────────────────────────────────────────────

if (( OPT_NO_SHUTDOWN )); then
    log "Auto-shutdown disabled (--no-shutdown). Instance will keep running."
    log "Destroy manually: vast destroy instance <id>"
else
    log "Shutting down instance in 60s (Ctrl+C to cancel, or pass --no-shutdown)..."
    sleep 60
    log "Shutting down now. Goodbye."
    shutdown -h now
fi
