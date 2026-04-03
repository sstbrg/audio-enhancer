#!/bin/bash
# Dataset manager for audio-enhancer.
# Downloads, uploads to Google Drive, pulls to new instances.
#
# Usage:
#   ./infra/datasets.sh status          — show dashboard
#   ./infra/datasets.sh download        — download all datasets
#   ./infra/datasets.sh download egip   — download one dataset
#   ./infra/datasets.sh upload          — upload all to Google Drive
#   ./infra/datasets.sh upload musdb    — upload one to Drive
#   ./infra/datasets.sh pull            — pull all from Google Drive
#   ./infra/datasets.sh pull egip       — pull one from Drive
#   ./infra/datasets.sh watch           — live dashboard (refreshes every 5s)
set -euo pipefail

GDRIVE_DIR="audio-enhancer-datasets"
RAW_DIR="${DATASETS_RAW:-$(cd "$(dirname "$0")/.." && pwd)/datasets/raw}"
mkdir -p "$RAW_DIR"

# ── Dataset catalog ───────────────────────────────────────────────────────────

declare -A DS_NAME DS_URL DS_SIZE DS_FILE
DATASETS=(egipt musdb vctk moisesdb)

DS_NAME[egipt]="EG-IPT (96kHz guitar)"
DS_URL[egipt]="https://zenodo.org/records/15205644/files/EG-IPT.zip?download=1"
DS_SIZE[egipt]=22
DS_FILE[egipt]="EG-IPT.zip"

DS_NAME[musdb]="MUSDB18-HQ (44.1kHz music)"
DS_URL[musdb]="https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
DS_SIZE[musdb]=22
DS_FILE[musdb]="musdb18hq.zip"

DS_NAME[vctk]="VCTK 96kHz (speech)"
DS_URL[vctk]="https://datashare.ed.ac.uk/bitstream/handle/10283/2774/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y"
DS_SIZE[vctk]=11
DS_FILE[vctk]="vctk96k.zip"

DS_NAME[moisesdb]="MoisesDB (music stems)"
DS_URL[moisesdb]="hf:wearemusicai/moisesdb"
DS_SIZE[moisesdb]=25
DS_FILE[moisesdb]="moisesdb"

# ── Helpers ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

human_size() {
  local bytes=$1
  if (( bytes >= 1073741824 )); then
    printf "%.1fG" "$(echo "$bytes / 1073741824" | bc -l)"
  elif (( bytes >= 1048576 )); then
    printf "%.0fM" "$(echo "$bytes / 1048576" | bc -l)"
  else
    printf "%.0fK" "$(echo "$bytes / 1024" | bc -l)"
  fi
}

file_size_bytes() {
  local f="$1"
  if [ -f "$f" ]; then
    stat --format='%s' "$f" 2>/dev/null || echo 0
  elif [ -d "$f" ]; then
    du -sb "$f" 2>/dev/null | cut -f1 || echo 0
  else
    echo 0
  fi
}

is_downloading() {
  local f="${DS_FILE[$1]}"
  pgrep -f "wget.*$f" > /dev/null 2>&1 || pgrep -f "curl.*$f" > /dev/null 2>&1 || pgrep -f "huggingface.*moisesdb" > /dev/null 2>&1
}

is_on_drive() {
  local f="${DS_FILE[$1]}"
  rclone lsf "gdrive:$GDRIVE_DIR/$f" > /dev/null 2>&1
}

# ── Status / Dashboard ────────────────────────────────────────────────────────

show_status() {
  echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}║              Audio Enhancer — Dataset Dashboard                 ║${NC}"
  echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════╣${NC}"
  printf  "${BOLD}║ %-5s │ %-26s │ %8s │ %6s │ %6s ║${NC}\n" "ID" "Dataset" "Local" "Drive" "Status"
  echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════╣${NC}"

  local total_local=0
  local total_expected=0

  for ds in "${DATASETS[@]}"; do
    local name="${DS_NAME[$ds]}"
    local file="$RAW_DIR/${DS_FILE[$ds]}"
    local expected_gb="${DS_SIZE[$ds]}"
    local expected_bytes=$((expected_gb * 1073741824))
    total_expected=$((total_expected + expected_bytes))

    # Local status
    local local_bytes
    local_bytes=$(file_size_bytes "$file")
    total_local=$((total_local + local_bytes))
    local local_str
    if [ "$local_bytes" -eq 0 ]; then
      local_str="-"
    else
      local_str=$(human_size "$local_bytes")
    fi

    # Drive status
    local drive_str
    if is_on_drive "$ds"; then
      drive_str="${GREEN}  ✓${NC}"
    else
      drive_str="${DIM}  -${NC}"
    fi

    # Overall status
    local status_str
    if is_downloading "$ds"; then
      local pct=0
      if [ "$expected_bytes" -gt 0 ] && [ "$local_bytes" -gt 0 ]; then
        pct=$((local_bytes * 100 / expected_bytes))
      fi
      status_str="${YELLOW}⬇ ${pct}%${NC}"
    elif [ "$local_bytes" -gt 0 ] && [ "$local_bytes" -ge $((expected_bytes * 9 / 10)) ]; then
      status_str="${GREEN}ready${NC}"
    elif [ "$local_bytes" -gt 0 ]; then
      status_str="${RED}partial${NC}"
    else
      status_str="${DIM}none${NC}"
    fi

    # Truncate name to 26 chars
    name="${name:0:26}"
    printf "║ ${CYAN}%-5s${NC} │ %-26s │ %8s │ %b │ %b  ║\n" \
      "$ds" "$name" "$local_str" "$drive_str" "$status_str"
  done

  echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════╣${NC}"

  local total_local_str
  total_local_str=$(human_size "$total_local")
  local total_expected_str
  total_expected_str=$(human_size "$total_expected")
  local disk_free
  disk_free=$(df -h "$RAW_DIR" 2>/dev/null | tail -1 | awk '{print $4}')

  printf "║ ${BOLD}Total local:${NC} %-10s ${BOLD}Expected:${NC} %-10s ${BOLD}Disk free:${NC} %-7s ║\n" \
    "$total_local_str" "$total_expected_str" "$disk_free"
  echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
}

# ── Download ──────────────────────────────────────────────────────────────────

download_one() {
  local ds="$1"
  local url="${DS_URL[$ds]}"
  local file="$RAW_DIR/${DS_FILE[$ds]}"

  echo -e "${BLUE}Downloading ${DS_NAME[$ds]}...${NC}"

  if [[ "$url" == hf:* ]]; then
    local repo="${url#hf:}"
    source "$(dirname "$0")/../.venv/bin/activate" 2>/dev/null || true
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', repo_type='dataset', local_dir='$file')
"
  else
    wget --tries=5 --continue -q --show-progress -O "$file" "$url"
  fi

  echo -e "${GREEN}✓ ${DS_NAME[$ds]} downloaded${NC}"
}

download_all() {
  for ds in "${DATASETS[@]}"; do
    local file="$RAW_DIR/${DS_FILE[$ds]}"
    local expected_bytes=$(( ${DS_SIZE[$ds]} * 1073741824 ))
    local current_bytes
    current_bytes=$(file_size_bytes "$file")

    if [ "$current_bytes" -ge $((expected_bytes * 9 / 10)) ]; then
      echo -e "${GREEN}✓ ${DS_NAME[$ds]} already downloaded${NC}"
      continue
    fi

    download_one "$ds" &
  done
  echo -e "${YELLOW}All downloads started in background. Use '$0 watch' to monitor.${NC}"
  wait
}

# ── Upload to Google Drive ────────────────────────────────────────────────────

upload_one() {
  local ds="$1"
  local file="$RAW_DIR/${DS_FILE[$ds]}"

  if [ ! -e "$file" ]; then
    echo -e "${RED}✗ ${DS_FILE[$ds]} not found locally. Download first.${NC}"
    return 1
  fi

  if is_on_drive "$ds"; then
    echo -e "${GREEN}✓ ${DS_NAME[$ds]} already on Drive${NC}"
    return 0
  fi

  echo -e "${BLUE}Uploading ${DS_NAME[$ds]} to gdrive:$GDRIVE_DIR/...${NC}"
  rclone copy "$file" "gdrive:$GDRIVE_DIR/" --progress --drive-chunk-size 128M
  echo -e "${GREEN}✓ ${DS_NAME[$ds]} uploaded to Drive${NC}"
}

upload_all() {
  for ds in "${DATASETS[@]}"; do
    upload_one "$ds" || true
  done
}

# ── Pull from Google Drive ────────────────────────────────────────────────────

pull_one() {
  local ds="$1"
  local file="$RAW_DIR/${DS_FILE[$ds]}"

  if [ -e "$file" ]; then
    local current_bytes
    current_bytes=$(file_size_bytes "$file")
    local expected_bytes=$(( ${DS_SIZE[$ds]} * 1073741824 ))
    if [ "$current_bytes" -ge $((expected_bytes * 9 / 10)) ]; then
      echo -e "${GREEN}✓ ${DS_NAME[$ds]} already exists locally${NC}"
      return 0
    fi
  fi

  if ! is_on_drive "$ds"; then
    echo -e "${RED}✗ ${DS_NAME[$ds]} not on Drive. Upload first.${NC}"
    return 1
  fi

  echo -e "${BLUE}Pulling ${DS_NAME[$ds]} from Drive...${NC}"
  rclone copy "gdrive:$GDRIVE_DIR/${DS_FILE[$ds]}" "$RAW_DIR/" --progress
  echo -e "${GREEN}✓ ${DS_NAME[$ds]} pulled from Drive${NC}"
}

pull_all() {
  for ds in "${DATASETS[@]}"; do
    pull_one "$ds" || true
  done
}

# ── Main ──────────────────────────────────────────────────────────────────────

case "${1:-status}" in
  status)
    show_status
    ;;
  watch)
    while true; do
      clear
      show_status
      echo -e "\n${DIM}Refreshing every 5s. Ctrl+C to exit.${NC}"
      sleep 5
    done
    ;;
  download)
    if [ -n "${2:-}" ]; then
      download_one "$2"
    else
      download_all
    fi
    ;;
  upload)
    if [ -n "${2:-}" ]; then
      upload_one "$2"
    else
      upload_all
    fi
    ;;
  pull)
    if [ -n "${2:-}" ]; then
      pull_one "$2"
    else
      pull_all
    fi
    ;;
  help|*)
    echo "Usage: $0 <command> [dataset]"
    echo ""
    echo "Commands:"
    echo "  status              Show dataset dashboard"
    echo "  watch               Live dashboard (refreshes every 5s)"
    echo "  download [id]       Download datasets from source"
    echo "  upload [id]         Upload datasets to Google Drive"
    echo "  pull [id]           Pull datasets from Google Drive"
    echo ""
    echo "Dataset IDs: ${DATASETS[*]}"
    ;;
esac
