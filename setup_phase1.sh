#!/bin/bash
# Phase 1 Dataset Setup: Download, Extract, Prepare
# Run this script to get everything ready for training.
set -e

RAW_DIR="/home/stas/audio-enhancer/datasets/raw"
EXTRACT_DIR="/home/stas/audio-enhancer/datasets/extracted"
PREPARED_DIR="/home/stas/audio-enhancer/datasets/prepared"

mkdir -p "$RAW_DIR" "$EXTRACT_DIR" "$PREPARED_DIR"

echo "============================================"
echo "Phase 1 Dataset Setup"
echo "============================================"

# --- Check downloads ---
echo ""
echo "Checking downloads..."
check_file() {
    local file="$1"
    local min_mb="$2"
    local name="$3"
    if [ -f "$RAW_DIR/$file" ]; then
        size_mb=$(du -m "$RAW_DIR/$file" | cut -f1)
        if [ "$size_mb" -ge "$min_mb" ]; then
            echo "  ✓ $name ($size_mb MB)"
            return 0
        else
            echo "  ⏳ $name ($size_mb MB / ${min_mb}+ MB expected) — still downloading"
            return 1
        fi
    else
        echo "  ✗ $name — not found"
        return 1
    fi
}

all_ready=true
check_file "EG-IPT.zip" 20000 "EG-IPT (96kHz guitar)" || all_ready=false
check_file "musdb18hq.zip" 20000 "MUSDB18-HQ (mixed music)" || all_ready=false
check_file "vctk96k-wav1.zip" 8000 "VCTK 96kHz part 1" || all_ready=false
check_file "vctk96k-wav23.zip" 17000 "VCTK 96kHz part 2+3" || all_ready=false
# MoisesDB is optional for Phase 0
check_file "moisesdb.zip" 20000 "MoisesDB" || echo "  (MoisesDB optional for Phase 0)"

if [ "$all_ready" = false ]; then
    echo ""
    echo "Some downloads are still in progress. Run this script again when done."
    echo "Check progress: ls -lh $RAW_DIR/"
    exit 1
fi

# --- Extract ---
echo ""
echo "Extracting datasets..."

if [ ! -d "$EXTRACT_DIR/EG-IPT" ]; then
    echo "  Extracting EG-IPT..."
    unzip -q "$RAW_DIR/EG-IPT.zip" -d "$EXTRACT_DIR/EG-IPT"
else
    echo "  ✓ EG-IPT already extracted"
fi

if [ ! -d "$EXTRACT_DIR/musdb18hq" ]; then
    echo "  Extracting MUSDB18-HQ..."
    unzip -q "$RAW_DIR/musdb18hq.zip" -d "$EXTRACT_DIR/musdb18hq"
else
    echo "  ✓ MUSDB18-HQ already extracted"
fi

if [ ! -d "$EXTRACT_DIR/vctk96k" ]; then
    echo "  Extracting VCTK 96kHz..."
    mkdir -p "$EXTRACT_DIR/vctk96k"
    unzip -q "$RAW_DIR/vctk96k-wav1.zip" -d "$EXTRACT_DIR/vctk96k"
    unzip -q "$RAW_DIR/vctk96k-wav23.zip" -d "$EXTRACT_DIR/vctk96k"
else
    echo "  ✓ VCTK 96kHz already extracted"
fi

if [ -f "$RAW_DIR/moisesdb.zip" ] && [ ! -d "$EXTRACT_DIR/moisesdb" ]; then
    echo "  Extracting MoisesDB..."
    unzip -q "$RAW_DIR/moisesdb.zip" -d "$EXTRACT_DIR/moisesdb"
elif [ -d "$EXTRACT_DIR/moisesdb" ]; then
    echo "  ✓ MoisesDB already extracted"
fi

# --- Prepare (resample to 192kHz) ---
echo ""
echo "Preparing training data (resampling to 192kHz)..."
echo "This will take a while — processing hundreds of hours of audio."
echo ""

source /home/stas/audio-enhancer/.venv/bin/activate

# Process each dataset
for dataset in EG-IPT musdb18hq vctk96k moisesdb; do
    src="$EXTRACT_DIR/$dataset"
    dst="$PREPARED_DIR/$dataset"
    if [ -d "$src" ] && [ ! -d "$dst" ]; then
        echo "  Processing $dataset..."
        python3 /home/stas/audio-enhancer/prepare_dataset.py "$src" "$dst" \
            --target-sr 192000 --min-duration 5 --workers 8
    elif [ -d "$dst" ]; then
        echo "  ✓ $dataset already prepared"
    fi
done

echo ""
echo "============================================"
echo "Done! Training data is in: $PREPARED_DIR"
echo ""
echo "To start Phase 0 training (5 hours):"
echo "  cd /home/stas/audio-enhancer"
echo "  source .venv/bin/activate"
echo "  python3 train.py --data_dir datasets/prepared --config configs/phase0.yaml --max-hours 5"
echo "============================================"
