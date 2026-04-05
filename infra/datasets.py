#!/usr/bin/env python3
"""Dataset manager for audio-enhancer.

Downloads from various sources, uploads to Google Drive, pulls to new instances.

Usage:
    python infra/datasets.py status          — show dashboard (includes codec variants)
    python infra/datasets.py watch           — live dashboard (refreshes every 5s)
    python infra/datasets.py download        — download all datasets
    python infra/datasets.py download egipt  — download one dataset
    python infra/datasets.py upload          — upload all to Google Drive
    python infra/datasets.py upload musdb    — upload one to Drive
    python infra/datasets.py pull            — pull all from Google Drive
    python infra/datasets.py verify          — verify all dataset integrity
    python infra/datasets.py verify musdb    — verify one dataset

Phase 1 codec variants:
    python infra/datasets.py push-codec-variants [--from-dir DIR]
                                               — upload pre-computed codec WAVs to Drive
    python infra/datasets.py pull-codec-variants [--to-dir DIR]
                                               — pull codec WAVs from Drive
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


GDRIVE_DIR = "audio-enhancer-datasets"
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.environ.get("DATASETS_RAW", PROJECT_DIR / "datasets" / "raw"))

# Phase 1: pre-computed codec variant storage
# Local:  datasets/codec_variants/<dataset_name>/
# Drive:  gdrive:audio-enhancer-datasets/codec_variants/
CODEC_VARIANTS_DIR = Path(os.environ.get(
    "CODEC_VARIANTS_DIR",
    PROJECT_DIR / "datasets" / "codec_variants",
))
GDRIVE_CODEC_VARIANTS = f"{GDRIVE_DIR}/codec_variants"

# Phase 1 combined training dir (symlinks clean audio + codec variants)
PHASE1_COMBINED_DIR = Path(os.environ.get(
    "PHASE1_COMBINED_DIR",
    PROJECT_DIR / "datasets" / "phase1_combined",
))

# Retry configuration
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_RETRY_DELAY_S = 10
RCLONE_RETRIES = 5  # passed to rclone --retries flag


# ── Dataset catalog ───────────────────────────────────────────────────────────

@dataclass
class Dataset:
    id: str
    name: str
    source: str          # "url", "huggingface", "python", "manual"
    location: str        # URL, HF repo, or python callable name
    expected_gb: float
    filename: str        # local filename or directory name
    # Minimum expected file count after extraction (0 = skip file-count check)
    min_files: int = 0
    # Expected audio extensions for file-count check
    audio_exts: tuple = field(default_factory=lambda: (".wav", ".flac", ".mp3", ".ogg"))

CATALOG = [
    Dataset("egipt",    "EG-IPT (96kHz guitar)",      "url",
            "https://zenodo.org/records/15205644/files/EG-IPT.zip?download=1",
            22, "EG-IPT.zip", min_files=100),
    Dataset("musdb",    "MUSDB18-HQ (44.1kHz music)",  "url",
            "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1",
            22, "musdb18hq.zip", min_files=100),
    Dataset("vctk",     "VCTK 96kHz (speech)",         "url",
            "https://datashare.ed.ac.uk/bitstream/handle/10283/2774/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y",
            20, "vctk96k.zip", min_files=40000),
    Dataset("moisesdb", "MoisesDB (music stems)",      "manual",
            "https://developer.moises.ai/research",
            88, "moisesdb.zip"),
    Dataset("gtsinger", "GTSinger (48kHz vocals)",     "huggingface",
            "GTSinger/GTSinger",
            5, "gtsinger.tar.gz"),
    Dataset("maestro",  "MAESTRO v3 (piano)",          "url",
            "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
            101, "maestro-v3.0.0.zip", min_files=1000),
    Dataset("medleydb", "MedleyDB v2 (pro recordings)","manual",
            "https://medleydb.weebly.com",
            30, "medleydb-v2.zip"),
    Dataset("musicnet", "MusicNet (classical)",        "url",
            "https://zenodo.org/records/5120004/files/musicnet.tar.gz?download=1",
            11, "musicnet.tar.gz", min_files=300),
]

CATALOG_MAP = {ds.id: ds for ds in CATALOG}


# ── Custom download functions ─────────────────────────────────────────────────

def download_moisesdb(dest: Path):
    """Download MoisesDB using the datasets library from HuggingFace."""
    try:
        from datasets import load_dataset
        print("  Downloading MoisesDB via HuggingFace datasets library...")
        ds = load_dataset("wearemusicai/moisesdb", cache_dir=str(dest), trust_remote_code=True)
        print(f"  MoisesDB downloaded to {dest}")
    except ImportError:
        print("  Installing 'datasets' library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "soundfile"])
        from datasets import load_dataset
        ds = load_dataset("wearemusicai/moisesdb", cache_dir=str(dest), trust_remote_code=True)
        print(f"  MoisesDB downloaded to {dest}")


CUSTOM_DOWNLOADERS = {
    "download_moisesdb": download_moisesdb,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def local_path(ds: Dataset) -> Path:
    return RAW_DIR / ds.filename


def local_size_bytes(ds: Dataset) -> int:
    p = local_path(ds)
    if p.is_file():
        return p.stat().st_size
    elif p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def human_size(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f}G"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f}M"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.0f}K"
    return f"{b}B"


def is_downloading(ds: Dataset) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", ds.filename], stderr=subprocess.DEVNULL, text=True)
        for line in out.strip().split("\n"):
            if "wget" in line or "curl" in line or "huggingface" in line or "snapshot_download" in line or "datasets" in line:
                return True
    except subprocess.CalledProcessError:
        pass
    return False


def is_uploading(ds: Dataset) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", ds.filename], stderr=subprocess.DEVNULL, text=True)
        for line in out.strip().split("\n"):
            if "rclone" in line:
                return True
    except subprocess.CalledProcessError:
        pass
    return False


def is_on_drive(ds: Dataset) -> bool:
    try:
        result = subprocess.run(
            ["rclone", "lsf", f"gdrive:{GDRIVE_DIR}/{ds.filename}"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def progress_bar(pct: float, width: int = 20) -> str:
    filled = int(width * pct / 100)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def retry(fn, attempts: int = DOWNLOAD_MAX_RETRIES, delay: int = DOWNLOAD_RETRY_DELAY_S, label: str = ""):
    """Run fn(), retrying on exception up to `attempts` times."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  {C_YELLOW}Attempt {attempt}/{attempts} failed: {exc}. Retrying in {delay}s...{C_NC}")
            time.sleep(delay)


# ── Verify ────────────────────────────────────────────────────────────────────

def _count_audio_files(path: Path, exts: tuple) -> int:
    """Count audio files recursively under path."""
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in exts and f.is_file())


def _archive_is_valid(path: Path) -> Optional[str]:
    """Return None if archive is valid, or an error string if corrupt."""
    if not path.exists():
        return "file not found"
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
            if bad:
                return f"corrupt member: {bad}"
            return None
        except zipfile.BadZipFile as e:
            return str(e)
    if path.suffix in (".gz", ".tgz") or path.name.endswith(".tar.gz"):
        try:
            with tarfile.open(path) as tf:
                tf.getmembers()  # full member scan
            return None
        except tarfile.TarError as e:
            return str(e)
    # Unknown archive type — just check it's non-empty
    if path.stat().st_size == 0:
        return "file is empty"
    return None


def verify_one(ds: Dataset) -> bool:
    """Check local file integrity. Returns True if OK."""
    dest = local_path(ds)
    expected_bytes = int(ds.expected_gb * (1 << 30))
    current = local_size_bytes(ds)

    print(f"\n  {C_BOLD}{ds.name}{C_NC}")

    if current == 0:
        if ds.source == "manual":
            print(f"    {C_DIM}manual download — skipping{C_NC}")
            return True
        print(f"    {C_RED}not downloaded{C_NC}")
        return False

    # Size check (90% threshold)
    pct = int(current * 100 / expected_bytes) if expected_bytes > 0 else 100
    if current < expected_bytes * 0.9:
        print(f"    {C_RED}partial: {human_size(current)} / ~{human_size(expected_bytes)} ({pct}%){C_NC}")
        return False
    print(f"    size: {human_size(current)} ({pct}% of expected) {C_GREEN}OK{C_NC}")

    # Archive integrity check
    if dest.is_file() and dest.suffix in (".zip", ".gz", ".tgz") or (
            dest.is_file() and ".tar" in dest.name):
        print(f"    checking archive integrity...")
        err_msg = _archive_is_valid(dest)
        if err_msg:
            print(f"    {C_RED}archive corrupt: {err_msg}{C_NC}")
            return False
        print(f"    archive integrity {C_GREEN}OK{C_NC}")

    # File count check (for extracted directories)
    extracted_root = dest.parent.parent / "extracted" / dest.stem.split(".")[0]
    if ds.min_files > 0 and extracted_root.exists():
        count = _count_audio_files(extracted_root, ds.audio_exts)
        if count < ds.min_files:
            print(f"    {C_RED}extracted audio files: {count} < expected {ds.min_files}{C_NC}")
            return False
        print(f"    extracted audio files: {count} (>= {ds.min_files}) {C_GREEN}OK{C_NC}")
    elif ds.min_files > 0 and dest.is_dir():
        count = _count_audio_files(dest, ds.audio_exts)
        if count < ds.min_files:
            print(f"    {C_RED}audio files: {count} < expected {ds.min_files}{C_NC}")
            return False
        print(f"    audio files: {count} (>= {ds.min_files}) {C_GREEN}OK{C_NC}")

    return True


def verify_all() -> bool:
    """Verify all datasets. Returns True if all pass."""
    print(f"\n{C_BOLD}Verifying datasets...{C_NC}")
    results = {}
    for ds in CATALOG:
        results[ds.id] = verify_one(ds)

    print(f"\n{C_BOLD}Summary:{C_NC}")
    all_ok = True
    for ds in CATALOG:
        ok = results[ds.id]
        icon = f"{C_GREEN}PASS{C_NC}" if ok else f"{C_RED}FAIL{C_NC}"
        if ds.source == "manual" and local_size_bytes(ds) == 0:
            icon = f"{C_DIM}SKIP{C_NC}"
        print(f"  {icon}  {ds.name}")
        if not ok and ds.source != "manual":
            all_ok = False

    if all_ok:
        print(f"\n{C_GREEN}All datasets OK{C_NC}")
    else:
        print(f"\n{C_RED}Some datasets failed verification — re-download them{C_NC}")
    return all_ok


# ── Dashboard ─────────────────────────────────────────────────────────────────

C_RED = "\033[0;31m"
C_GREEN = "\033[0;32m"
C_YELLOW = "\033[1;33m"
C_BLUE = "\033[0;34m"
C_CYAN = "\033[0;36m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_NC = "\033[0m"


def show_status():
    print(f"{C_BOLD}{'=' * 78}{C_NC}")
    print(f"{C_BOLD}  Audio Enhancer — Dataset Dashboard{C_NC}")
    print(f"{C_BOLD}{'=' * 78}{C_NC}")
    print(f"  {'ID':<10} {'Dataset':<28} {'Local':>8}  {'Drive':>5}  {'Status':<20}")
    print(f"  {'-' * 10} {'-' * 28} {'-' * 8}  {'-' * 5}  {'-' * 20}")

    total_local = 0
    total_expected = 0

    for ds in CATALOG:
        local_bytes = local_size_bytes(ds)
        expected_bytes = int(ds.expected_gb * (1 << 30))
        total_local += local_bytes
        total_expected += expected_bytes

        # Local size
        local_str = human_size(local_bytes) if local_bytes > 0 else "-"

        # Drive
        on_drive = is_on_drive(ds)
        drive_str = f"{C_GREEN}yes{C_NC}" if on_drive else f"{C_DIM} - {C_NC}"

        # Status
        uploading = is_uploading(ds)
        downloading = is_downloading(ds)

        if uploading:
            status = f"{C_BLUE}uploading to Drive{C_NC}"
        elif downloading:
            pct = min(100, int(local_bytes * 100 / expected_bytes)) if expected_bytes > 0 else 0
            status = f"{C_YELLOW}{progress_bar(pct, 15)} {pct}%{C_NC}"
        elif on_drive and local_bytes == 0:
            status = f"{C_GREEN}on Drive{C_NC}"
        elif local_bytes > 0 and local_bytes >= expected_bytes * 0.9:
            if on_drive:
                status = f"{C_GREEN}ready + Drive{C_NC}"
            else:
                status = f"{C_GREEN}ready{C_NC}"
        elif local_bytes > 0:
            pct = min(100, int(local_bytes * 100 / expected_bytes)) if expected_bytes > 0 else 0
            status = f"{C_RED}partial ({pct}%){C_NC}"
        elif ds.source == "manual":
            status = f"{C_DIM}manual download{C_NC}"
        else:
            status = f"{C_DIM}not downloaded{C_NC}"

        print(f"  {C_CYAN}{ds.id:<10}{C_NC} {ds.name:<28} {local_str:>8}  {drive_str:>5}  {status}")

    print(f"  {'-' * 72}")

    disk_free = shutil.disk_usage(RAW_DIR).free
    print(f"  {C_BOLD}Local:{C_NC} {human_size(total_local)}  "
          f"{C_BOLD}Expected:{C_NC} {human_size(total_expected)}  "
          f"{C_BOLD}Disk free:{C_NC} {human_size(disk_free)}")
    print(f"{C_BOLD}{'=' * 78}{C_NC}")

    # Phase 1 codec variants status
    local_count = codec_variants_local_count(PHASE1_COMBINED_DIR)
    on_drive = codec_variants_on_drive()
    drive_str = f"{C_GREEN}yes{C_NC}" if on_drive else f"{C_DIM} - {C_NC}"
    local_str = f"{local_count} files" if local_count > 0 else f"{C_DIM}-{C_NC}"
    print(f"\n  {C_BOLD}Phase 1 — Codec Variants{C_NC}")
    print(f"  {'Local':<12} {'Drive':>5}  {'Notes':<30}")
    print(f"  {'-' * 12} {'-' * 5}  {'-' * 30}")
    notes = ""
    if local_count == 0 and not on_drive:
        notes = f"{C_DIM}run precompute_codecs.py first{C_NC}"
    elif local_count == 0:
        notes = f"{C_GREEN}ready to pull from Drive{C_NC}"
    elif not on_drive:
        notes = f"{C_YELLOW}not on Drive yet (run push-codec-variants){C_NC}"
    else:
        notes = f"{C_GREEN}synced{C_NC}"
    print(f"  {local_str:<12} {drive_str:>5}  {notes}")
    print(f"{C_BOLD}{'=' * 78}{C_NC}")


# ── Download ──────────────────────────────────────────────────────────────────

def download_one(ds: Dataset):
    dest = local_path(ds)
    expected_bytes = int(ds.expected_gb * (1 << 30))
    current = local_size_bytes(ds)

    if current >= expected_bytes * 0.9:
        print(f"  {C_GREEN}✓ {ds.name} already downloaded{C_NC}")
        return

    print(f"  {C_BLUE}Downloading {ds.name} (~{ds.expected_gb}GB)...{C_NC}")

    if ds.source == "url":
        def _wget():
            subprocess.run([
                "wget", "--tries=3", "--continue", "-q", "--show-progress",
                "-O", str(dest), ds.location,
            ], check=True)
        retry(_wget, label=ds.name)

    elif ds.source == "huggingface":
        def _hf():
            from huggingface_hub import snapshot_download
            snapshot_download(ds.location, repo_type="dataset", local_dir=str(dest))
        retry(_hf, label=ds.name)

    elif ds.source == "manual":
        print(f"  {C_YELLOW}Manual download required: {ds.location}{C_NC}")
        return

    elif ds.source == "python":
        fn = CUSTOM_DOWNLOADERS[ds.location]
        retry(lambda: fn(dest), label=ds.name)

    print(f"  {C_GREEN}✓ {ds.name} downloaded{C_NC}")


def download_all():
    for ds in CATALOG:
        try:
            download_one(ds)
        except Exception as e:
            print(f"  {C_RED}✗ {ds.name}: {e}{C_NC}")


# ── Upload to Google Drive ────────────────────────────────────────────────────

def upload_one(ds: Dataset):
    src = local_path(ds)
    if not src.exists():
        print(f"  {C_RED}✗ {ds.filename} not found locally{C_NC}")
        return

    if is_on_drive(ds):
        print(f"  {C_GREEN}✓ {ds.name} already on Drive{C_NC}")
        return

    print(f"  {C_BLUE}Uploading {ds.name} to Drive...{C_NC}")
    def _upload():
        subprocess.run([
            "rclone", "copy", str(src), f"gdrive:{GDRIVE_DIR}/",
            "--progress", "--drive-chunk-size", "128M",
            "--retries", str(RCLONE_RETRIES),
        ], check=True)
    retry(_upload, label=ds.name)
    print(f"  {C_GREEN}✓ {ds.name} uploaded{C_NC}")


def upload_all():
    for ds in CATALOG:
        try:
            upload_one(ds)
        except Exception as e:
            print(f"  {C_RED}✗ {ds.name}: {e}{C_NC}")


# ── Pull from Google Drive ────────────────────────────────────────────────────

def pull_one(ds: Dataset):
    dest = local_path(ds)
    current = local_size_bytes(ds)
    expected_bytes = int(ds.expected_gb * (1 << 30))

    if current >= expected_bytes * 0.9:
        print(f"  {C_GREEN}✓ {ds.name} already exists locally{C_NC}")
        return

    if not is_on_drive(ds):
        print(f"  {C_RED}✗ {ds.name} not on Drive{C_NC}")
        return

    print(f"  {C_BLUE}Pulling {ds.name} from Drive (~{ds.expected_gb}GB)...{C_NC}")
    def _pull():
        subprocess.run([
            "rclone", "copy", f"gdrive:{GDRIVE_DIR}/{ds.filename}", str(RAW_DIR) + "/",
            "--progress", "--retries", str(RCLONE_RETRIES),
        ], check=True)
    retry(_pull, label=ds.name)
    print(f"  {C_GREEN}✓ {ds.name} pulled{C_NC}")


def pull_all():
    for ds in CATALOG:
        try:
            pull_one(ds)
        except Exception as e:
            print(f"  {C_RED}✗ {ds.name}: {e}{C_NC}")


# ── Codec variants (Phase 1) ──────────────────────────────────────────────────

# Pre-computed codec variant filename pattern: *_mp3_128.wav, *_aac_256.wav, etc.
_CODEC_VARIANT_RE = re.compile(r"_(mp3|aac|ogg)_(\d+)\.wav$", re.IGNORECASE)

# rclone include patterns for codec variant files
_CODEC_VARIANT_INCLUDES = ["*_mp3_*.wav", "*_aac_*.wav", "*_ogg_*.wav"]


def codec_variants_local_count(directory: Path) -> int:
    """Count pre-computed codec variant WAVs in directory (recursively)."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*.wav") if _CODEC_VARIANT_RE.search(p.name))


def codec_variants_on_drive() -> bool:
    """Return True if codec_variants folder exists and has content on Drive."""
    try:
        result = subprocess.run(
            ["rclone", "lsf", f"gdrive:{GDRIVE_CODEC_VARIANTS}/", "--max-depth", "1"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def push_codec_variants(source_dir: Optional[Path] = None):
    """Upload codec variant WAV files from source_dir to Google Drive.

    Only files matching the *_{codec}_{bitrate}.wav naming convention produced by
    data/precompute_codecs.py are uploaded.  Directory structure is preserved so
    that pull_codec_variants() restores files to the correct locations.

    Args:
        source_dir: Directory tree to scan.  Defaults to PHASE1_COMBINED_DIR.
    """
    source_dir = source_dir or PHASE1_COMBINED_DIR
    if not source_dir.exists():
        print(f"  {C_RED}✗ Source dir {source_dir} does not exist{C_NC}")
        print(f"  {C_DIM}Run: python data/precompute_codecs.py --input <data_dir>{C_NC}")
        return

    count = codec_variants_local_count(source_dir)
    if count == 0:
        print(f"  {C_YELLOW}No codec variant files found in {source_dir}{C_NC}")
        print(f"  {C_DIM}Run: python data/precompute_codecs.py --input {source_dir}{C_NC}")
        return

    print(f"  {C_BLUE}Uploading {count} codec variant files from {source_dir}...{C_NC}")
    include_flags: list[str] = []
    for pat in _CODEC_VARIANT_INCLUDES:
        include_flags += ["--include", pat]

    def _upload():
        subprocess.run(
            [
                "rclone", "copy", str(source_dir),
                f"gdrive:{GDRIVE_CODEC_VARIANTS}/",
                *include_flags,
                "--progress",
                "--drive-chunk-size", "64M",
                "--retries", str(RCLONE_RETRIES),
            ],
            check=True,
        )

    retry(_upload, label="codec_variants")
    print(f"  {C_GREEN}✓ Codec variants uploaded → gdrive:{GDRIVE_CODEC_VARIANTS}/{C_NC}")


def pull_codec_variants(dest_dir: Optional[Path] = None):
    """Pull codec variant files from Drive into dest_dir.

    Files are written preserving the directory structure stored on Drive, so they
    land alongside their corresponding clean audio files -- ready for
    DegradedAudioDataset (data/dataset_phase1.py).

    Args:
        dest_dir: Local destination directory.  Defaults to PHASE1_COMBINED_DIR.
    """
    dest_dir = dest_dir or PHASE1_COMBINED_DIR

    if not codec_variants_on_drive():
        print(f"  {C_RED}✗ No codec variants on Drive (gdrive:{GDRIVE_CODEC_VARIANTS}/){C_NC}")
        print(f"  {C_DIM}Upload first: python infra/datasets.py push-codec-variants{C_NC}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {C_BLUE}Pulling codec variants → {dest_dir}...{C_NC}")

    def _pull():
        subprocess.run(
            [
                "rclone", "copy",
                f"gdrive:{GDRIVE_CODEC_VARIANTS}/",
                str(dest_dir) + "/",
                "--progress",
                "--retries", str(RCLONE_RETRIES),
            ],
            check=True,
        )

    retry(_pull, label="codec_variants")
    count = codec_variants_local_count(dest_dir)
    print(f"  {C_GREEN}✓ {count} codec variant files pulled → {dest_dir}{C_NC}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Audio Enhancer dataset manager")
    parser.add_argument("command", nargs="?", default="status",
                        choices=[
                            "status", "watch", "download", "upload", "pull", "verify",
                            "push-codec-variants", "pull-codec-variants", "help",
                        ])
    parser.add_argument("dataset", nargs="?", default=None,
                        help=f"Dataset ID: {', '.join(ds.id for ds in CATALOG)}")
    parser.add_argument("--from-dir", type=Path, default=None,
                        help="Source directory for push-codec-variants (default: PHASE1_COMBINED_DIR)")
    parser.add_argument("--to-dir", type=Path, default=None,
                        help="Destination directory for pull-codec-variants (default: PHASE1_COMBINED_DIR)")
    args = parser.parse_args()

    if args.command == "status":
        show_status()

    elif args.command == "watch":
        try:
            while True:
                os.system("clear")
                show_status()
                print(f"\n  {C_DIM}Refreshing every 5s. Ctrl+C to exit.{C_NC}")
                time.sleep(5)
        except KeyboardInterrupt:
            pass

    elif args.command == "download":
        if args.dataset:
            ds = CATALOG_MAP.get(args.dataset)
            if not ds:
                print(f"Unknown dataset: {args.dataset}")
                sys.exit(1)
            download_one(ds)
        else:
            download_all()

    elif args.command == "upload":
        if args.dataset:
            ds = CATALOG_MAP.get(args.dataset)
            if not ds:
                print(f"Unknown dataset: {args.dataset}")
                sys.exit(1)
            upload_one(ds)
        else:
            upload_all()

    elif args.command == "pull":
        if args.dataset:
            ds = CATALOG_MAP.get(args.dataset)
            if not ds:
                print(f"Unknown dataset: {args.dataset}")
                sys.exit(1)
            pull_one(ds)
        else:
            pull_all()

    elif args.command == "verify":
        if args.dataset:
            ds = CATALOG_MAP.get(args.dataset)
            if not ds:
                print(f"Unknown dataset: {args.dataset}")
                sys.exit(1)
            ok = verify_one(ds)
            sys.exit(0 if ok else 1)
        else:
            ok = verify_all()
            sys.exit(0 if ok else 1)

    elif args.command == "push-codec-variants":
        push_codec_variants(source_dir=args.from_dir)

    elif args.command == "pull-codec-variants":
        pull_codec_variants(dest_dir=args.to_dir)

    elif args.command == "help":
        parser.print_help()


if __name__ == "__main__":
    main()
