#!/usr/bin/env python3
"""Dataset manager for audio-enhancer.

Downloads from various sources, uploads to Google Drive, pulls to new instances.

Usage:
    python infra/datasets.py status          — show dashboard
    python infra/datasets.py watch           — live dashboard (refreshes every 5s)
    python infra/datasets.py download        — download all datasets
    python infra/datasets.py download egipt  — download one dataset
    python infra/datasets.py upload          — upload all to Google Drive
    python infra/datasets.py upload musdb    — upload one to Drive
    python infra/datasets.py pull            — pull all from Google Drive
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


GDRIVE_DIR = "audio-enhancer-datasets"
RAW_DIR = Path(os.environ.get("DATASETS_RAW", Path(__file__).resolve().parent.parent / "datasets" / "raw"))


# ── Dataset catalog ───────────────────────────────────────────────────────────

@dataclass
class Dataset:
    id: str
    name: str
    source: str          # "url", "huggingface", "python"
    location: str        # URL, HF repo, or python callable name
    expected_gb: float
    filename: str        # local filename or directory name

CATALOG = [
    Dataset("egipt",    "EG-IPT (96kHz guitar)",      "url",
            "https://zenodo.org/records/15205644/files/EG-IPT.zip?download=1",
            22, "EG-IPT.zip"),
    Dataset("musdb",    "MUSDB18-HQ (44.1kHz music)",  "url",
            "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1",
            22, "musdb18hq.zip"),
    Dataset("vctk",     "VCTK 96kHz (speech)",         "url",
            "https://datashare.ed.ac.uk/bitstream/handle/10283/2774/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y",
            20, "vctk96k.zip"),
    Dataset("moisesdb", "MoisesDB (music stems)",      "python",
            "download_moisesdb",
            25, "moisesdb"),
    Dataset("gtsinger", "GTSinger (48kHz vocals)",     "huggingface",
            "GTSinger/GTSinger",
            30, "gtsinger"),
    Dataset("maestro",  "MAESTRO v3 (piano)",          "url",
            "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
            120, "maestro-v3.0.0.zip"),
    Dataset("medleydb", "MedleyDB v2 (pro recordings)","url",
            "https://zenodo.org/records/1715175/files/MedleyDB-V2.zip?download=1",
            30, "medleydb-v2.zip"),
    Dataset("musicnet", "MusicNet (classical)",        "url",
            "https://zenodo.org/records/5120004/files/musicnet.tar.gz?download=1",
            11, "musicnet.tar.gz"),
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
        else:
            status = f"{C_DIM}not downloaded{C_NC}"

        print(f"  {C_CYAN}{ds.id:<10}{C_NC} {ds.name:<28} {local_str:>8}  {drive_str:>5}  {status}")

    print(f"  {'-' * 72}")

    disk_free = shutil.disk_usage(RAW_DIR).free
    print(f"  {C_BOLD}Local:{C_NC} {human_size(total_local)}  "
          f"{C_BOLD}Expected:{C_NC} {human_size(total_expected)}  "
          f"{C_BOLD}Disk free:{C_NC} {human_size(disk_free)}")
    print(f"{C_BOLD}{'=' * 78}{C_NC}")


# ── Download ──────────────────────────────────────────────────────────────────

def download_one(ds: Dataset):
    dest = local_path(ds)
    expected_bytes = int(ds.expected_gb * (1 << 30))
    current = local_size_bytes(ds)

    if current >= expected_bytes * 0.9:
        print(f"  {C_GREEN}✓ {ds.name} already downloaded{C_NC}")
        return

    print(f"  {C_BLUE}Downloading {ds.name}...{C_NC}")

    if ds.source == "url":
        subprocess.run([
            "wget", "--tries=5", "--continue", "-q", "--show-progress",
            "-O", str(dest), ds.location,
        ], check=True)

    elif ds.source == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(ds.location, repo_type="dataset", local_dir=str(dest))

    elif ds.source == "python":
        fn = CUSTOM_DOWNLOADERS[ds.location]
        fn(dest)

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
    subprocess.run([
        "rclone", "copy", str(src), f"gdrive:{GDRIVE_DIR}/",
        "--progress", "--drive-chunk-size", "128M",
    ], check=True)
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

    print(f"  {C_BLUE}Pulling {ds.name} from Drive...{C_NC}")
    subprocess.run([
        "rclone", "copy", f"gdrive:{GDRIVE_DIR}/{ds.filename}", str(RAW_DIR) + "/",
        "--progress",
    ], check=True)
    print(f"  {C_GREEN}✓ {ds.name} pulled{C_NC}")


def pull_all():
    for ds in CATALOG:
        try:
            pull_one(ds)
        except Exception as e:
            print(f"  {C_RED}✗ {ds.name}: {e}{C_NC}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Audio Enhancer dataset manager")
    parser.add_argument("command", nargs="?", default="status",
                        choices=["status", "watch", "download", "upload", "pull", "help"])
    parser.add_argument("dataset", nargs="?", default=None,
                        help=f"Dataset ID: {', '.join(ds.id for ds in CATALOG)}")
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

    elif args.command == "help":
        parser.print_help()


if __name__ == "__main__":
    main()
