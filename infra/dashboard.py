#!/usr/bin/env python3
"""Training and dataset monitoring dashboard.

Read-only Gradio dashboard with five panels:
  - Datasets       : local disk vs Google Drive status for all catalog entries
  - Training       : TensorBoard loss curves, step/epoch info
  - Checkpoints    : list of saved checkpoint files with epoch, timestamp, size
  - Infrastructure : Live Vast.ai instance status (GPU util, VRAM, cost, uptime)
                     + Google Drive reachability
  - Agent Team     : Live agent states, task summary, and recent messages from
                     the agent tracker SQLite DB

Vast.ai API key is read from (in order):
  1. VAST_API_KEY environment variable
  2. ~/.config/vastai/vast_api_key  (written by `vastai set api-key`)

Usage:
    python infra/dashboard.py
    python infra/dashboard.py --lang ru
    python infra/dashboard.py --log-dir checkpoints/phase0/logs
    python infra/dashboard.py --checkpoint-dir checkpoints/phase0
    python infra/dashboard.py --data-dir datasets/raw
    python infra/dashboard.py --port 7862
"""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import gradio as gr

# ── Paths ──────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCALES_DIR = _REPO_ROOT / "locales"

# ── Defaults (no magic numbers in code) ───────────────────────────────────────

DEFAULT_PORT = 7862
DEFAULT_LOG_DIR = "checkpoints/phase0/logs"
DEFAULT_CHECKPOINT_DIR = "checkpoints/phase0"
DEFAULT_DATA_DIR = "datasets/raw"
DEFAULT_GDRIVE_REMOTE = "gdrive:"
DEFAULT_GDRIVE_DIR = "audio-enhancer-datasets"

# Fraction of expected size considered "ready"
READY_THRESHOLD = 0.9

# Auto-refresh interval shown in UI label
AUTO_REFRESH_INTERVAL_S = 30

# Maximum TensorBoard scalar steps loaded per tag
MAX_TB_STEPS = 2000

# Loss tags we care about — order determines plot order
GENERATOR_LOSS_TAGS = [
    "train/g_loss",
    "train/g_adv_loss",
    "train/g_fm_loss",
    "train/g_stft_loss",
    "train/g_mel_loss",
    "train/g_mastering_loss",
]
DISCRIMINATOR_LOSS_TAGS = [
    "train/d_loss",
    "train/d_adv_loss",
]

# Loss plot colour palette (dark-theme friendly)
PLOT_COLORS = ["#00d4ff", "#ff6b6b", "#ffd166", "#06d6a0", "#a8dadc", "#e63946", "#457b9d"]

# ── Runtime config (overridden by CLI) ────────────────────────────────────────

_log_dir: Path = _REPO_ROOT / DEFAULT_LOG_DIR
_checkpoint_dir: Path = _REPO_ROOT / DEFAULT_CHECKPOINT_DIR
_data_dir: Path = _REPO_ROOT / DEFAULT_DATA_DIR
_lang: str = "en"

# ── Localisation ──────────────────────────────────────────────────────────────

_locales: dict[str, dict] = {}


def _load_locales():
    for code in ("en", "ru"):
        p = _LOCALES_DIR / f"{code}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _locales[code] = json.load(f)


def t(key: str) -> str:
    return _locales.get(_lang, _locales.get("en", {})).get(key, key)


_load_locales()

# ── CSS ────────────────────────────────────────────────────────────────────────

DASHBOARD_CSS = """
.gradio-container { max-width: 95% !important; margin: 0 auto !important; }

/* Status table */
.ds-table { font-family: 'Segoe UI', monospace; width: 100%;
            border-collapse: collapse; margin-bottom: 10px; }
.ds-table th { color: #0088cc; text-align: left; padding: 6px 10px;
               border-bottom: 1px solid #333; }
.ds-table td { padding: 5px 10px; vertical-align: middle; }
.ds-table tr:nth-child(even) td { background: rgba(255,255,255,0.03); }

/* Status badges */
.badge-good { color: #00aa66; font-weight: bold; }
.badge-ok   { color: #cc8800; font-weight: bold; }
.badge-bad  { color: #cc3333; font-weight: bold; }
.badge-info { color: #0088cc; font-weight: bold; }
.muted      { color: #888; }

/* Training summary */
.train-summary { font-family: 'Segoe UI', sans-serif; padding: 8px 0; }
.train-summary table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
.train-summary td.label { color: #888; padding: 4px 8px; width: 45%; }
.train-summary td.value { padding: 4px 8px; font-weight: bold; }
.train-summary h3 { color: #0088cc; margin: 8px 0 4px; }

/* Checkpoint / infra table */
.info-table { font-family: 'Segoe UI', monospace; width: 100%;
              border-collapse: collapse; margin-bottom: 10px; }
.info-table th { color: #0088cc; text-align: left; padding: 6px 10px;
                 border-bottom: 1px solid #333; }
.info-table td { padding: 5px 10px; }
.info-table tr:nth-child(even) td { background: rgba(255,255,255,0.03); }
.info-table h3 { color: #0088cc; margin: 10px 0 4px; }

/* Footer note */
.note { color: #888; font-size: 0.82em; margin-top: 6px; }

/* Agent team status */
.agent-table { font-family: 'Segoe UI', monospace; width: 100%;
               border-collapse: collapse; margin-bottom: 10px; }
.agent-table th { color: #0088cc; text-align: left; padding: 6px 10px;
                  border-bottom: 1px solid #333; }
.agent-table td { padding: 5px 10px; vertical-align: top; }
.agent-table tr:nth-child(even) td { background: rgba(255,255,255,0.03); }
.agent-role { color: #888; font-size: 0.8em; }
.msg-summary { max-width: 600px; overflow: hidden; text-overflow: ellipsis;
               white-space: nowrap; font-size: 0.85em; }
.task-counts { display: flex; gap: 16px; margin: 8px 0 12px; }
.task-count-box { padding: 6px 14px; border-radius: 6px; text-align: center; }
.task-count-box .count { font-size: 1.5em; font-weight: bold; display: block; }
.task-count-box .label { font-size: 0.75em; color: #aaa; }
"""

# ── Import infra/datasets.py ──────────────────────────────────────────────────

def _import_datasets_module():
    spec = importlib.util.spec_from_file_location(
        "datasets_infra",
        Path(__file__).resolve().parent / "datasets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["datasets_infra"] = mod
    spec.loader.exec_module(mod)
    return mod


_ds_mod = _import_datasets_module()
_CATALOG = _ds_mod.CATALOG
_local_size_bytes = _ds_mod.local_size_bytes
_human_size = _ds_mod.human_size
_is_on_drive = _ds_mod.is_on_drive
_is_downloading = _ds_mod.is_downloading
_is_uploading = _ds_mod.is_uploading


# ── Panel 1: Dataset status ───────────────────────────────────────────────────

def _dataset_status_html(ds, local_bytes: int, on_drive: bool) -> str:
    expected_bytes = int(ds.expected_gb * (1 << 30))

    if _is_uploading(ds):
        return f'<span class="badge-info">{t("ds_status_uploading")}</span>'
    if _is_downloading(ds):
        pct = min(100, int(local_bytes * 100 / expected_bytes)) if expected_bytes > 0 else 0
        label = t("ds_status_downloading").format(pct=pct)
        return f'<span class="badge-ok">{label}</span>'
    if local_bytes >= expected_bytes * READY_THRESHOLD:
        if on_drive:
            return f'<span class="badge-good">{t("ds_status_ready_drive")}</span>'
        return f'<span class="badge-good">{t("ds_status_ready")}</span>'
    if local_bytes > 0:
        pct = min(100, int(local_bytes * 100 / expected_bytes)) if expected_bytes > 0 else 0
        label = t("ds_status_partial").format(pct=pct)
        return f'<span class="badge-bad">{label}</span>'
    if on_drive:
        return f'<span class="badge-info">{t("ds_status_drive_only")}</span>'
    if ds.source == "manual":
        return f'<span class="muted">{t("ds_status_manual")}</span>'
    return f'<span class="muted">{t("ds_status_not_downloaded")}</span>'


def build_dataset_table() -> str:
    rows_html = ""
    total_local = 0
    total_expected = 0

    # Override RAW_DIR so local_path() resolves against the configured data dir
    _ds_mod.RAW_DIR = _data_dir

    for ds in _CATALOG:
        local_bytes = _local_size_bytes(ds)
        expected_bytes = int(ds.expected_gb * (1 << 30))
        total_local += local_bytes
        total_expected += expected_bytes

        local_str = _human_size(local_bytes) if local_bytes > 0 else '<span class="muted">-</span>'
        expected_str = f"{ds.expected_gb:.0f} GB"
        on_drive = _is_on_drive(ds)
        drive_str = (
            f'<span class="badge-good">yes</span>'
            if on_drive
            else '<span class="muted">-</span>'
        )
        status_html = _dataset_status_html(ds, local_bytes, on_drive)

        rows_html += f"""
        <tr>
          <td><b>{ds.name}</b><br>
              <span class="muted" style="font-size:0.8em">{ds.id}</span></td>
          <td>{expected_str}</td>
          <td>{local_str}</td>
          <td style="text-align:center">{drive_str}</td>
          <td>{status_html}</td>
        </tr>
        """

    try:
        data_path = _data_dir if _data_dir.exists() else Path(".")
        disk = shutil.disk_usage(data_path)
        disk_free_str = _human_size(disk.free)
        disk_total_str = _human_size(disk.total)
        disk_used_pct = int(disk.used * 100 / disk.total)
    except Exception:
        disk_free_str = disk_total_str = t("train_unknown")
        disk_used_pct = 0

    footer = f"""
    <tr style="border-top:1px solid #333">
      <td colspan="2"><b>{t("ds_totals")}</b></td>
      <td><b>{_human_size(total_local)}</b> / {_human_size(total_expected)}</td>
      <td colspan="2" class="muted">
        Disk: {disk_free_str} free / {disk_total_str} total ({disk_used_pct}% used)
      </td>
    </tr>
    """

    html = f"""
    <table class="ds-table">
      <thead>
        <tr>
          <th>{t("ds_col_dataset")}</th>
          <th>{t("ds_col_expected")}</th>
          <th>{t("ds_col_local")}</th>
          <th>{t("ds_col_drive")}</th>
          <th>{t("ds_col_status")}</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
        {footer}
      </tbody>
    </table>
    <p class="note">{t("ds_data_dir")}: {_data_dir.resolve()}</p>
    """
    return html


def refresh_datasets() -> str:
    try:
        return build_dataset_table()
    except Exception as e:
        return f'<p class="badge-bad">Error: {e}</p>'


# ── Panel 2: Training progress ────────────────────────────────────────────────

def _read_tb_scalars(log_dir: Path, tags: list[str]) -> dict[str, tuple[list, list]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    if not log_dir.exists():
        return {}
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": MAX_TB_STEPS})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            result[tag] = ([e.step for e in events], [e.value for e in events])
    return result


def build_training_summary() -> tuple[str, list, list, list, list]:
    log_dir = _log_dir
    no_data_html = (
        f'<div class="train-summary">'
        f'<p class="muted">{t("train_no_data")}<br>'
        f'<code>{log_dir.resolve()}</code></p>'
        f'<p class="note">{t("train_no_data_hint")}</p>'
        f'</div>'
    )

    all_tags = GENERATOR_LOSS_TAGS + DISCRIMINATOR_LOSS_TAGS
    scalars = _read_tb_scalars(log_dir, all_tags)
    if not scalars:
        return no_data_html, [], [], [], []

    ref_tag = next(
        (tg for tg in ["train/g_loss", "train/d_loss"] + list(scalars.keys()) if tg in scalars),
        None,
    )
    current_step = 0
    current_epoch: int | str = t("train_unknown")
    speed_str = t("train_unknown")

    if ref_tag:
        steps, _ = scalars[ref_tag]
        current_step = steps[-1] if steps else 0
        if "train/epoch" in scalars:
            current_epoch = int(scalars["train/epoch"][1][-1])

        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(log_dir), size_guidance={"scalars": MAX_TB_STEPS})
            ea.Reload()
            if ref_tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(ref_tag)
                if len(events) >= 2:
                    tail = events[-10:]
                    total_steps = tail[-1].step - tail[0].step
                    total_time = tail[-1].wall_time - tail[0].wall_time
                    if total_time > 0 and total_steps > 0:
                        sps = total_steps / total_time
                        speed_str = t("train_speed_fmt").format(sps=sps, sec=1 / sps)
        except Exception:
            pass

    # Latest checkpoint age
    latest_ckpt_path = _find_latest_checkpoint(_checkpoint_dir)
    if latest_ckpt_path:
        cp = Path(latest_ckpt_path)
        age_s = time.time() - cp.stat().st_mtime
        if age_s < 3600:
            age_str = t("train_age_min").format(n=int(age_s // 60))
        elif age_s < 86400:
            age_str = t("train_age_hours").format(h=age_s / 3600)
        else:
            age_str = t("train_age_days").format(d=age_s / 86400)
        ckpt_str = f"{cp.name} ({age_str})"
    else:
        ckpt_str = f'<span class="muted">{t("train_no_checkpoint")}</span>'

    if isinstance(current_step, int):
        step_display = f"{current_step:,}"
    else:
        step_display = str(current_step)

    summary_html = f"""
    <div class="train-summary">
      <h3>{t("train_state_header")}</h3>
      <table>
        <tr><td class="label">{t("train_current_step")}</td>
            <td class="value">{step_display}</td></tr>
        <tr><td class="label">{t("train_current_epoch")}</td>
            <td class="value">{current_epoch}</td></tr>
        <tr><td class="label">{t("train_speed")}</td>
            <td class="value">{speed_str}</td></tr>
        <tr><td class="label">{t("train_last_checkpoint")}</td>
            <td class="value">{ckpt_str}</td></tr>
        <tr><td class="label">{t("train_log_dir")}</td>
            <td class="value muted" style="font-size:0.85em">{log_dir.resolve()}</td></tr>
      </table>
    </div>
    """

    g_steps: list = []
    g_series: list[tuple[str, list]] = []
    for tag in GENERATOR_LOSS_TAGS:
        if tag in scalars:
            steps_t, vals = scalars[tag]
            if not g_steps:
                g_steps = steps_t
            g_series.append((tag.split("/")[-1], vals))

    d_steps: list = []
    d_series: list[tuple[str, list]] = []
    for tag in DISCRIMINATOR_LOSS_TAGS:
        if tag in scalars:
            steps_t, vals = scalars[tag]
            if not d_steps:
                d_steps = steps_t
            d_series.append((tag.split("/")[-1], vals))

    return summary_html, g_steps, g_series, d_steps, d_series


def _make_loss_figure(steps: list, series: list[tuple[str, list]], title: str):
    if not steps or not series:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#888")
    ax.set_xlabel("Step", color="#aaa")
    ax.set_ylabel("Loss", color="#aaa")
    ax.set_title(title, color="#fff", fontsize=13)
    ax.grid(True, color="#2a2a4a", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    for i, (label, values) in enumerate(series):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        ax.plot(steps[: len(values)], values, label=label, color=color, linewidth=1.2, alpha=0.85)

    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ccc", fontsize=9)
    plt.tight_layout()
    return fig


def _build_epoch_summary_html(scalars: dict) -> str:
    """Build an HTML table summarising per-epoch average loss values.

    Reads 'train/epoch' scalar to determine epoch boundaries, then computes
    per-epoch means for g_loss and d_loss.  Falls back to a single-row
    summary when no epoch tag is available (epoch 0 completed case).
    """
    g_steps_all, g_vals_all = scalars.get("train/g_loss", ([], []))
    d_steps_all, d_vals_all = scalars.get("train/d_loss", ([], []))

    if not g_steps_all and not d_steps_all:
        return (
            f'<p class="muted note">{t("train_epoch_no_data")}</p>'
        )

    # If epoch tag is present use it to bucket steps; otherwise treat all
    # existing data as epoch 0.
    epoch_scalars = scalars.get("train/epoch", ([], []))
    epoch_steps_raw, epoch_vals_raw = epoch_scalars

    if epoch_steps_raw:
        # Build a step→epoch mapping
        step_to_epoch: dict[int, int] = {}
        for s, e in zip(epoch_steps_raw, epoch_vals_raw):
            step_to_epoch[int(s)] = int(e)

        def _epoch_for_step(step: int) -> int:
            # Find the closest epoch boundary at or before this step
            best = 0
            for es in sorted(step_to_epoch):
                if es <= step:
                    best = step_to_epoch[es]
                else:
                    break
            return best

        # Collect per-epoch g/d values
        epoch_g: dict[int, list[float]] = {}
        for step, val in zip(g_steps_all, g_vals_all):
            ep = _epoch_for_step(int(step))
            epoch_g.setdefault(ep, []).append(val)

        epoch_d: dict[int, list[float]] = {}
        for step, val in zip(d_steps_all, d_vals_all):
            ep = _epoch_for_step(int(step))
            epoch_d.setdefault(ep, []).append(val)

        all_epochs = sorted(set(list(epoch_g.keys()) + list(epoch_d.keys())))
    else:
        # No epoch tag: treat everything as epoch 0
        epoch_g = {0: list(g_vals_all)} if g_vals_all else {}
        epoch_d = {0: list(d_vals_all)} if d_vals_all else {}
        all_epochs = [0]

    if not all_epochs:
        return f'<p class="muted note">{t("train_epoch_no_data")}</p>'

    rows_html = ""
    for ep in all_epochs:
        g_list = epoch_g.get(ep, [])
        d_list = epoch_d.get(ep, [])
        n_steps = max(len(g_list), len(d_list))
        g_avg = sum(g_list) / len(g_list) if g_list else None
        d_avg = sum(d_list) / len(d_list) if d_list else None
        rows_html += f"""
        <tr>
          <td style="text-align:center">{ep}</td>
          <td style="text-align:right">{n_steps:,}</td>
          <td style="text-align:right;font-family:monospace">{_format_loss(g_avg)}</td>
          <td style="text-align:right;font-family:monospace">{_format_loss(d_avg)}</td>
        </tr>
        """

    return f"""
    <div class="train-summary" style="margin-top:12px">
      <h3>{t("train_epoch_summary_header")}</h3>
      <table class="info-table">
        <thead>
          <tr>
            <th>{t("train_epoch_col_epoch")}</th>
            <th style="text-align:right">{t("train_epoch_col_steps")}</th>
            <th style="text-align:right">{t("train_epoch_col_g_loss")}</th>
            <th style="text-align:right">{t("train_epoch_col_d_loss")}</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """


def refresh_training():
    summary_html, g_steps, g_series, d_steps, d_series = build_training_summary()
    # Build epoch summary from the same scalars so we avoid double I/O
    all_tags = GENERATOR_LOSS_TAGS + DISCRIMINATOR_LOSS_TAGS + ["train/epoch"]
    scalars_full = _read_tb_scalars(_log_dir, all_tags)
    epoch_html = _build_epoch_summary_html(scalars_full)
    combined_html = summary_html + epoch_html
    g_fig = _make_loss_figure(g_steps, g_series, t("train_g_loss_plot"))
    d_fig = _make_loss_figure(d_steps, d_series, t("train_d_loss_plot"))
    return combined_html, g_fig, d_fig


# ── Panel 3: Checkpoint history ───────────────────────────────────────────────

def _find_latest_checkpoint(base_dir: Path) -> str | None:
    candidates = list(base_dir.glob("*.pt"))
    if not candidates:
        candidates = list(base_dir.glob("**/*.pt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _read_checkpoint_meta(path: Path) -> dict:
    """Load epoch, step, g_loss, d_loss, phase from a checkpoint file.

    Returns a dict with keys: epoch, step, g_loss, d_loss, phase.
    All values may be None if not found or if torch is unavailable.
    """
    meta: dict = {"epoch": None, "step": None, "g_loss": None, "d_loss": None, "phase": None}
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        epoch = ckpt.get("epoch")
        if epoch is not None:
            meta["epoch"] = int(epoch)
        step = ckpt.get("step") or ckpt.get("global_step")
        if step is not None:
            meta["step"] = int(step)
        # Loss values stored directly or under a losses sub-dict
        losses = ckpt.get("losses") or {}
        g_loss = ckpt.get("g_loss") or losses.get("g_loss")
        d_loss = ckpt.get("d_loss") or losses.get("d_loss")
        if g_loss is not None:
            meta["g_loss"] = float(g_loss)
        if d_loss is not None:
            meta["d_loss"] = float(d_loss)
        meta["phase"] = ckpt.get("phase") or ckpt.get("training_phase")
    except Exception:
        pass
    # Fall back to parsing epoch from filename (e.g. checkpoint_0003.pt)
    if meta["epoch"] is None:
        stem = path.stem
        for part in reversed(stem.split("_")):
            if part.isdigit():
                meta["epoch"] = int(part)
                break
    return meta


def _checkpoint_drive_path(cp: Path) -> str:
    """Return the expected Google Drive path for a checkpoint file."""
    return f"{DEFAULT_GDRIVE_REMOTE}{DEFAULT_GDRIVE_DIR}/checkpoints/{cp.name}"


def _format_loss(val: float | None) -> str:
    """Format a loss value for display, or return the N/A marker."""
    if val is None:
        return f'<span class="muted">{t("ckpt_metrics_na")}</span>'
    return f"{val:.2f}"


def build_checkpoint_table() -> str:
    candidates = sorted(
        _checkpoint_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return (
            f'<p class="muted">{t("ckpt_no_checkpoints")}</p>'
            f'<p class="note">{t("ckpt_dir")}: {_checkpoint_dir.resolve()}</p>'
        )

    # Check which checkpoints are on Google Drive (non-blocking: use cached drive status)
    try:
        import subprocess
        result = subprocess.run(
            ["rclone", "lsf", f"{DEFAULT_GDRIVE_REMOTE}{DEFAULT_GDRIVE_DIR}/checkpoints/"],
            capture_output=True, text=True, timeout=10,
        )
        drive_files = set(result.stdout.splitlines()) if result.returncode == 0 else set()
    except Exception:
        drive_files = None  # None = check failed, show "…"

    rows_html = ""
    for cp in candidates:
        stat = cp.stat()
        age_s = time.time() - stat.st_mtime
        if age_s < 3600:
            mod_str = t("train_age_min").format(n=int(age_s // 60))
        elif age_s < 86400:
            mod_str = t("train_age_hours").format(h=age_s / 3600)
        else:
            mod_str = t("train_age_days").format(d=age_s / 86400)

        size_str = _human_size(stat.st_size)
        meta = _read_checkpoint_meta(cp)

        epoch_str = str(meta["epoch"]) if meta["epoch"] is not None else t("ckpt_epoch_unknown")
        g_loss_str = _format_loss(meta["g_loss"])
        d_loss_str = _format_loss(meta["d_loss"])

        step_note = (
            f'<br><span class="muted" style="font-size:0.8em">'
            f'{t("ckpt_step")}: {meta["step"]:,}</span>'
            if meta["step"] is not None
            else ""
        )
        phase_note = (
            f'<br><span class="muted" style="font-size:0.8em">'
            f'{t("ckpt_phase")}: {meta["phase"]}</span>'
            if meta["phase"]
            else ""
        )

        # Drive presence indicator
        if drive_files is None:
            drive_html = f'<span class="muted">{t("ckpt_drive_checking")}</span>'
        elif cp.name in drive_files or (cp.name + "/") in drive_files:
            drive_html = f'<span class="badge-good">{t("ckpt_drive_yes")}</span>'
        else:
            drive_html = f'<span class="badge-bad">{t("ckpt_drive_no")}</span>'

        rows_html += f"""
        <tr>
          <td><b>{cp.name}</b>{phase_note}{step_note}</td>
          <td style="text-align:center">{epoch_str}</td>
          <td style="text-align:right;font-family:monospace">{g_loss_str}</td>
          <td style="text-align:right;font-family:monospace">{d_loss_str}</td>
          <td>{mod_str}</td>
          <td style="text-align:right">{size_str}</td>
          <td style="text-align:center">{drive_html}</td>
        </tr>
        """

    drive_path = f"{DEFAULT_GDRIVE_REMOTE}{DEFAULT_GDRIVE_DIR}/checkpoints/"
    html = f"""
    <table class="info-table">
      <thead>
        <tr>
          <th>{t("ckpt_col_file")}</th>
          <th>{t("ckpt_col_epoch")}</th>
          <th>{t("ckpt_col_g_loss")}</th>
          <th>{t("ckpt_col_d_loss")}</th>
          <th>{t("ckpt_col_modified")}</th>
          <th>{t("ckpt_col_size")}</th>
          <th>{t("ckpt_col_drive")}</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    <p class="note">{t("ckpt_dir")}: {_checkpoint_dir.resolve()}</p>
    <p class="note">{t("ckpt_gdrive_path")}: <code>{drive_path}</code></p>
    """
    return html


def refresh_checkpoints() -> str:
    try:
        return build_checkpoint_table()
    except Exception as e:
        return f'<p class="badge-bad">Error: {e}</p>'


# ── Panel 4: Infrastructure status ───────────────────────────────────────────

# Vast.ai REST API base URL
_VASTAI_API_BASE = "https://console.vast.ai/api/v0"

# Default key location written by `vastai set api-key`
_VASTAI_KEY_FILE = Path.home() / ".config" / "vastai" / "vast_api_key"

# How long to cache the rclone Drive reachability result (seconds)
DRIVE_CACHE_TTL_S = 60

# How long to cache the Vast.ai instances result (seconds)
VASTAI_CACHE_TTL_S = 30

# ── Non-blocking caches ───────────────────────────────────────────────────────

import threading as _threading

_drive_cache: dict = {"html": '<span class="muted">checking…</span>', "ts": 0.0}
_drive_lock = _threading.Lock()
_drive_checking = False  # guard against concurrent background threads

_vastai_cache: dict = {"html": f'<p class="muted">checking…</p>', "ts": 0.0}
_vastai_lock = _threading.Lock()
_vastai_checking = False  # guard against concurrent background threads


def _check_drive_reachability() -> None:
    """Run rclone check in a background thread and update the cache."""
    global _drive_checking
    import subprocess
    try:
        result = subprocess.run(
            ["rclone", "lsf", f"{DEFAULT_GDRIVE_REMOTE}{DEFAULT_GDRIVE_DIR}/"],
            capture_output=True, text=True, timeout=15,
        )
        html = (
            '<span class="badge-good">reachable</span>'
            if result.returncode == 0
            else '<span class="badge-bad">unreachable</span>'
        )
    except FileNotFoundError:
        html = '<span class="badge-bad">rclone not found</span>'
    except Exception:
        html = '<span class="badge-ok">check failed</span>'

    with _drive_lock:
        _drive_cache["html"] = html
        _drive_cache["ts"] = time.time()
        _drive_checking = False


def _get_drive_status_html() -> str:
    """Return cached Drive status; trigger a background refresh if stale."""
    global _drive_checking
    with _drive_lock:
        age = time.time() - _drive_cache["ts"]
        cached_html = _drive_cache["html"]
        should_refresh = age > DRIVE_CACHE_TTL_S and not _drive_checking
        if should_refresh:
            _drive_checking = True

    if should_refresh:
        _threading.Thread(target=_check_drive_reachability, daemon=True).start()

    return cached_html


def _get_vastai_api_key() -> str:
    """Return API key from VAST_API_KEY env var or ~/.config/vastai/vast_api_key."""
    import os
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        return key
    if _VASTAI_KEY_FILE.exists():
        return _VASTAI_KEY_FILE.read_text().strip()
    return ""


def _fetch_vastai_instances_api(api_key: str) -> list[dict]:
    """Fetch instances via Vast.ai REST API."""
    import requests
    resp = requests.get(
        f"{_VASTAI_API_BASE}/instances/",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("instances", [])


def _fetch_vastai_instances_cli() -> list[dict]:
    """Fetch instances via vastai CLI as fallback."""
    import subprocess
    result = subprocess.run(
        ["vastai", "show", "instances", "--raw"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def _pct_bar(pct: float, width: int = 12) -> str:
    """Inline block-character progress bar with colour coding."""
    filled = int(width * pct / 100)
    color = "#00aa66" if pct < 80 else ("#cc8800" if pct < 95 else "#cc3333")
    return (
        f'<span style="font-family:monospace;color:{color}">'
        f'{"█" * filled}{"░" * (width - filled)}'
        f'</span> {pct:.1f}%'
    )


def _format_uptime(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h >= 24:
        return f"{h // 24}d {h % 24}h {m}m"
    return f"{h}h {m}m"


def _instance_status_badge(actual: str, cur: str) -> str:
    """Render status badge; show cur_state separately when it differs from actual_status."""
    a = (actual or "").lower()
    c = (cur or "").lower()

    if a == "running":
        badge = '<span class="badge-good">running</span>'
    elif a in ("loading", "provisioning", "stopping", "exited"):
        badge = f'<span class="badge-ok">{a}</span>'
    else:
        badge = f'<span class="badge-bad">{a or "unknown"}</span>'

    # Show cur_state as a secondary note when it carries extra info
    if c and c != a and c not in ("running",):
        badge += f' <span class="muted" style="font-size:0.85em">({c})</span>'
    return badge


def _provisioning(inst: dict) -> bool:
    """True when the instance is not yet running — util fields will be zero/null."""
    s = (inst.get("actual_status") or inst.get("cur_state") or "").lower()
    return s in ("loading", "provisioning", "stopped", "created")


def _build_instance_html(inst: dict) -> str:
    iid      = inst.get("id", t("train_unknown"))
    actual   = inst.get("actual_status") or ""
    cur      = inst.get("cur_state") or ""
    gpu_name = inst.get("gpu_name", t("train_unknown"))
    num_gpus = inst.get("num_gpus", 1)
    gpu_label = f"{num_gpus}x {gpu_name}" if num_gpus and num_gpus > 1 else gpu_name

    booting = _provisioning(inst)
    provisioning_note = '<span class="muted"> (provisioning…)</span>' if booting else ""

    # GPU utilisation — suppress bars while booting to avoid false 0% alarms
    gpu_util = inst.get("gpu_util")
    if booting:
        gpu_util_html = f'<span class="muted">-{provisioning_note}</span>'
    elif gpu_util is not None:
        gpu_util_html = _pct_bar(gpu_util)
    else:
        gpu_util_html = '<span class="muted">-</span>'

    # VRAM: vmem_usage is in GB, gpu_totalram is in MB
    vmem_gb         = inst.get("vmem_usage")
    gpu_totalram_gb = (inst.get("gpu_totalram") or inst.get("gpu_ram", 0)) / 1024
    if not booting and vmem_gb is not None and gpu_totalram_gb > 0:
        vram_pct  = min(100.0, vmem_gb / gpu_totalram_gb * 100)
        vram_html = f"{_pct_bar(vram_pct)} ({vmem_gb:.1f} / {gpu_totalram_gb:.0f} GB)"
    else:
        vram_html = '<span class="muted">-</span>'

    # GPU temperature
    gpu_temp = inst.get("gpu_temp")
    if not booting and gpu_temp is not None:
        temp_cls  = "badge-bad" if gpu_temp > 85 else ("badge-ok" if gpu_temp > 75 else "badge-good")
        temp_html = f'<span class="{temp_cls}">{gpu_temp:.0f} °C</span>'
    else:
        temp_html = '<span class="muted">-</span>'

    # CPU utilisation
    cpu_util = inst.get("cpu_util")
    cpu_html = (
        _pct_bar(cpu_util)
        if not booting and cpu_util is not None
        else '<span class="muted">-</span>'
    )

    # Cost/hr
    dph_total = inst.get("dph_total")
    dph_base  = inst.get("dph_base")
    if dph_total is not None:
        base_note = (
            f' <span class="muted">(base ${dph_base:.3f})</span>'
            if dph_base and abs(dph_total - dph_base) > 0.001
            else ""
        )
        cost_html = f"${dph_total:.3f}/hr{base_note}"
    else:
        cost_html = '<span class="muted">-</span>'

    # Total spend: derive from start_date (Unix timestamp when billing began).
    # client_run_time resets on restart; duration is host uptime — both unreliable.
    start_date = inst.get("start_date")
    if dph_total is not None and start_date is not None:
        age_hours = (time.time() - start_date) / 3600
        spend_html = f"<b>${age_hours * dph_total:.2f}</b>"
    else:
        spend_html = '<span class="muted">-</span>'

    # Uptime (current boot session)
    uptime_mins = inst.get("uptime_mins")
    uptime_html = _format_uptime(uptime_mins) if uptime_mins is not None else '<span class="muted">-</span>'

    # Network bandwidth (useful for spotting rclone transfers)
    inet_down = inst.get("inet_down")
    inet_up   = inst.get("inet_up")
    if inet_down is not None and inet_up is not None:
        net_html = f"↓ {inet_down:.0f} MB/s &nbsp; ↑ {inet_up:.0f} MB/s"
    else:
        net_html = '<span class="muted">-</span>'

    # SSH command (uses ssh_host + ssh_port from API — confirmed reliable fields)
    ssh_host = inst.get("ssh_host", "")
    ssh_port = inst.get("ssh_port", "")
    ssh_html = (
        f'<code>ssh -p {ssh_port} root@{ssh_host}</code>'
        if ssh_host and ssh_port
        else '<span class="muted">-</span>'
    )

    # Disk
    disk_usage = inst.get("disk_usage")
    disk_space = inst.get("disk_space")
    disk_html  = (
        f"{disk_usage:.0f} / {disk_space:.0f} GB"
        if disk_usage is not None and disk_space is not None
        else '<span class="muted">-</span>'
    )

    geolocation = inst.get("geolocation", "")
    image       = inst.get("image_uuid", "")

    rows = [
        (t("infra_status"),      _instance_status_badge(actual, cur)),
        (t("infra_gpu_type"),    gpu_label),
        (t("infra_gpu_util"),    gpu_util_html),
        (t("infra_vram"),        vram_html),
        (t("infra_gpu_temp"),    temp_html),
        (t("infra_cpu_util"),    cpu_html),
        (t("infra_disk"),        disk_html),
        (t("infra_network"),     net_html),
        (t("infra_cost_hr"),     cost_html),
        (t("infra_total_spend"), spend_html),
        (t("infra_uptime"),      uptime_html),
        (t("infra_ssh"),         ssh_html),
        (t("infra_instance_id"), f'<code>{iid}</code>'),
        (t("infra_location"),    geolocation or '<span class="muted">-</span>'),
        (t("infra_image"),       f'<span class="muted" style="font-size:0.8em">{image}</span>' if image else '<span class="muted">-</span>'),
    ]

    rows_html = "".join(
        f'<tr><td class="label">{label}</td><td>{value}</td></tr>'
        for label, value in rows
    )
    return f"""
    <div style="margin-bottom:16px">
      <table style="width:100%;border-collapse:collapse">
        {rows_html}
      </table>
    </div>
    """


def _build_vastai_section_html(instances: list[dict], source: str, fetch_error: str, api_key: str) -> str:
    if fetch_error and not instances:
        return f'<p class="badge-bad">Failed to fetch instances: {fetch_error}</p>'
    if not instances:
        key_note = (
            '<p class="muted" style="font-size:0.85em">'
            'Set <code>VAST_API_KEY</code> or run '
            '<code>vastai set api-key YOUR_KEY</code></p>'
        ) if not api_key else ""
        return '<p class="badge-ok">No running instances found.</p>' + key_note
    source_note = f'<p class="note">Source: {source} &nbsp;·&nbsp; {len(instances)} instance(s)</p>'
    return source_note + "".join(_build_instance_html(i) for i in instances)


def _fetch_vastai_section() -> None:
    """Fetch Vast.ai instances in a background thread and update the cache."""
    global _vastai_checking
    api_key     = _get_vastai_api_key()
    instances: list[dict] = []
    source      = ""
    fetch_error = ""

    if api_key:
        try:
            instances = _fetch_vastai_instances_api(api_key)
            source = "REST API"
        except Exception as e:
            fetch_error = str(e)

    if not instances and not fetch_error:
        try:
            instances = _fetch_vastai_instances_cli()
            source = "vastai CLI"
        except Exception as e:
            fetch_error = str(e)

    html = _build_vastai_section_html(instances, source, fetch_error, api_key)

    with _vastai_lock:
        _vastai_cache["html"] = html
        _vastai_cache["ts"] = time.time()
        _vastai_checking = False


def _get_vastai_section_html() -> str:
    """Return cached Vast.ai section HTML; trigger background refresh if stale."""
    global _vastai_checking
    with _vastai_lock:
        age = time.time() - _vastai_cache["ts"]
        cached_html = _vastai_cache["html"]
        should_refresh = age > VASTAI_CACHE_TTL_S and not _vastai_checking
        if should_refresh:
            _vastai_checking = True

    if should_refresh:
        _threading.Thread(target=_fetch_vastai_section, daemon=True).start()

    return cached_html


def build_infra_html() -> str:
    # Both Vast.ai and Drive sections are non-blocking — return cached values
    # and trigger background refreshes if stale.
    gdrive_rows = f"""
    <tr><td class="label">{t("infra_drive_remote")}</td>
        <td><code>{DEFAULT_GDRIVE_REMOTE}</code></td></tr>
    <tr><td class="label">{t("infra_drive_dir")}</td>
        <td><code>{DEFAULT_GDRIVE_DIR}</code></td></tr>
    <tr><td class="label">{t("infra_status")}</td>
        <td>{_get_drive_status_html()}</td></tr>
    """

    return f"""
    <div class="train-summary">
      <h3>{t("infra_vastai_header")}</h3>
      {_get_vastai_section_html()}

      <h3>{t("infra_gdrive_header")}</h3>
      <table style="width:100%;border-collapse:collapse">
        {gdrive_rows}
      </table>
    </div>
    """


def refresh_infra() -> str:
    try:
        return build_infra_html()
    except Exception as e:
        return f'<p class="badge-bad">Error: {e}</p>'


# ── Panel 5: Agent Team Status ────────────────────────────────────────────────

# Default path to the agent tracker SQLite database
_AGENT_DB_DEFAULT = _REPO_ROOT / ".claude" / "mcp" / "agent_tracker" / "agent_tracker.db"

# Cache TTL for agent team status (seconds)
AGENTS_CACHE_TTL_S = 15

# Maximum recent messages shown
AGENTS_MSG_LIMIT = 20

# Max characters of message summary shown in table
AGENTS_MSG_PREVIEW_CHARS = 120

_agents_cache: dict = {"html": '<p class="muted">loading…</p>', "ts": 0.0}
_agents_lock = _threading.Lock()
_agents_checking = False


def _agent_status_badge(status: str) -> str:
    s = (status or "").lower()
    if s == "working":
        return f'<span class="badge-good">{s}</span>'
    if s == "idle":
        return f'<span class="muted">{s}</span>'
    if s in ("blocked", "error"):
        return f'<span class="badge-bad">{s}</span>'
    return f'<span class="badge-info">{s or "unknown"}</span>'


def _task_status_badge(status: str) -> str:
    s = (status or "").lower()
    if s == "done":
        return f'<span class="badge-good">{s}</span>'
    if s == "in_progress":
        return f'<span class="badge-ok">in progress</span>'
    if s == "pending":
        return f'<span class="muted">{s}</span>'
    return f'<span class="badge-info">{s or "unknown"}</span>'


def _time_ago(ts_str: str) -> str:
    """Convert ISO timestamp string to a human-readable 'X ago' string."""
    try:
        import datetime
        ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        age_s = (now - ts).total_seconds()
        if age_s < 60:
            return f"{int(age_s)}s ago"
        if age_s < 3600:
            return f"{int(age_s // 60)}m ago"
        if age_s < 86400:
            return f"{age_s / 3600:.1f}h ago"
        return f"{age_s / 86400:.1f}d ago"
    except Exception:
        return ts_str or ""


def _build_agents_html_from_db() -> str:
    import sqlite3

    db_path = _AGENT_DB_DEFAULT
    if not db_path.exists():
        return (
            f'<p class="badge-bad">{t("agents_db_not_found")}</p>'
            f'<p class="note">{t("agents_db_path")}: <code>{db_path}</code></p>'
        )

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # ── Agents ──────────────────────────────────────────────────────────────
        cur.execute(
            "SELECT name, role, status, current_task, turns_used, max_turns,"
            "       last_activity_at"
            "  FROM agents"
            " ORDER BY CASE status WHEN 'working' THEN 0 ELSE 1 END, last_activity_at DESC"
        )
        agents = cur.fetchall()

        # ── Tasks ────────────────────────────────────────────────────────────────
        cur.execute("SELECT status, COUNT(*) AS cnt FROM tasks GROUP BY status")
        task_counts: dict[str, int] = {r["status"]: r["cnt"] for r in cur.fetchall()}
        pending_cnt    = task_counts.get("pending", 0)
        in_progress_cnt = task_counts.get("in_progress", 0)
        done_cnt       = task_counts.get("done", 0)

        cur.execute(
            "SELECT id, description, assigned_to, status FROM tasks"
            " WHERE status = 'in_progress'"
            " ORDER BY id"
        )
        active_tasks = cur.fetchall()

        # ── Messages ─────────────────────────────────────────────────────────────
        cur.execute(
            "SELECT from_agent, to_agent, summary, timestamp FROM messages"
            " ORDER BY timestamp DESC LIMIT ?",
            (AGENTS_MSG_LIMIT,),
        )
        messages = cur.fetchall()
        conn.close()
    except Exception as exc:
        return f'<p class="badge-bad">DB error: {exc}</p>'

    # ── Build agents table HTML ──────────────────────────────────────────────────
    if not agents:
        agents_rows_html = f'<tr><td colspan="5" class="muted">{t("agents_no_agents")}</td></tr>'
        working_count = idle_count = 0
    else:
        working_count = sum(1 for a in agents if (a["status"] or "").lower() == "working")
        idle_count    = len(agents) - working_count
        agents_rows_html = ""
        for a in agents:
            role_short = (a["role"] or "").split("—")[0].strip()
            role_short = role_short[:60] + ("…" if len(role_short) > 60 else "")
            task_str   = a["current_task"] or f'<span class="muted">—</span>'
            turns_str  = f'{a["turns_used"]}/{a["max_turns"]}'
            age_str    = _time_ago(a["last_activity_at"] or "")
            agents_rows_html += f"""
            <tr>
              <td><b>{a["name"]}</b><br>
                  <span class="agent-role">{role_short}</span></td>
              <td>{_agent_status_badge(a["status"])}</td>
              <td style="font-size:0.85em">{task_str}</td>
              <td style="text-align:center;font-family:monospace">{turns_str}</td>
              <td class="muted" style="font-size:0.8em">{age_str}</td>
            </tr>
            """

    summary_line = (
        f'<p class="note" style="margin-bottom:6px">'
        f'{t("agents_summary_label").format(working=working_count, idle=idle_count, total=len(agents))}'
        f'</p>'
    )

    agents_table_html = f"""
    {summary_line}
    <table class="agent-table">
      <thead>
        <tr>
          <th>{t("agents_col_name")}</th>
          <th>{t("agents_col_status")}</th>
          <th>{t("agents_col_task")}</th>
          <th style="text-align:center">{t("agents_col_turns")}</th>
          <th>{t("agents_col_last_active")}</th>
        </tr>
      </thead>
      <tbody>
        {agents_rows_html}
      </tbody>
    </table>
    """

    # ── Build task summary HTML ──────────────────────────────────────────────────
    total_tasks = pending_cnt + in_progress_cnt + done_cnt
    task_counts_html = f"""
    <div class="task-counts">
      <div class="task-count-box" style="background:rgba(0,170,102,0.12);border:1px solid #00aa66">
        <span class="count" style="color:#00aa66">{done_cnt}</span>
        <span class="label">{t("agents_tasks_done")}</span>
      </div>
      <div class="task-count-box" style="background:rgba(204,136,0,0.12);border:1px solid #cc8800">
        <span class="count" style="color:#cc8800">{in_progress_cnt}</span>
        <span class="label">{t("agents_tasks_in_progress")}</span>
      </div>
      <div class="task-count-box" style="background:rgba(136,136,136,0.12);border:1px solid #666">
        <span class="count" style="color:#aaa">{pending_cnt}</span>
        <span class="label">{t("agents_tasks_pending")}</span>
      </div>
    </div>
    """

    if active_tasks:
        active_rows_html = ""
        for task in active_tasks:
            desc_short = (task["description"] or "")[:100]
            if len(task["description"] or "") > 100:
                desc_short += "…"
            assignee = task["assigned_to"] or f'<span class="muted">—</span>'
            active_rows_html += f"""
            <tr>
              <td style="text-align:center;font-family:monospace">{task["id"]}</td>
              <td style="font-size:0.85em">{desc_short}</td>
              <td>{assignee}</td>
            </tr>
            """
        active_tasks_html = f"""
        <h3 style="color:#cc8800;margin:8px 0 4px">{t("agents_in_progress_tasks")}</h3>
        <table class="agent-table">
          <thead>
            <tr>
              <th style="width:40px">{t("agents_col_task_id")}</th>
              <th>{t("agents_col_task_desc")}</th>
              <th>{t("agents_col_task_assigned")}</th>
            </tr>
          </thead>
          <tbody>{active_rows_html}</tbody>
        </table>
        """
    else:
        active_tasks_html = ""

    tasks_html = f"""
    <div class="train-summary">
      <h3>{t("agents_header_tasks")}</h3>
      {task_counts_html}
      {active_tasks_html}
    </div>
    """

    # ── Build messages HTML ──────────────────────────────────────────────────────
    if not messages:
        msg_rows_html = f'<tr><td colspan="4" class="muted">{t("agents_no_messages")}</td></tr>'
    else:
        msg_rows_html = ""
        for msg in messages:
            preview = (msg["summary"] or "")[:AGENTS_MSG_PREVIEW_CHARS]
            if len(msg["summary"] or "") > AGENTS_MSG_PREVIEW_CHARS:
                preview += "…"
            age_str = _time_ago(msg["timestamp"] or "")
            msg_rows_html += f"""
            <tr>
              <td><b>{msg["from_agent"]}</b></td>
              <td>{msg["to_agent"]}</td>
              <td class="msg-summary" title="{(msg['summary'] or '').replace(chr(34), '&quot;')}">{preview}</td>
              <td class="muted" style="font-size:0.8em;white-space:nowrap">{age_str}</td>
            </tr>
            """

    messages_html = f"""
    <div class="train-summary" style="margin-top:16px">
      <h3>{t("agents_header_messages")}</h3>
      <table class="agent-table">
        <thead>
          <tr>
            <th>{t("agents_col_msg_from")}</th>
            <th>{t("agents_col_msg_to")}</th>
            <th>{t("agents_col_msg_summary")}</th>
            <th>{t("agents_col_msg_time")}</th>
          </tr>
        </thead>
        <tbody>{msg_rows_html}</tbody>
      </table>
    </div>
    """

    return f"""
    <div class="train-summary">
      <h3>{t("agents_header_agents")}</h3>
      {agents_table_html}
    </div>
    {tasks_html}
    {messages_html}
    <p class="note">{t("agents_db_path")}: <code>{db_path}</code></p>
    """


def _fetch_agents_section() -> None:
    """Fetch agent team status in a background thread and update the cache."""
    global _agents_checking
    try:
        html = _build_agents_html_from_db()
    except Exception as exc:
        html = f'<p class="badge-bad">Error: {exc}</p>'
    with _agents_lock:
        _agents_cache["html"] = html
        _agents_cache["ts"] = time.time()
        _agents_checking = False


def _get_agents_section_html() -> str:
    """Return cached agent status HTML; trigger background refresh if stale."""
    global _agents_checking
    with _agents_lock:
        age = time.time() - _agents_cache["ts"]
        cached_html = _agents_cache["html"]
        should_refresh = age > AGENTS_CACHE_TTL_S and not _agents_checking
        if should_refresh:
            _agents_checking = True
    if should_refresh:
        _threading.Thread(target=_fetch_agents_section, daemon=True).start()
    return cached_html


def refresh_agents() -> str:
    try:
        return _build_agents_html_from_db()
    except Exception as e:
        return f'<p class="badge-bad">Error: {e}</p>'


# ── Gradio app ────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title=t("dashboard_title"),
        analytics_enabled=False,
    ) as app:
        gr.Markdown(f"# {t('dashboard_title')}")
        gr.Markdown(t("dashboard_desc"))

        with gr.Tabs():

            # ── Tab 1: Datasets ───────────────────────────────────────────
            with gr.TabItem(t("tab_datasets")):
                gr.Markdown(t("ds_desc"))
                with gr.Row():
                    ds_btn = gr.Button(t("btn_refresh"), variant="primary", size="sm")
                # No value= here — populated via app.load() to avoid blocking startup
                ds_html = gr.HTML()
                ds_btn.click(fn=refresh_datasets, inputs=[], outputs=[ds_html])

            # ── Tab 2: Training ───────────────────────────────────────────
            with gr.TabItem(t("tab_training")):
                gr.Markdown(t("train_desc"))
                with gr.Row():
                    train_btn = gr.Button(t("btn_refresh"), variant="primary", size="sm")
                    auto_cb = gr.Checkbox(
                        label=t("train_auto_refresh").format(interval=AUTO_REFRESH_INTERVAL_S),
                        value=False,
                    )
                train_html = gr.HTML()
                with gr.Row():
                    with gr.Column():
                        g_plot = gr.Plot(label=t("train_g_loss_plot"))
                    with gr.Column():
                        d_plot = gr.Plot(label=t("train_d_loss_plot"))

                train_btn.click(
                    fn=refresh_training,
                    inputs=[],
                    outputs=[train_html, g_plot, d_plot],
                )
                timer = gr.Timer(value=AUTO_REFRESH_INTERVAL_S, active=False)
                timer.tick(fn=refresh_training, inputs=[], outputs=[train_html, g_plot, d_plot])
                auto_cb.change(
                    fn=lambda active: gr.update(active=active),
                    inputs=[auto_cb],
                    outputs=[timer],
                )

            # ── Tab 3: Checkpoints ────────────────────────────────────────
            with gr.TabItem(t("tab_checkpoints")):
                gr.Markdown(t("ckpt_desc"))
                with gr.Row():
                    ckpt_btn = gr.Button(t("btn_refresh"), variant="primary", size="sm")
                ckpt_html = gr.HTML()
                ckpt_btn.click(fn=refresh_checkpoints, inputs=[], outputs=[ckpt_html])

            # ── Tab 4: Infrastructure ─────────────────────────────────────
            with gr.TabItem(t("tab_infrastructure")):
                gr.Markdown(t("infra_desc"))
                with gr.Row():
                    infra_btn = gr.Button(t("btn_refresh"), variant="primary", size="sm")
                infra_html = gr.HTML()
                infra_btn.click(fn=refresh_infra, inputs=[], outputs=[infra_html])

            # ── Tab 5: Agent Team Status ───────────────────────────────────
            with gr.TabItem(t("tab_agents")):
                gr.Markdown(t("agents_desc"))
                with gr.Row():
                    agents_btn = gr.Button(t("btn_refresh"), variant="primary", size="sm")
                    agents_auto_cb = gr.Checkbox(
                        label=t("train_auto_refresh").format(interval=AGENTS_CACHE_TTL_S),
                        value=False,
                    )
                agents_html = gr.HTML()
                agents_btn.click(fn=refresh_agents, inputs=[], outputs=[agents_html])
                agents_timer = gr.Timer(value=AGENTS_CACHE_TTL_S, active=False)
                agents_timer.tick(fn=refresh_agents, inputs=[], outputs=[agents_html])
                agents_auto_cb.change(
                    fn=lambda active: gr.update(active=active),
                    inputs=[agents_auto_cb],
                    outputs=[agents_timer],
                )

        # Populate all tabs after the server is up (avoids blocking startup with
        # slow rclone/API calls during gr.HTML(value=fn) eager evaluation)
        app.load(fn=refresh_datasets,   inputs=[], outputs=[ds_html])
        app.load(fn=refresh_training,   inputs=[], outputs=[train_html, g_plot, d_plot])
        app.load(fn=refresh_checkpoints, inputs=[], outputs=[ckpt_html])
        app.load(fn=refresh_infra,       inputs=[], outputs=[infra_html])
        app.load(fn=refresh_agents,      inputs=[], outputs=[agents_html])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _log_dir, _checkpoint_dir, _data_dir, _lang

    parser = argparse.ArgumentParser(description="Audio Enhancer monitoring dashboard")
    parser.add_argument(
        "--lang", default="en", choices=["en", "ru"],
        help="UI language (default: en)",
    )
    parser.add_argument(
        "--log-dir", default=str(_REPO_ROOT / DEFAULT_LOG_DIR),
        help=f"TensorBoard log directory (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--checkpoint-dir", default=str(_REPO_ROOT / DEFAULT_CHECKPOINT_DIR),
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--data-dir", default=str(_REPO_ROOT / DEFAULT_DATA_DIR),
        help=f"Datasets raw directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--port", default=DEFAULT_PORT, type=int,
        help=f"Port to serve on (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()

    _lang = args.lang
    _log_dir = Path(args.log_dir)
    _checkpoint_dir = Path(args.checkpoint_dir)
    _data_dir = Path(args.data_dir)
    _ds_mod.RAW_DIR = _data_dir

    app = build_app()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        theme=gr.themes.Soft(primary_hue="cyan"),
        css=DASHBOARD_CSS,
    )


if __name__ == "__main__":
    main()
