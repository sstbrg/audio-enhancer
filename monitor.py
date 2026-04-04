#!/usr/bin/env python3
"""Training and dataset monitoring dashboard.

Usage:
    python monitor.py                              # defaults
    python monitor.py --log-dir path/to/logs       # custom TensorBoard log dir
    python monitor.py --data-dir path/to/datasets  # custom datasets dir
    python monitor.py --port 7861                  # custom port (default 7861)
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import gradio as gr

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PORT = 7861
DEFAULT_LOG_DIR = "checkpoints/phase0/logs"
DEFAULT_DATA_DIR = "datasets/raw"

# Fraction of expected size that counts as "ready"
READY_THRESHOLD = 0.9

# Auto-refresh interval in seconds
AUTO_REFRESH_INTERVAL_S = 30

# Maximum number of scalar steps to load per tag (keeps UI fast)
MAX_STEPS = 2000

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

# ── Configuration (set at startup via CLI) ────────────────────────────────────

_log_dir: Path = Path(DEFAULT_LOG_DIR)
_data_dir: Path = Path(DEFAULT_DATA_DIR)


# ── Gradio CSS (mirrors analyzer_ui.py palette) ───────────────────────────────

DASHBOARD_CSS = """
.gradio-container { max-width: 95% !important; margin: 0 auto !important; }

/* Status table */
.ds-table { font-family: 'Segoe UI', monospace; width: 100%;
            border-collapse: collapse; margin-bottom: 10px; }
.ds-table th  { color: #0088cc; text-align: left; padding: 6px 10px;
                border-bottom: 1px solid #333; }
.ds-table td  { padding: 5px 10px; }
.ds-table tr:nth-child(even) td { background: rgba(255,255,255,0.03); }

/* Status badges */
.badge-good   { color: #00aa66; font-weight: bold; }
.badge-ok     { color: #cc8800; font-weight: bold; }
.badge-bad    { color: #cc3333; font-weight: bold; }
.badge-info   { color: #0088cc; font-weight: bold; }
.muted        { color: #888; }

/* Training summary */
.train-summary { font-family: 'Segoe UI', sans-serif; padding: 8px 0; }
.train-summary table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
.train-summary td.label { color: #888; padding: 4px 8px; width: 45%; }
.train-summary td.value { padding: 4px 8px; font-weight: bold; }
.train-summary h3 { color: #0088cc; margin: 8px 0 4px; }

/* Footer note */
.note { color: #888; font-size: 0.82em; margin-top: 6px; }
"""

# ── Dataset helpers (import from infra/datasets.py) ──────────────────────────

def _import_datasets_module():
    """Import infra/datasets.py without relying on it being a package."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "datasets_infra",
        Path(__file__).parent / "infra" / "datasets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["datasets_infra"] = mod
    spec.loader.exec_module(mod)
    return mod


_ds_mod = _import_datasets_module()
CATALOG = _ds_mod.CATALOG
local_size_bytes = _ds_mod.local_size_bytes
human_size = _ds_mod.human_size
is_on_drive = _ds_mod.is_on_drive
is_downloading = _ds_mod.is_downloading
is_uploading = _ds_mod.is_uploading
local_path = _ds_mod.local_path


# ── Dataset status logic ──────────────────────────────────────────────────────

def _dataset_status_label(ds, local_bytes: int, on_drive: bool) -> str:
    """Return an HTML status label for a dataset row."""
    expected_bytes = int(ds.expected_gb * (1 << 30))

    if is_uploading(ds):
        return '<span class="badge-info">uploading to Drive</span>'
    if is_downloading(ds):
        pct = min(100, int(local_bytes * 100 / expected_bytes)) if expected_bytes > 0 else 0
        return f'<span class="badge-ok">downloading {pct}%</span>'
    if local_bytes >= expected_bytes * READY_THRESHOLD:
        if on_drive:
            return '<span class="badge-good">ready + Drive</span>'
        return '<span class="badge-good">ready</span>'
    if local_bytes > 0:
        pct = min(100, int(local_bytes * 100 / expected_bytes)) if expected_bytes > 0 else 0
        return f'<span class="badge-bad">partial ({pct}%)</span>'
    if on_drive:
        return '<span class="badge-info">on Drive only</span>'
    if ds.source == "manual":
        return '<span class="muted">manual download</span>'
    return '<span class="muted">not downloaded</span>'


def build_dataset_table() -> str:
    """Build the full datasets HTML table including disk-free footer."""
    rows_html = ""
    total_local = 0
    total_expected = 0

    for ds in CATALOG:
        local_bytes = local_size_bytes(ds)
        expected_bytes = int(ds.expected_gb * (1 << 30))
        total_local += local_bytes
        total_expected += expected_bytes

        local_str = human_size(local_bytes) if local_bytes > 0 else '<span class="muted">-</span>'
        expected_str = f"{ds.expected_gb:.0f} GB"

        on_drive = is_on_drive(ds)
        drive_str = (
            '<span class="badge-good">yes</span>'
            if on_drive
            else '<span class="muted">-</span>'
        )
        status_html = _dataset_status_label(ds, local_bytes, on_drive)

        rows_html += f"""
        <tr>
          <td><b>{ds.name}</b><br><span class="muted" style="font-size:0.8em">{ds.id}</span></td>
          <td>{expected_str}</td>
          <td>{local_str}</td>
          <td style="text-align:center">{drive_str}</td>
          <td>{status_html}</td>
        </tr>
        """

    # Disk free
    try:
        data_path = _data_dir if _data_dir.exists() else Path(".")
        disk = shutil.disk_usage(data_path)
        disk_free_str = human_size(disk.free)
        disk_total_str = human_size(disk.total)
        disk_used_pct = int(disk.used * 100 / disk.total)
    except Exception:
        disk_free_str = "unknown"
        disk_total_str = "unknown"
        disk_used_pct = 0

    footer = f"""
    <tr style="border-top:1px solid #333">
      <td colspan="2"><b>Totals</b></td>
      <td><b>{human_size(total_local)}</b> / {human_size(total_expected)}</td>
      <td colspan="2" class="muted">
        Disk: {disk_free_str} free / {disk_total_str} total ({disk_used_pct}% used)
      </td>
    </tr>
    """

    html = f"""
    <table class="ds-table">
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Expected</th>
          <th>Local size</th>
          <th>On Drive</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
        {footer}
      </tbody>
    </table>
    <p class="note">Data dir: {_data_dir.resolve()}</p>
    """
    return html


def refresh_datasets() -> str:
    try:
        return build_dataset_table()
    except Exception as e:
        return f'<p class="badge-bad">Error loading dataset status: {e}</p>'


# ── TensorBoard event reading ─────────────────────────────────────────────────

def _read_tb_scalars(log_dir: Path, tags: list[str]) -> dict[str, tuple[list, list]]:
    """
    Read scalar series from TensorBoard event files.
    Returns {tag: (steps, values)} for every tag that has data.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    if not log_dir.exists():
        return {}

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": MAX_STEPS})
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            result[tag] = (steps, values)
    return result


def _find_latest_checkpoint(base_dir: Path) -> str | None:
    """Return the path of the most recently modified .pt checkpoint, or None."""
    candidates = list(base_dir.glob("*.pt"))
    if not candidates:
        # Walk one level deeper (e.g. checkpoints/phase0/)
        candidates = list(base_dir.glob("**/*.pt"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _steps_per_sec(steps: list, timestamps: list | None = None) -> float | None:
    """
    Estimate steps/sec from the last portion of the step sequence.
    If TensorBoard wall-time data is available use it; otherwise return None.
    """
    if timestamps and len(timestamps) >= 2:
        dt = timestamps[-1] - timestamps[-2]
        ds = steps[-1] - steps[-2]
        if dt > 0 and ds > 0:
            return ds / dt
    return None


def build_training_summary() -> tuple[str, list, list, list, list]:
    """
    Returns:
        summary_html  – HTML text block with epoch/step/speed/checkpoint info
        g_steps       – x-axis for generator loss plot
        g_loss_series – list of (name, values) pairs for generator losses
        d_steps       – x-axis for discriminator loss plot
        d_loss_series – list of (name, values) pairs for discriminator losses
    """
    log_dir = _log_dir
    no_data_html = (
        f'<div class="train-summary">'
        f'<p class="muted">No TensorBoard event files found in:<br>'
        f'<code>{log_dir.resolve()}</code></p>'
        f'<p class="note">Start training to populate this tab.</p>'
        f'</div>'
    )

    all_tags = GENERATOR_LOSS_TAGS + DISCRIMINATOR_LOSS_TAGS
    scalars = _read_tb_scalars(log_dir, all_tags)

    if not scalars:
        return no_data_html, [], [], [], []

    # ── Step / epoch info ───────────────────────────────────────────────────
    # Use the primary g_loss or d_loss tag to determine current step
    ref_tag = None
    for t in ["train/g_loss", "train/d_loss"] + list(scalars.keys()):
        if t in scalars:
            ref_tag = t
            break

    current_step = 0
    current_epoch = 0
    speed_str = "unknown"

    if ref_tag:
        steps, _ = scalars[ref_tag]
        current_step = steps[-1] if steps else 0

        # Try to read epoch from a dedicated scalar, else estimate from step
        if "train/epoch" in scalars:
            current_epoch = int(scalars["train/epoch"][1][-1])
        else:
            # Estimate: no per-epoch info available from scalars alone
            current_epoch = "unknown"

        # Speed estimation via wall-time from TensorBoard (requires raw events)
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(log_dir), size_guidance={"scalars": MAX_STEPS})
            ea.Reload()
            if ref_tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(ref_tag)
                if len(events) >= 2:
                    wall_times = [e.wall_time for e in events[-10:]]
                    event_steps = [e.step for e in events[-10:]]
                    if wall_times[-1] - wall_times[0] > 0:
                        total_steps = event_steps[-1] - event_steps[0]
                        total_time = wall_times[-1] - wall_times[0]
                        sps = total_steps / total_time if total_time > 0 else 0
                        if sps > 0:
                            speed_str = f"{sps:.2f} steps/sec ({1/sps:.1f} sec/step)"
        except Exception:
            pass

    # ── Checkpoint info ─────────────────────────────────────────────────────
    ckpt_dir = log_dir.parent  # logs/ lives inside the checkpoint dir
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        ckpt_path = Path(latest_ckpt)
        import time
        mtime = ckpt_path.stat().st_mtime
        age_s = time.time() - mtime
        if age_s < 3600:
            age_str = f"{int(age_s // 60)} min ago"
        elif age_s < 86400:
            age_str = f"{age_s / 3600:.1f} h ago"
        else:
            age_str = f"{age_s / 86400:.1f} days ago"
        ckpt_str = f"{ckpt_path.name} ({age_str})"
    else:
        ckpt_str = '<span class="muted">none found</span>'

    summary_html = f"""
    <div class="train-summary">
      <h3>Training State</h3>
      <table>
        <tr><td class="label">Current step</td>
            <td class="value">{current_step:,}</td></tr>
        <tr><td class="label">Current epoch</td>
            <td class="value">{current_epoch}</td></tr>
        <tr><td class="label">Training speed</td>
            <td class="value">{speed_str}</td></tr>
        <tr><td class="label">Last checkpoint</td>
            <td class="value">{ckpt_str}</td></tr>
        <tr><td class="label">Log directory</td>
            <td class="value muted" style="font-size:0.85em">{log_dir.resolve()}</td></tr>
      </table>
    </div>
    """

    # ── Build series for plots ───────────────────────────────────────────────
    g_steps: list[int] = []
    g_loss_series: list[tuple[str, list[float]]] = []
    for tag in GENERATOR_LOSS_TAGS:
        if tag in scalars:
            steps_t, vals = scalars[tag]
            label = tag.split("/")[-1]  # e.g. "g_loss"
            if not g_steps:
                g_steps = steps_t
            g_loss_series.append((label, vals))

    d_steps: list[int] = []
    d_loss_series: list[tuple[str, list[float]]] = []
    for tag in DISCRIMINATOR_LOSS_TAGS:
        if tag in scalars:
            steps_t, vals = scalars[tag]
            label = tag.split("/")[-1]
            if not d_steps:
                d_steps = steps_t
            d_loss_series.append((label, vals))

    return summary_html, g_steps, g_loss_series, d_steps, d_loss_series


def _make_loss_figure(steps: list, series: list[tuple[str, list]], title: str):
    """Return a matplotlib Figure for loss curves, or None if no data."""
    if not steps or not series:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Colour palette consistent with analyzer_ui.py dark theme
    COLORS = ["#00d4ff", "#ff6b6b", "#ffd166", "#06d6a0", "#a8dadc", "#e63946", "#457b9d"]

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
        color = COLORS[i % len(COLORS)]
        ax.plot(steps[:len(values)], values, label=label, color=color, linewidth=1.2, alpha=0.85)

    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ccc", fontsize=9)
    plt.tight_layout()
    return fig


def refresh_training():
    """Callable wired to Gradio outputs for the Training tab."""
    summary_html, g_steps, g_series, d_steps, d_series = build_training_summary()
    g_fig = _make_loss_figure(g_steps, g_series, "Generator Losses")
    d_fig = _make_loss_figure(d_steps, d_series, "Discriminator Losses")
    return summary_html, g_fig, d_fig


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Audio Enhancer Monitor",
    theme=gr.themes.Soft(primary_hue="cyan"),
    css=DASHBOARD_CSS,
) as app:
    gr.Markdown("# Audio Enhancer — Monitor")

    with gr.Tabs():

        # ── Tab 1: Datasets ───────────────────────────────────────────────────
        with gr.TabItem("Datasets"):
            gr.Markdown(
                "Dataset sync status: local disk vs Google Drive. "
                "Drive checks may take a few seconds (rclone)."
            )
            with gr.Row():
                ds_refresh_btn = gr.Button("Refresh", variant="primary", size="sm")

            ds_table_html = gr.HTML(value=refresh_datasets)

            ds_refresh_btn.click(fn=refresh_datasets, inputs=[], outputs=[ds_table_html])

        # ── Tab 2: Training ───────────────────────────────────────────────────
        with gr.TabItem("Training"):
            gr.Markdown(
                "Training progress from TensorBoard event files. "
                "Toggle auto-refresh to poll every 30 seconds."
            )
            with gr.Row():
                train_refresh_btn = gr.Button("Refresh", variant="primary", size="sm")
                auto_refresh_toggle = gr.Checkbox(
                    label=f"Auto-refresh every {AUTO_REFRESH_INTERVAL_S}s",
                    value=False,
                )

            with gr.Row():
                train_summary_html = gr.HTML()

            with gr.Row():
                with gr.Column():
                    g_loss_plot = gr.Plot(label="Generator Losses")
                with gr.Column():
                    d_loss_plot = gr.Plot(label="Discriminator Losses")

            # Manual refresh
            train_refresh_btn.click(
                fn=refresh_training,
                inputs=[],
                outputs=[train_summary_html, g_loss_plot, d_loss_plot],
            )

            # Auto-refresh via gr.Timer
            timer = gr.Timer(value=AUTO_REFRESH_INTERVAL_S, active=False)
            timer.tick(
                fn=refresh_training,
                inputs=[],
                outputs=[train_summary_html, g_loss_plot, d_loss_plot],
            )

            # Toggle wires the checkbox to the timer's active state
            auto_refresh_toggle.change(
                fn=lambda active: gr.Timer(active=active),
                inputs=[auto_refresh_toggle],
                outputs=[timer],
            )

            # Populate on first load
            app.load(
                fn=refresh_training,
                inputs=[],
                outputs=[train_summary_html, g_loss_plot, d_loss_plot],
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _log_dir, _data_dir

    parser = argparse.ArgumentParser(description="Audio Enhancer monitoring dashboard")
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"TensorBoard log directory (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Datasets directory for disk-usage stats (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port to serve on (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()

    _log_dir = Path(args.log_dir)
    _data_dir = Path(args.data_dir)

    # Override the datasets module's RAW_DIR so local_path() resolves correctly
    _ds_mod.RAW_DIR = _data_dir

    app.queue()
    app.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
