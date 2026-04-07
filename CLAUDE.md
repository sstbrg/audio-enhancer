# Audio Enhancer

GAN-based audio super-resolution: upscales 48kHz → 96kHz / 24-bit.
Target quality: hi-fi playback, Dire Straits-level mastering.

## Project structure

```
train.py              # Training script (GAN with AMP, torch.compile)
enhance.py            # Inference: any audio → 96kHz/24-bit wav
evaluate.py           # CLI model evaluation
analyze.py            # CLI audio quality analysis
analyzer_ui.py        # Gradio web UI (Enhance + Analyze tabs, i18n EN/RU)
prepare_dataset.py    # Resample audio to target SR for training
models/
  constants.py        # All defaults: sample rates, architecture params, slopes
  generator.py        # HiFi-GAN generator (48kHz → 96kHz, 2x upsample)
  discriminator.py    # Multi-period (2,3,5,7,11,17,23) + multi-scale (3)
  losses.py           # STFT, mel spectrogram, adversarial, feature matching
  mastering_losses.py # Perceptual STFT, stereo, dynamics, encodec, CLAP, audiobox
data/
  dataset.py          # AudioSRDataset: creates (48kHz, 96kHz) pairs on the fly
  degradations.py     # (Phase 1) Degradation transforms: codec, EQ, compression, stereo, clipping, noise
  degradation_chain.py # (Phase 1) Chain builder + curriculum scheduler
  precompute_codecs.py # (Phase 1) Pre-compute MP3/AAC/OGG codec variants
  dataset_phase1.py   # (Phase 1) DegradedAudioDataset: (degraded_48k, clean_96k) pairs
metrics/
  evaluate.py           # SI-SNR, SDR, CDPAM, ViSQOL, Audiobox, PAM, MuQ-Eval, chroma/MFCC/onset
  music_analysis.py     # Genre, mood, instruments, key, BPM (Essentia, CLAP, MERT)
  upscale_potential.py  # Upscale potential assessment: sample rate ceiling, codec artifacts, bit depth headroom
configs/
  phase0.yaml         # Phase 0: 96kHz target, batch 8, checkpoint every epoch
  phase1.yaml         # (Phase 1) Degradation restoration: lower LR, curriculum, loss weights
  default.yaml        # Default config
locales/
  en.json             # English UI strings
  ru.json             # Russian UI strings
infra/
  datasets.py         # Dataset manager: download/upload/pull/status dashboard
  datasets.sh         # Shell version (deprecated, use .py)
  setup-vastai.sh     # One-command Vast.ai instance setup
  deploy.sh           # GCP deployment helper (kept for future use)
  main.tf             # Terraform config (GCP, kept for future use)
third_party/          # (gitignored) PAM, MuQ-Eval clones
tests/
  test_compile_amp.py   # torch.compile + AMP compatibility tests (Kyle)
  test_mastering_losses.py # mastering_losses.py unit tests (Jack)
  test_checkpoint.py    # Checkpoint save/load round-trip + _unwrap_state_dict tests (Jack)
docs/
  ARCHITECTURE.md     # Detailed architecture: generator, discriminators, losses, data flow
  training_guide.md   # Step-by-step: Vast.ai setup, datasets, training, monitoring
  inference_guide.md  # enhance.py usage: formats, sample rate logic, batch processing
  infrastructure_guide.md # Vast.ai, rclone, GCP, monitoring setup
  phase0_training_analysis.md # Epoch 0 results, loss decomposition, optimization plan
  phase1_degradation_plan.md  # Phase 1 degradation pipeline design and file layout
  phase1_finetuning_recipe.md # Phase 1 fine-tuning spec: LR, loss weights, mixed batches (Lara)
  code_review_report.md # Florence/Florence-2 code review findings and fix status
  ai_strategy_review.md # AI/ML strategy analysis and recommendations
  model_architecture_review.md # Generator/discriminator architecture deep-dive
  system_integration_report.md # System integration status and cross-component review
  training_pipeline_audit.md # Training loop audit and performance analysis
ARCHITECTURE.md       # Quick architecture reference (diagrams)
```

## Training

Training runs on Vast.ai (RTX 3090 spot, ~$0.13-0.22/hr).

```bash
# On Vast.ai instance:
cd /workspace/audio-enhancer && source .venv/bin/activate
python train.py \
  --data_dir datasets/phase0_combined \
  --config configs/phase0.yaml \
  --checkpoint_dir checkpoints/phase0 \
  --max-hours 5
```

Optimizations: AMP (mixed precision), torch.compile, cudnn.benchmark, persistent DataLoader workers.
Checkpoints save every epoch. TensorBoard via `ssh -N -L 6006:localhost:6006`.
Resume: `--resume checkpoints/phase0/latest.pt`

## Current status

- Phase 0 epoch 0 complete, checkpoint saved to Google Drive
- Checkpoint: `gdrive:audio-enhancer-datasets/checkpoints/checkpoint_0000.pt`
- Losses at end of epoch 0: d≈4.2, g≈35 (stable, encodec spikes resolved). Analysis: `docs/phase0_training_analysis.md`
- Training optimizations (AMP, torch.compile) committed but not yet tested in production training
- `_unwrap_state_dict` helper wired into all checkpoint save paths in `train.py` — strips `_orig_mod.` prefix from compiled model state dicts (done)
- Vast.ai auto-shutdown after 15min idle (cron checks for train.py process)
- **Tests:** 176/176 pass (Jack). New tests: `tests/test_mastering_losses.py` (Jack), `tests/test_checkpoint.py` (Jack). More coverage tests pending.
- **Code review (Florence-2):** 5 critical issues found — weights_only=False in torch.load, temp file leak in validation, hardcoded nyquist assumption, magic sample rates in enhance.py, magic STFT params in losses.py. Do NOT merge to main until fixed. See `docs/code_review_report.md`.
- **Phase 1 implementation in progress:** degradations.py, degradation_chain.py, dataset_phase1.py, precompute_codecs.py, configs/phase1.yaml all added. train.py --phase flag added. Fine-tuning recipe: `docs/phase1_finetuning_recipe.md`.

## Datasets

Stored on Google Drive (`gdrive:audio-enhancer-datasets/`), pulled to Vast.ai instances.
Manage with: `python infra/datasets.py status|download|upload|pull|watch`

On Drive: EG-IPT (22GB), MUSDB18-HQ (22GB), VCTK 96kHz (20GB), MusicNet (11GB)
Downloading: MAESTRO (120GB), GTSinger (30GB) — may need re-download on next instance
Manual: MoisesDB (requested at developer.moises.ai), MedleyDB (requested at medleydb.weebly.com)

rclone config uses custom OAuth client ID (GCP project stoked-mapper-258810).

## Web UIs

| App | Port | File | Description |
|-----|------|------|-------------|
| Analyzer UI | 7860 | analyzer_ui.py | Enhance + Analyze tabs, i18n EN/RU |
| Monitor (deprecated) | 7861 | monitor.py | Superseded by infra/dashboard.py |
| Dashboard | 7862 | infra/dashboard.py | Datasets, Training, Checkpoints, Infrastructure, Agent Team tabs, i18n EN/RU |

### Analyzer GUI

```bash
python analyzer_ui.py           # English, http://localhost:7860
python analyzer_ui.py --lang ru  # Russian
```

Two tabs: Enhance (apply model) + Analyze (quality metrics).
All strings from locale files, all styles in CSS. Supports wav/flac/mp3/ogg/webm.

### Monitoring Dashboard

```bash
python infra/dashboard.py           # English, http://localhost:7862
python infra/dashboard.py --lang ru  # Russian
```

Five tabs: Datasets (Drive sync status), Training (TensorBoard loss curves), Checkpoints (list of .pt files), Infrastructure (live Vast.ai GPU/cost/uptime via REST API + Google Drive reachability), Agent Team Status (live agent states, task summary, recent messages from SQLite DB).
Vast.ai API key read from `VAST_API_KEY` env var or `~/.config/vastai/vast_api_key`.

## Agent team

This project uses a 12-person agent team defined in `.claude/agents/`.
**Always delegate work to the appropriate agent(s) rather than doing it directly.**

### Team roster
| Agent | Name | Domain |
|-------|------|--------|
| `system-engineer` | Jason | System oversight, final arbiter on conflicts |
| `ai-team-lead` | Lara | AI/ML strategy, research, planning |
| `ai-engineer-training` | Adam | Training pipeline, losses, metrics |
| `ai-engineer-gans` | Kyle | Generator, discriminator architecture |
| `data-engineer` | Cain | Datasets, augmentation, data pipelines |
| `frontend-server` | Rona | Server-side dashboards, monitoring UI |
| `frontend-analyzer` | Pierce | Analyzer Gradio UI, i18n |
| `backend-analyzer` | Anton | Audio metrics, music analysis |
| `devops-engineer` | Marina | Vast.ai, cloud, MLOps |
| `docs-manager` | Perla | Documentation |
| `git-expert` | Florence | Code review, git management |
| `qa-expert` | Jack | Testing, QA, bug reporting |

### Orchestration rules
1. **Route every task** to the agent whose domain matches. If a task spans domains, spawn multiple agents in parallel.
2. **Jason (system-engineer) has final say** on cross-team conflicts and system-level decisions.
3. **Lara (ai-team-lead) has final say** on AI/ML conflicts between Adam, Kyle, and Cain.
4. **Jack (qa-expert) can flag bugs to anyone** — all team members are responsible for fixing QA-reported bugs.
5. **Florence (git-expert) reviews all code** before merges to main.
6. When uncertain which agent to use, ask Jason to triage.
7. Agents should read CLAUDE.md and relevant code before making decisions.
8. Agents can talk to each other.
9. All agents can commit and push to git develop branch.
10. **If an agent hits its maxTurns limit and stops responding**, the team lead must immediately spawn a fresh agent of the same type (with a `-2` suffix, e.g., `rona-2`) to continue the unfinished work. Pass full context of what was done and what remains in the new agent's prompt.
11. Agents are tracked via the agents manager MCP (under .claude/mcp/)

## Commands

- ALWAYS use `.venv` — never install packages globally or with --user
- ALWAYS work on `develop` branch, not `main`
- Python 3.12, PyTorch 2.11, CUDA 12.8/13.0
- No magic numbers — all constants in models/constants.py or configs/

## Infrastructure

- **Compute**: Vast.ai (RTX 3090 spot). SSH: `ssh -i ~/.ssh/id_ed25519 -p PORT root@sshN.vast.ai`
- **Code**: GitHub public repo sstbrg/audio-enhancer, branch: develop
- **Data**: Google Drive via rclone (remote name: `gdrive:`)
- **GCP**: project stoked-mapper-258810, GPU quota denied (can retry after 48h)
- **Monitoring**: TensorBoard via SSH tunnel, `infra/datasets.py watch` for data

## Architecture

- Generator: HiFi-GAN, ~10M params, single 2x upsample (48k→96k), skip connection
- Discriminators: MPD (periods 2,3,5,7,11,17,23) + MSD (3 scales, spectral norm on first)
- Training losses: adversarial + feature matching + multi-res STFT + mel + mastering
- Mastering losses: perceptual STFT (auraloss), stereo image, dynamics (K-weighted), encodec embedding
- CLAP + Audiobox: validation only (not differentiable)
- Gradient clipping: max_norm=5.0, LeakyReLU slope=0.1 (from constants)

## Training phases

### Phase 0: Super-Resolution (current)
- **Goal:** Upscale 48kHz audio to 96kHz/24-bit using GAN
- **Input:** High-quality audio downsampled to 48kHz
- **Target:** Original high-quality audio at 96kHz
- **Model:** HiFi-GAN generator (~10M params, 2x upsample) + MPD/MSD discriminators
- **Losses:** Adversarial + feature matching + multi-res STFT + mel + mastering (perceptual STFT, dynamics, encodec)
- **Datasets:** All available (EG-IPT, MUSDB18-HQ, VCTK, MusicNet, GTSinger, MoisesDB, MAESTRO)
- **What it learns:** Reconstruct missing high-frequency harmonics above 24kHz that were lost in downsampling

### Phase 1: Degradation Restoration (implementation in progress)
- **Goal:** Restore quality of poorly mastered / lossy-compressed audio
- **Input:** Good audio degraded at 96kHz (codec, EQ, compression, stereo damage), then downsampled to 48kHz for model input
- **Target:** Original clean 96kHz audio
- **Architecture:** Same 48kHz→96kHz HiFi-GAN generator — no new model needed. Model learns restoration AND upsampling simultaneously.
- **Fine-tuning:** From Phase 0 checkpoint (reset optimizer, lower LR: 0.0001, 5-epoch warmup)
- **Mixed batches:** 20% Phase 0 SR pairs per batch (catastrophic forgetting prevention)
- **Losses:** Same stack + `lambda_dynamics=15.0` (3× Phase 0) + `HighFrequencyBandLoss` (16–24 kHz, weight 10.0)
- **Config:** `configs/phase1.yaml`. Fine-tuning recipe: `docs/phase1_finetuning_recipe.md`
- **What it learns:** Undo codec artifacts (MP3/AAC), fix bad EQ, restore dynamics, recover stereo width — AND upscale to 96kHz

### Inference behavior (enhance.py)
- **44.1kHz input** → resample to 48kHz → GAN → 96kHz/24-bit output
- **48kHz input** → GAN → 96kHz/24-bit output
- **96kHz input** → skip super-resolution (phase 1: apply mastering enhancement only)
- **>96kHz input** → pass through unchanged

### Phase 2: Full Pipeline (planned)
- **Goal:** End-to-end enhancement: any audio in → studio-quality out
- **Pipeline:** Apollo (lossy restore) → AudioSR (bandwidth extension) → GAN (upsample + master)
- **Training:** Fine-tune on real-world audio pairs, optimize for perceptual metrics

## Next steps

1. **[BLOCKING] Fix code review criticals (Adam)** — 5 critical issues from Florence-2 review must be fixed before merging to main. See `docs/code_review_report.md`.
2. Test AMP + torch.compile training on Vast.ai (committed, not yet run in production)
3. Evaluate epoch 0 checkpoint quality with analyzer GUI
4. Continue Phase 0 training (more epochs; see `docs/phase0_training_analysis.md`)
5. MAESTRO dataset: re-download on Vast.ai (101/120GB incomplete, auto-retry script in place)
6. MoisesDB: user requested access at developer.moises.ai — download when link arrives
7. MedleyDB: user requested access at medleydb.weebly.com — download when link arrives
8. **Phase 1 implementation** (Cain + Adam + Kyle + Jack, in progress) — degradations, chain, dataset, config done; train.py --phase flag done. Next: run first Phase 1 training session.
9. Vast.ai instance may still be running — check and destroy if done
