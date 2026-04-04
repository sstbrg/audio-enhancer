#!/usr/bin/env python3
"""Training script for the 48kHz -> 96kHz GAN audio upsampler.

Usage:
    python train.py --data_dir /path/to/highres/audio
    python train.py --data_dir /path/to/highres/audio --config configs/default.yaml --resume checkpoints/latest.pt

Expects high-quality audio files (96kHz+ preferred).
The training process:
1. Loads high-res audio
2. Downsamples to 48kHz as input
3. Trains generator to reconstruct the 96kHz version
4. Discriminators enforce realistic waveform generation
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import AudioSRDataset
from models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
    MelSpectrogramLoss,
    MultiResolutionSTFTLoss,
)
from models.mastering_losses import MasteringLoss


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def _run_validation(generator, loader, device, writer, epoch, global_step, output_sr):
    """Run full validation metrics and log to TensorBoard."""
    import tempfile
    import numpy as np
    import soundfile as sf

    generator.eval()

    si_snrs, sdrs = [], []
    cdpam_scores, audiobox_pqs = [], []
    chroma_sims, mfcc_sims = [], []
    # SR-specific metrics
    hf_energy_ratios_ref, hf_energy_ratios_enh = [], []
    spectral_rolloffs_ref, spectral_rolloffs_enh = [], []
    val_samples = 4

    for i, (lr_audio, hr_audio) in enumerate(loader):
        if i >= val_samples:
            break
        lr_audio = lr_audio.to(device)
        hr_audio = hr_audio.to(device)
        hr_hat = generator(lr_audio)

        min_len = min(hr_hat.shape[-1], hr_audio.shape[-1])
        hr_hat = hr_hat[..., :min_len]
        hr_audio = hr_audio[..., :min_len]

        ref = hr_audio[0].squeeze().cpu()
        enh = hr_hat[0].squeeze().cpu()

        # SI-SNR
        ref_z = ref - ref.mean()
        enh_z = enh - enh.mean()
        dot = torch.dot(enh_z, ref_z)
        s_target = dot * ref_z / (torch.dot(ref_z, ref_z) + 1e-8)
        e_noise = enh_z - s_target
        si_snr_val = 10 * torch.log10(
            torch.dot(s_target, s_target) / (torch.dot(e_noise, e_noise) + 1e-8) + 1e-8
        )
        si_snrs.append(si_snr_val.item())

        # SDR
        noise = ref - enh
        sdr_val = 10 * torch.log10(
            torch.dot(ref, ref) / (torch.dot(noise, noise) + 1e-8) + 1e-8
        )
        sdrs.append(sdr_val.item())

        # SR-specific metrics (computed on tensors, no file I/O)
        try:
            import librosa
            ref_np = ref.numpy()
            enh_np = enh.numpy()

            # High-frequency energy ratio (energy above input_nyquist / total)
            input_nyquist = output_sr // 4  # 48kHz input → 24kHz nyquist
            S_ref = np.abs(librosa.stft(ref_np, n_fft=4096))
            S_enh = np.abs(librosa.stft(enh_np, n_fft=4096))
            freqs = librosa.fft_frequencies(sr=output_sr, n_fft=4096)
            hf_mask = freqs >= input_nyquist

            hf_ref = (S_ref[hf_mask] ** 2).sum() / (S_ref ** 2).sum() + 1e-10
            hf_enh = (S_enh[hf_mask] ** 2).sum() / (S_enh ** 2).sum() + 1e-10
            hf_energy_ratios_ref.append(float(hf_ref))
            hf_energy_ratios_enh.append(float(hf_enh))

            # Spectral rolloff (95%)
            rolloff_ref = librosa.feature.spectral_rolloff(y=ref_np, sr=output_sr, roll_percent=0.95)[0].mean()
            rolloff_enh = librosa.feature.spectral_rolloff(y=enh_np, sr=output_sr, roll_percent=0.95)[0].mean()
            spectral_rolloffs_ref.append(float(rolloff_ref))
            spectral_rolloffs_enh.append(float(rolloff_enh))
        except Exception:
            pass

        # File-based metrics (save to temp files)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_ref, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_enh:
                sf.write(f_ref.name, ref.numpy(), output_sr)
                sf.write(f_enh.name, enh.numpy(), output_sr)

                from metrics.evaluate import AudioMetrics
                metrics = AudioMetrics(device=str(device))

                # CDPAM
                try:
                    cdpam = metrics.cdpam_score(f_ref.name, f_enh.name)
                    if not np.isnan(cdpam.score):
                        cdpam_scores.append(cdpam.score)
                except Exception:
                    pass

                # Audiobox PQ (no-reference on enhanced)
                try:
                    ab = metrics.audiobox_aesthetics(f_enh.name)
                    if isinstance(ab.score, dict) and "PQ" in ab.score:
                        audiobox_pqs.append(ab.score["PQ"])
                except Exception:
                    pass

                # Content preservation
                try:
                    chroma = metrics.chroma_similarity(f_ref.name, f_enh.name)
                    mfcc = metrics.mfcc_similarity(f_ref.name, f_enh.name)
                    chroma_sims.append(chroma.score)
                    mfcc_sims.append(mfcc.score)
                except Exception:
                    pass

                import os
                os.unlink(f_ref.name)
                os.unlink(f_enh.name)
        except Exception:
            pass

    # Log all metrics
    print(f"\n  Validation (epoch {epoch}):")
    if si_snrs:
        avg_sisnr = np.mean(si_snrs)
        writer.add_scalar("val/si_snr", avg_sisnr, global_step)
        print(f"    SI-SNR:  {avg_sisnr:.2f} dB")
    if sdrs:
        avg_sdr = np.mean(sdrs)
        writer.add_scalar("val/sdr", avg_sdr, global_step)
        print(f"    SDR:     {avg_sdr:.2f} dB")
    if cdpam_scores:
        avg_cdpam = np.mean(cdpam_scores)
        writer.add_scalar("val/cdpam", avg_cdpam, global_step)
        print(f"    CDPAM:   {avg_cdpam:.4f} (lower=better)")
    if audiobox_pqs:
        avg_pq = np.mean(audiobox_pqs)
        writer.add_scalar("val/audiobox_pq", avg_pq, global_step)
        print(f"    Audiobox PQ: {avg_pq:.2f}/10")
    if chroma_sims:
        avg_chroma = np.mean(chroma_sims)
        avg_mfcc = np.mean(mfcc_sims)
        writer.add_scalar("val/chroma_similarity", avg_chroma, global_step)
        writer.add_scalar("val/mfcc_similarity", avg_mfcc, global_step)
        print(f"    Chroma:  {avg_chroma:.4f}")
        print(f"    MFCC:    {avg_mfcc:.4f}")
    if hf_energy_ratios_enh:
        avg_hf_ref = np.mean(hf_energy_ratios_ref)
        avg_hf_enh = np.mean(hf_energy_ratios_enh)
        avg_ro_ref = np.mean(spectral_rolloffs_ref)
        avg_ro_enh = np.mean(spectral_rolloffs_enh)
        writer.add_scalar("val/sr_hf_energy_ref", avg_hf_ref, global_step)
        writer.add_scalar("val/sr_hf_energy_enh", avg_hf_enh, global_step)
        writer.add_scalar("val/sr_hf_energy_ratio", avg_hf_enh / (avg_hf_ref + 1e-10), global_step)
        writer.add_scalar("val/sr_rolloff_ref_hz", avg_ro_ref, global_step)
        writer.add_scalar("val/sr_rolloff_enh_hz", avg_ro_enh, global_step)
        print(f"    HF energy: ref={avg_hf_ref:.6f}, enh={avg_hf_enh:.6f} (ratio: {avg_hf_enh / (avg_hf_ref + 1e-10):.2f}x)")
        print(f"    Rolloff:   ref={avg_ro_ref:.0f} Hz, enh={avg_ro_enh:.0f} Hz")

    generator.train()


def train(args):
    config = load_config(args.config)
    gan_cfg = config["gan"]
    train_cfg = gan_cfg["training"]
    output_sr = config["output"]["sample_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Performance flags
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
    use_amp = device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Using device: {device} (AMP: {use_amp}, cudnn.benchmark: {torch.backends.cudnn.benchmark})")
    print(f"Target: 48kHz → {output_sr // 1000}kHz")

    # Dataset
    dataset = AudioSRDataset(
        root_dir=args.data_dir,
        target_sr=output_sr,
        input_sr=48000,
        segment_length=train_cfg["segment_length"],
    )

    # DataLoader: use WeightedRandomSampler when quality_sampling is configured,
    # otherwise fall back to plain shuffle so the interface stays backward-compatible.
    qs_cfg = train_cfg.get("quality_sampling")
    if qs_cfg:
        sample_weights = dataset.get_sample_weights(
            weight_hires=qs_cfg["weight_hires"],
            weight_midres=qs_cfg["weight_midres"],
            weight_standard=qs_cfg["weight_standard"],
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        print(
            f"Quality sampling enabled: hires={qs_cfg['weight_hires']}, "
            f"midres={qs_cfg['weight_midres']}, standard={qs_cfg['weight_standard']}"
        )
        loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        loader_kwargs = dict(shuffle=True)

    num_workers = min(os.cpu_count() or 4, 8)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
        **loader_kwargs,
    )
    print(f"DataLoader: batch_size={train_cfg['batch_size']}, workers={num_workers}")

    # Models
    gen_cfg = gan_cfg["generator"]
    generator = Generator(
        channels=gen_cfg["channels"],
        upsample_rates=gen_cfg["upsample_rates"],
        upsample_kernel_sizes=gen_cfg["upsample_kernel_sizes"],
        resblock_kernel_sizes=gen_cfg["resblock_kernel_sizes"],
        resblock_dilation_sizes=gen_cfg["resblock_dilation_sizes"],
    ).to(device)

    mpd = MultiPeriodDiscriminator(
        periods=gan_cfg["discriminator"]["periods"]
    ).to(device)

    msd = MultiScaleDiscriminator(
        num_scales=gan_cfg["discriminator"]["scales"]
    ).to(device)

    # Losses
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    mel_loss_fn = MelSpectrogramLoss(sample_rate=output_sr).to(device)

    # Mastering quality losses
    mastering_cfg = train_cfg.get("mastering", {})
    mastering_loss_fn = MasteringLoss(
        sample_rate=output_sr,
        device=str(device),
        lambda_perceptual_stft=mastering_cfg.get("lambda_perceptual_stft", 45.0),
        lambda_stereo=mastering_cfg.get("lambda_stereo", 10.0),
        lambda_dynamics=mastering_cfg.get("lambda_dynamics", 5.0),
        lambda_encodec=mastering_cfg.get("lambda_encodec", 10.0),
        lambda_audiobox_pq=mastering_cfg.get("lambda_audiobox_pq", 0.0),
        lambda_clap=mastering_cfg.get("lambda_clap", 0.0),
    )

    # Optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        lr=train_cfg["learning_rate_g"],
        betas=tuple(train_cfg["betas"]),
    )
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=train_cfg["learning_rate_d"],
        betas=tuple(train_cfg["betas"]),
    )

    # LR schedulers
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999)

    # Resume from checkpoint — must happen BEFORE torch.compile to avoid
    # _orig_mod.* key prefix mismatches in state_dict loading.
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"], strict=False)
        mpd.load_state_dict(ckpt["mpd"])
        msd.load_state_dict(ckpt["msd"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        # Restore scheduler state if available (backward-compatible)
        if "sched_g" in ckpt:
            sched_g.load_state_dict(ckpt["sched_g"])
        if "sched_d" in ckpt:
            sched_d.load_state_dict(ckpt["sched_d"])
        # Restore AMP scaler state if available (backward-compatible)
        if "scaler_g" in ckpt:
            scaler_g.load_state_dict(ckpt["scaler_g"])
        if "scaler_d" in ckpt:
            scaler_d.load_state_dict(ckpt["scaler_d"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Compile models for faster execution (PyTorch 2.x).
    # Must be done AFTER checkpoint load — torch.compile wraps parameters
    # under _orig_mod.* keys which would break state_dict compatibility.
    if hasattr(torch, "compile"):
        try:
            generator = torch.compile(generator)
            mpd = torch.compile(mpd)
            msd = torch.compile(msd)
            print("Models compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not available: {e}")

    # Logging
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "logs"))

    global_step = start_epoch * len(loader)
    train_start = time.time()
    max_seconds = args.max_hours * 3600 if args.max_hours else float("inf")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        # Time limit check
        elapsed = time.time() - train_start
        if elapsed >= max_seconds:
            print(f"\nTime limit reached ({args.max_hours}h). Stopping.")
            break

        generator.train()
        mpd.train()
        msd.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for lr_audio, hr_audio in pbar:
            lr_audio = lr_audio.to(device, non_blocking=True)
            hr_audio = hr_audio.to(device, non_blocking=True)

            # Generate high-res audio
            with torch.amp.autocast("cuda", enabled=use_amp):
                hr_hat = generator(lr_audio)

                # Align lengths
                min_len = min(hr_hat.shape[-1], hr_audio.shape[-1])
                hr_hat = hr_hat[..., :min_len]
                hr_audio = hr_audio[..., :min_len]

            # ---- Train Discriminator ----
            optim_d.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                mpd_real, mpd_fake, _, _ = mpd(hr_audio, hr_hat.detach())
                loss_mpd = discriminator_loss(mpd_real, mpd_fake)

                msd_real, msd_fake, _, _ = msd(hr_audio, hr_hat.detach())
                loss_msd = discriminator_loss(msd_real, msd_fake)

                loss_d = loss_mpd + loss_msd

            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(optim_d)
            torch.nn.utils.clip_grad_norm_(
                list(mpd.parameters()) + list(msd.parameters()), max_norm=10.0
            )
            scaler_d.step(optim_d)
            scaler_d.update()

            # ---- Train Generator ----
            optim_g.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Re-run discriminators on updated generator output
                mpd_real, mpd_fake, mpd_real_fm, mpd_fake_fm = mpd(hr_audio, hr_hat)
                msd_real, msd_fake, msd_real_fm, msd_fake_fm = msd(hr_audio, hr_hat)

                # Adversarial losses
                loss_g_mpd = generator_loss(mpd_fake)
                loss_g_msd = generator_loss(msd_fake)

                # Feature matching
                loss_fm_mpd = feature_loss(mpd_real_fm, mpd_fake_fm)
                loss_fm_msd = feature_loss(msd_real_fm, msd_fake_fm)

                # Spectral losses
                loss_stft = stft_loss_fn(hr_hat, hr_audio)
                loss_mel = mel_loss_fn(hr_hat, hr_audio)

                # Mastering quality losses
                loss_mastering, mastering_details = mastering_loss_fn(hr_hat, hr_audio)

                loss_g = (
                    loss_g_mpd + loss_g_msd
                    + train_cfg["lambda_fm"] * (loss_fm_mpd + loss_fm_msd)
                    + train_cfg["lambda_stft"] * loss_stft
                    + train_cfg["lambda_mel"] * loss_mel
                    + loss_mastering
                )

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
            scaler_g.step(optim_g)
            scaler_g.update()

            # Logging
            global_step += 1
            if global_step % train_cfg["log_interval"] == 0:
                log_dict = {
                    "loss/generator": loss_g.item(),
                    "loss/discriminator": loss_d.item(),
                    "loss/stft": loss_stft.item(),
                    "loss/mel": loss_mel.item(),
                    "loss/fm": (loss_fm_mpd + loss_fm_msd).item(),
                    "loss/mastering_total": mastering_details.get("mastering_total", 0),
                }
                for k, v in mastering_details.items():
                    if k != "mastering_total":
                        log_dict[f"loss/mastering_{k}"] = v

                # TensorBoard
                for k, v in log_dict.items():
                    writer.add_scalar(k, v, global_step)

            pbar.set_postfix(
                g=f"{loss_g.item():.3f}",
                d=f"{loss_d.item():.3f}",
            )

        sched_g.step()
        sched_d.step()

        # Periodic validation with quality metrics
        if (epoch + 1) % train_cfg["checkpoint_interval"] == 0:
            _run_validation(generator, loader, device, writer, epoch, global_step, output_sr)

        # Save checkpoint
        if (epoch + 1) % train_cfg["checkpoint_interval"] == 0 or epoch == train_cfg["epochs"] - 1:
            ckpt_path = ckpt_dir / f"checkpoint_{epoch:04d}.pt"
            ckpt_data = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "mpd": mpd.state_dict(),
                "msd": msd.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "sched_g": sched_g.state_dict(),
                "sched_d": sched_d.state_dict(),
                "scaler_g": scaler_g.state_dict(),
                "scaler_d": scaler_d.state_dict(),
                "config": config,
            }
            torch.save(ckpt_data, ckpt_path)

            # Also save as latest
            torch.save(ckpt_data, ckpt_dir / "latest.pt")

            print(f"\nSaved checkpoint: {ckpt_path}")

    # Save final checkpoint
    elapsed = time.time() - train_start
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "mpd": mpd.state_dict(),
        "msd": msd.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "sched_g": sched_g.state_dict(),
        "sched_d": sched_d.state_dict(),
        "scaler_g": scaler_g.state_dict(),
        "scaler_d": scaler_d.state_dict(),
        "config": config,
    }, ckpt_dir / "latest.pt")
    print(f"\nSaved final checkpoint (epoch {epoch})")

    writer.close()
    print(f"Training complete! Total time: {elapsed / 3600:.1f}h, {epoch + 1} epochs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio super-resolution GAN")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing high-quality audio files (96kHz+ preferred)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop training after this many hours")
    args = parser.parse_args()
    train(args)
