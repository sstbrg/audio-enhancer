#!/usr/bin/env python3
"""Audio Enhancement Pipeline — Upscale any audio to 192kHz/32-bit.

Multi-stage pipeline:
  Stage 1: Apollo — Restore lossy compression artifacts (MP3/AAC -> lossless quality)
  Stage 2: AudioSR — Neural bandwidth extension to 48kHz
  Stage 3: Custom GAN — Upsample 48kHz -> 192kHz with harmonic generation
  Stage 4: Output as 32-bit float WAV/FLAC

Usage:
    # Full pipeline (requires trained GAN checkpoint):
    python enhance.py input.mp3 -o output.wav --gan_checkpoint checkpoints/latest.pt

    # Without custom GAN (uses high-quality resampling for 48k->192k):
    python enhance.py input.mp3 -o output.wav

    # Skip Apollo (input is already lossless):
    python enhance.py input.flac -o output.wav --no-apollo

    # Process entire directory:
    python enhance.py /path/to/music/ -o /path/to/output/ --gan_checkpoint checkpoints/latest.pt

    # Batch with specific format:
    python enhance.py /path/to/music/ -o /path/to/output/ --format flac
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from utils.audio import load_audio, save_audio, resample_audio, get_audio_info


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class AudioEnhancer:
    """Multi-stage audio enhancement pipeline."""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        gan_checkpoint: str | None = None,
        device: str | None = None,
    ):
        self.config = load_config(config_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Device: {self.device}")

        self.apollo_model = None
        self.audiosr_model = None
        self.gan_model = None

        # Load GAN if checkpoint provided
        if gan_checkpoint:
            self._load_gan(gan_checkpoint)

    def _load_gan(self, checkpoint_path: str):
        """Load trained GAN generator."""
        from models.generator import Generator

        print(f"Loading GAN from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        gen_cfg = ckpt["config"]["gan"]["generator"]
        self.gan_model = Generator(
            channels=gen_cfg["channels"],
            upsample_rates=gen_cfg["upsample_rates"],
            upsample_kernel_sizes=gen_cfg["upsample_kernel_sizes"],
            resblock_kernel_sizes=gen_cfg["resblock_kernel_sizes"],
            resblock_dilation_sizes=gen_cfg["resblock_dilation_sizes"],
        ).to(self.device)

        self.gan_model.load_state_dict(ckpt["generator"])
        self.gan_model.eval()
        self.gan_model.remove_weight_norm()
        print("GAN loaded.")

    def _load_apollo(self):
        """Lazy-load Apollo model for lossy artifact restoration."""
        if self.apollo_model is not None:
            return

        print("Loading Apollo model...")
        try:
            from look2hear.models import apollo as apollo_module

            self.apollo_model = apollo_module.Apollo(
                "JusperLee/Apollo",
                sr=44100,
                win=20,
                feature_dim=256,
                layer=6,
            )
            self.apollo_model = self.apollo_model.to(self.device)
            self.apollo_model.eval()
            print("Apollo loaded.")
        except ImportError:
            print(
                "WARNING: Apollo not installed. Skipping lossy restoration.\n"
                "  Install: pip install look2hear  (or clone https://github.com/JusperLee/Apollo)"
            )
            self.config["pipeline"]["apollo"] = False

    def _load_audiosr(self):
        """Lazy-load AudioSR model."""
        if self.audiosr_model is not None:
            return

        print("Loading AudioSR model...")
        try:
            import audiosr

            sr_cfg = self.config["audiosr"]
            self.audiosr_model = audiosr.build_model(model_name=sr_cfg["model_name"])
            print("AudioSR loaded.")
        except ImportError:
            print(
                "WARNING: AudioSR not installed. Skipping neural SR.\n"
                "  Install: pip install audiosr"
            )
            self.config["pipeline"]["audiosr"] = False

    def enhance_file(
        self,
        input_path: str,
        output_path: str,
        use_apollo: bool | None = None,
        use_audiosr: bool | None = None,
        use_gan: bool | None = None,
    ) -> str:
        """Enhance a single audio file through the full pipeline.

        Returns the output file path.
        """
        pipeline = self.config["pipeline"]
        if use_apollo is not None:
            pipeline = {**pipeline, "apollo": use_apollo}
        if use_audiosr is not None:
            pipeline = {**pipeline, "audiosr": use_audiosr}
        if use_gan is not None:
            pipeline = {**pipeline, "gan_upsample": use_gan}

        info = get_audio_info(input_path)
        print(f"\nInput: {input_path}")
        print(f"  Sample rate: {info['sample_rate']} Hz")
        print(f"  Channels: {info['channels']}")
        print(f"  Duration: {info['duration']:.1f}s")
        print(f"  Format: {info['format']} ({info['subtype']})")

        target_sr = self.config["output"]["sample_rate"]
        bit_depth = self.config["output"]["bit_depth"]

        # Load audio
        waveform, sr = load_audio(input_path)
        original_channels = waveform.shape[0]

        t0 = time.time()

        # ---- Stage 1: Apollo (lossy artifact restoration) ----
        if pipeline["apollo"]:
            waveform, sr = self._run_apollo(waveform, sr)

        # ---- Stage 2: AudioSR (neural bandwidth extension to 48kHz) ----
        if pipeline["audiosr"] and sr < 48000:
            waveform, sr = self._run_audiosr(input_path, waveform, sr)

        # ---- Stage 3: GAN upsample (48kHz -> 192kHz) ----
        if pipeline["gan_upsample"] and self.gan_model is not None:
            waveform, sr = self._run_gan(waveform, sr)
        elif sr < target_sr:
            # Fallback: high-quality resampling
            print(f"  Resampling {sr} Hz -> {target_sr} Hz (Kaiser sinc)...")
            waveform = resample_audio(waveform, sr, target_sr)
            sr = target_sr

        elapsed = time.time() - t0
        print(f"  Processing time: {elapsed:.1f}s")

        # Save output
        save_audio(output_path, waveform, sr, bit_depth)
        out_info = get_audio_info(output_path)
        print(f"\nOutput: {output_path}")
        print(f"  Sample rate: {out_info['sample_rate']} Hz")
        print(f"  Format: {out_info['format']} ({out_info['subtype']})")
        print(f"  Duration: {out_info['duration']:.1f}s")

        return output_path

    def _run_apollo(self, waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        """Stage 1: Restore lossy compression artifacts."""
        self._load_apollo()
        if not self.config["pipeline"]["apollo"]:
            return waveform, sr

        print("  Stage 1: Apollo — restoring lossy artifacts...")

        # Apollo expects 44.1kHz
        if sr != 44100:
            waveform = resample_audio(waveform, sr, 44100)
            sr = 44100

        # Apollo expects (batch, channels, samples)
        with torch.no_grad():
            input_tensor = waveform.unsqueeze(0).to(self.device)
            # Process in chunks if very long (Apollo memory limit)
            chunk_size = 44100 * 30  # 30 seconds
            if waveform.shape[-1] > chunk_size:
                output = self._process_chunks(
                    input_tensor, self.apollo_model, chunk_size, overlap=44100
                )
            else:
                output = self.apollo_model(input_tensor)
            waveform = output.squeeze(0).cpu()

        print("  Apollo complete.")
        return waveform, sr

    def _run_audiosr(
        self, input_path: str, waveform: torch.Tensor, sr: int
    ) -> tuple[torch.Tensor, int]:
        """Stage 2: Neural super-resolution to 48kHz."""
        self._load_audiosr()
        if not self.config["pipeline"]["audiosr"]:
            return waveform, sr

        print("  Stage 2: AudioSR — neural bandwidth extension to 48kHz...")

        import audiosr

        sr_cfg = self.config["audiosr"]

        # AudioSR works on files, so save temp if needed
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            save_audio(tmp_path, waveform, sr)

        waveform_48k = audiosr.super_resolution(
            self.audiosr_model,
            tmp_path,
            seed=42,
            guidance_scale=sr_cfg["guidance_scale"],
            ddim_steps=sr_cfg["ddim_steps"],
        )

        Path(tmp_path).unlink(missing_ok=True)

        # audiosr returns numpy array (batch, channels, samples) at 48kHz
        if isinstance(waveform_48k, np.ndarray):
            waveform = torch.from_numpy(waveform_48k).squeeze(0).float()
        else:
            waveform = waveform_48k.squeeze(0).float()

        print("  AudioSR complete (48kHz).")
        return waveform, 48000

    def _run_gan(self, waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        """Stage 3: GAN upsampling to 192kHz."""
        target_sr = self.config["output"]["sample_rate"]
        print(f"  Stage 3: GAN upsample {sr} Hz -> {target_sr} Hz...")

        # Ensure input is at 48kHz for the GAN
        if sr != 48000:
            waveform = resample_audio(waveform, sr, 48000)
            sr = 48000

        channels = waveform.shape[0]
        output_channels = []

        # Process each channel separately (GAN is mono)
        for ch in range(channels):
            ch_audio = waveform[ch : ch + 1].unsqueeze(0).to(self.device)  # (1, 1, T)

            with torch.no_grad():
                # Process in chunks to manage memory
                chunk_samples = 48000 * 10  # 10 seconds at 48kHz
                if ch_audio.shape[-1] > chunk_samples:
                    output = self._gan_chunked(ch_audio, chunk_samples)
                else:
                    output = self.gan_model(ch_audio)

            output_channels.append(output.squeeze(0).cpu())

        waveform = torch.cat(output_channels, dim=0)
        waveform = torch.clamp(waveform, -1.0, 1.0)

        print(f"  GAN upsample complete ({target_sr} Hz).")
        return waveform, target_sr

    def _gan_chunked(
        self, audio: torch.Tensor, chunk_samples: int, overlap: int = 4800
    ) -> torch.Tensor:
        """Process long audio through GAN in overlapping chunks."""
        total = audio.shape[-1]
        upsample_factor = self.gan_model.upsample_factor
        stride = chunk_samples - overlap
        outputs = []
        pos = 0

        while pos < total:
            end = min(pos + chunk_samples, total)
            chunk = audio[..., pos:end]

            with torch.no_grad():
                out = self.gan_model(chunk)

            if pos == 0:
                outputs.append(out)
            else:
                # Crossfade overlap region
                overlap_hr = overlap * upsample_factor
                fade_in = torch.linspace(0, 1, overlap_hr, device=out.device)
                fade_out = 1 - fade_in

                out_start = out[..., :overlap_hr]
                prev_end = outputs[-1][..., -overlap_hr:]
                blended = prev_end * fade_out + out_start * fade_in

                outputs[-1] = outputs[-1][..., :-overlap_hr]
                outputs.append(torch.cat([blended, out[..., overlap_hr:]], dim=-1))

            pos += stride

        return torch.cat(outputs, dim=-1)

    def _process_chunks(
        self,
        audio: torch.Tensor,
        model: torch.nn.Module,
        chunk_size: int,
        overlap: int = 4410,
    ) -> torch.Tensor:
        """Generic chunked processing with crossfade."""
        total = audio.shape[-1]
        stride = chunk_size - overlap
        outputs = []
        pos = 0

        while pos < total:
            end = min(pos + chunk_size, total)
            chunk = audio[..., pos:end]

            with torch.no_grad():
                out = model(chunk)

            if pos == 0:
                outputs.append(out)
            else:
                fade_in = torch.linspace(0, 1, overlap, device=out.device)
                fade_out = 1 - fade_in

                out_start = out[..., :overlap]
                prev_end = outputs[-1][..., -overlap:]
                blended = prev_end * fade_out + out_start * fade_in

                outputs[-1] = outputs[-1][..., :-overlap]
                outputs.append(torch.cat([blended, out[..., overlap:]], dim=-1))

            pos += stride

        return torch.cat(outputs, dim=-1)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance audio to 192kHz/32-bit using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file or directory")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--gan_checkpoint", default=None,
                        help="Path to trained GAN checkpoint")
    parser.add_argument("--no-apollo", action="store_true",
                        help="Skip Apollo lossy restoration")
    parser.add_argument("--no-audiosr", action="store_true",
                        help="Skip AudioSR neural SR")
    parser.add_argument("--no-gan", action="store_true",
                        help="Skip GAN upsampling (use Kaiser resampling)")
    parser.add_argument("--format", choices=["wav", "flac"], default="wav",
                        help="Output format")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    enhancer = AudioEnhancer(
        config_path=args.config,
        gan_checkpoint=args.gan_checkpoint,
        device=args.device,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    audio_exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
                  ".aiff", ".aif", ".opus", ".alac"}

    if input_path.is_file():
        # Single file
        if output_path.suffix == "":
            output_path = output_path / (input_path.stem + f"_enhanced.{args.format}")
        enhancer.enhance_file(
            str(input_path),
            str(output_path),
            use_apollo=not args.no_apollo,
            use_audiosr=not args.no_audiosr,
            use_gan=not args.no_gan,
        )

    elif input_path.is_dir():
        # Batch process directory
        output_path.mkdir(parents=True, exist_ok=True)
        files = [f for f in input_path.rglob("*") if f.suffix.lower() in audio_exts]
        print(f"Found {len(files)} audio files to process")

        for i, f in enumerate(sorted(files)):
            print(f"\n[{i + 1}/{len(files)}] ", end="")
            out_file = output_path / f"{f.stem}_enhanced.{args.format}"
            try:
                enhancer.enhance_file(
                    str(f),
                    str(out_file),
                    use_apollo=not args.no_apollo,
                    use_audiosr=not args.no_audiosr,
                    use_gan=not args.no_gan,
                )
            except Exception as e:
                print(f"  ERROR processing {f}: {e}")
                continue

        print(f"\nDone! Enhanced {len(files)} files -> {output_path}")
    else:
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
