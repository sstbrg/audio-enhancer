"""Tests for torch.compile and AMP (mixed precision) compatibility.

Verifies that generator and discriminator models work correctly with:
- torch.compile (inductor backend on CUDA, eager fallback on CPU)
- torch.autocast (float16 / bfloat16 mixed precision)
- Combined compiled + AMP forward passes (mimics training loop)

These tests directly validate the CLAUDE.md next step:
"Test AMP + torch.compile training (committed, not yet run)"
"""

import math

import pytest
import torch

from models.constants import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    MPD_PERIODS,
    MSD_SCALES,
)
from models.generator import Generator
from models.discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)


UPSAMPLE_FACTOR = OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE  # 2
HAS_CUDA = torch.cuda.is_available()

# Use inductor on CUDA, eager on CPU (inductor needs triton/CUDA)
COMPILE_BACKEND = "inductor" if HAS_CUDA else "eager"
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
# float16 on CUDA, bfloat16 on CPU (float16 requires CUDA for autocast)
AMP_DTYPE = torch.float16 if HAS_CUDA else torch.bfloat16


def make_audio(batch: int = 1, samples: int = 16384, device: torch.device = DEVICE) -> torch.Tensor:
    """Create a sine wave tensor on the target device."""
    t = torch.linspace(0, samples / INPUT_SAMPLE_RATE, samples, device=device)
    wave = torch.sin(2 * math.pi * 440.0 * t)
    return wave.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1).clone()


def make_hr_audio(batch: int = 1, samples: int = 32768, device: torch.device = DEVICE) -> torch.Tensor:
    """Create high-rate audio tensor (96kHz domain) on the target device."""
    t = torch.linspace(0, samples / OUTPUT_SAMPLE_RATE, samples, device=device)
    wave = torch.sin(2 * math.pi * 440.0 * t)
    return wave.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1).clone()


# ── Generator + torch.compile ──────────────────────────────────────────────────


class TestGeneratorCompile:
    """Verify Generator works with torch.compile."""

    def test_compile_succeeds(self):
        """torch.compile should not raise on the Generator."""
        gen = Generator().to(DEVICE)
        gen.eval()
        compiled = torch.compile(gen, backend=COMPILE_BACKEND)
        assert compiled is not None

    def test_compiled_forward_shape(self):
        """Compiled generator should produce correct output shape."""
        gen = Generator().to(DEVICE)
        gen.eval()
        compiled = torch.compile(gen, backend=COMPILE_BACKEND)
        x = make_audio(batch=2, samples=8192)
        with torch.no_grad():
            y = compiled(x)
        assert y.shape == (2, 1, 8192 * UPSAMPLE_FACTOR)

    def test_compiled_output_finite(self):
        """Compiled generator should produce finite outputs (no NaN/Inf)."""
        gen = Generator().to(DEVICE)
        gen.eval()
        compiled = torch.compile(gen, backend=COMPILE_BACKEND)
        x = make_audio(batch=1, samples=16384)
        with torch.no_grad():
            y = compiled(x)
        assert torch.isfinite(y).all(), "Compiled generator produced NaN/Inf"

    def test_compiled_matches_eager(self):
        """Compiled and eager forward passes should produce identical results."""
        gen = Generator().to(DEVICE)
        gen.eval()

        x = make_audio(batch=1, samples=4096)
        with torch.no_grad():
            y_eager = gen(x)

        compiled = torch.compile(gen, backend=COMPILE_BACKEND)
        with torch.no_grad():
            y_compiled = compiled(x)

        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-4, atol=1e-4)


# ── Generator + AMP ────────────────────────────────────────────────────────────


class TestGeneratorAMP:
    """Verify Generator works with automatic mixed precision."""

    def test_autocast_forward(self):
        """Generator forward under autocast should produce finite output."""
        gen = Generator().to(DEVICE)
        gen.eval()
        x = make_audio(batch=1, samples=16384)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            y = gen(x)
        assert torch.isfinite(y.float()).all(), "AMP generator produced NaN/Inf"
        assert y.shape[-1] == 16384 * UPSAMPLE_FACTOR

    def test_autocast_output_shape(self):
        """AMP should not change output shape."""
        gen = Generator().to(DEVICE)
        gen.eval()
        x = make_audio(batch=2, samples=8192)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            y = gen(x)
        assert y.shape == (2, 1, 8192 * UPSAMPLE_FACTOR)

    @pytest.mark.skipif(not HAS_CUDA, reason="GradScaler requires CUDA")
    def test_amp_backward_with_scaler(self):
        """Generator backward pass with AMP GradScaler should work (mimics training)."""
        gen = Generator().to(DEVICE)
        gen.train()
        scaler = torch.amp.GradScaler("cuda")
        x = make_audio(batch=1, samples=4096)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = gen(x)
            loss = y.mean()

        scaler.scale(loss).backward()
        # Check gradients exist and are finite
        for name, param in gen.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"


# ── Generator + compile + AMP combined ─────────────────────────────────────────


class TestGeneratorCompileAMP:
    """Verify Generator works with torch.compile AND AMP together."""

    def test_compiled_amp_forward(self):
        """Compiled generator under autocast should produce finite output."""
        gen = Generator().to(DEVICE)
        gen.eval()
        compiled = torch.compile(gen, backend=COMPILE_BACKEND)
        x = make_audio(batch=1, samples=8192)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            y = compiled(x)
        assert torch.isfinite(y.float()).all(), "Compiled+AMP generator produced NaN/Inf"
        assert y.shape[-1] == 8192 * UPSAMPLE_FACTOR

    @pytest.mark.skipif(not HAS_CUDA, reason="GradScaler requires CUDA")
    def test_compiled_amp_backward(self):
        """Compiled generator backward with AMP scaler (full training scenario)."""
        gen = Generator().to(DEVICE)
        gen.train()
        compiled = torch.compile(gen, backend=COMPILE_BACKEND)
        scaler = torch.amp.GradScaler("cuda")
        x = make_audio(batch=1, samples=4096)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = compiled(x)
            loss = y.mean()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(gen.parameters(), lr=1e-4))
        scaler.update()
        # No crash = success; verify params are still finite
        for name, param in gen.named_parameters():
            assert torch.isfinite(param).all(), f"NaN/Inf in param {name} after step"


# ── Discriminator + torch.compile ──────────────────────────────────────────────


class TestDiscriminatorCompile:
    """Verify MPD and MSD work with torch.compile."""

    def test_mpd_compile_succeeds(self):
        """torch.compile should not raise on MultiPeriodDiscriminator."""
        mpd = MultiPeriodDiscriminator().to(DEVICE)
        mpd.eval()
        compiled = torch.compile(mpd, backend=COMPILE_BACKEND)
        assert compiled is not None

    def test_msd_compile_succeeds(self):
        """torch.compile should not raise on MultiScaleDiscriminator."""
        msd = MultiScaleDiscriminator().to(DEVICE)
        msd.eval()
        compiled = torch.compile(msd, backend=COMPILE_BACKEND)
        assert compiled is not None

    def test_compiled_mpd_forward(self):
        """Compiled MPD should produce correct number of outputs."""
        mpd = MultiPeriodDiscriminator().to(DEVICE)
        mpd.eval()
        compiled = torch.compile(mpd, backend=COMPILE_BACKEND)
        real = make_hr_audio(batch=1, samples=32768)
        fake = make_hr_audio(batch=1, samples=32768)
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = compiled(real, fake)
        assert len(r_out) == len(MPD_PERIODS)
        assert all(torch.isfinite(o).all() for o in r_out)

    def test_compiled_msd_forward(self):
        """Compiled MSD should produce correct number of outputs."""
        msd = MultiScaleDiscriminator().to(DEVICE)
        msd.eval()
        compiled = torch.compile(msd, backend=COMPILE_BACKEND)
        real = make_hr_audio(batch=1, samples=32768)
        fake = make_hr_audio(batch=1, samples=32768)
        with torch.no_grad():
            r_out, f_out, r_fmap, f_fmap = compiled(real, fake)
        assert len(r_out) == MSD_SCALES
        assert all(torch.isfinite(o).all() for o in r_out)


# ── Discriminator + AMP ────────────────────────────────────────────────────────


class TestDiscriminatorAMP:
    """Verify MPD and MSD work with automatic mixed precision."""

    def test_mpd_autocast_forward(self):
        """MPD under autocast should produce finite outputs."""
        mpd = MultiPeriodDiscriminator().to(DEVICE)
        mpd.eval()
        real = make_hr_audio(batch=1, samples=32768)
        fake = make_hr_audio(batch=1, samples=32768)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            r_out, f_out, _, _ = mpd(real, fake)
        for o in r_out + f_out:
            assert torch.isfinite(o.float()).all(), "AMP MPD produced NaN/Inf"

    def test_msd_autocast_forward(self):
        """MSD under autocast should produce finite outputs."""
        msd = MultiScaleDiscriminator().to(DEVICE)
        msd.eval()
        real = make_hr_audio(batch=1, samples=32768)
        fake = make_hr_audio(batch=1, samples=32768)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            r_out, f_out, _, _ = msd(real, fake)
        for o in r_out + f_out:
            assert torch.isfinite(o.float()).all(), "AMP MSD produced NaN/Inf"


# ── End-to-end: compiled generator → discriminator (training loop simulation) ──


class TestEndToEndCompileAMP:
    """Simulate a training step: compiled generator feeds discriminator under AMP."""

    def test_gen_to_mpd_pipeline(self):
        """Generator output → MPD forward should work with compile + AMP."""
        gen = Generator().to(DEVICE)
        gen.eval()
        compiled_gen = torch.compile(gen, backend=COMPILE_BACKEND)

        mpd = MultiPeriodDiscriminator().to(DEVICE)
        mpd.eval()

        lr_input = make_audio(batch=1, samples=8192)
        hr_real = make_hr_audio(batch=1, samples=8192 * UPSAMPLE_FACTOR)

        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            hr_fake = compiled_gen(lr_input)
            r_out, f_out, r_fmap, f_fmap = mpd(hr_real, hr_fake)

        assert len(r_out) == len(MPD_PERIODS)
        assert len(f_out) == len(MPD_PERIODS)
        for o in r_out + f_out:
            assert torch.isfinite(o.float()).all()

    def test_gen_to_msd_pipeline(self):
        """Generator output → MSD forward should work with compile + AMP."""
        gen = Generator().to(DEVICE)
        gen.eval()
        compiled_gen = torch.compile(gen, backend=COMPILE_BACKEND)

        msd = MultiScaleDiscriminator().to(DEVICE)
        msd.eval()

        lr_input = make_audio(batch=1, samples=8192)
        hr_real = make_hr_audio(batch=1, samples=8192 * UPSAMPLE_FACTOR)

        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            hr_fake = compiled_gen(lr_input)
            r_out, f_out, r_fmap, f_fmap = msd(hr_real, hr_fake)

        assert len(r_out) == MSD_SCALES
        assert len(f_out) == MSD_SCALES
        for o in r_out + f_out:
            assert torch.isfinite(o.float()).all()

    @pytest.mark.skipif(not HAS_CUDA, reason="Full training step requires CUDA")
    def test_full_training_step(self):
        """Simulate one complete training step: G forward → D forward → G backward.

        This is the critical test — it mimics exactly what train.py does with
        torch.compile + AMP + GradScaler.
        """
        gen = Generator().to(DEVICE)
        gen.train()
        compiled_gen = torch.compile(gen, backend="inductor")
        optim_g = torch.optim.AdamW(gen.parameters(), lr=1e-4)
        scaler_g = torch.amp.GradScaler("cuda")

        mpd = MultiPeriodDiscriminator().to(DEVICE)
        mpd.eval()

        lr_input = make_audio(batch=1, samples=4096)
        hr_real = make_hr_audio(batch=1, samples=4096 * UPSAMPLE_FACTOR)

        # Generator forward + discriminator forward + backward
        optim_g.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hr_fake = compiled_gen(lr_input)
            _, f_out, _, f_fmap = mpd(hr_real, hr_fake)

            # Simple adversarial loss (generator wants discriminator to say "real")
            g_loss = sum(torch.mean((o - 1) ** 2) for o in f_out)

        scaler_g.scale(g_loss).backward()
        scaler_g.step(optim_g)
        scaler_g.update()

        # Verify model is still healthy after the step
        for name, param in gen.named_parameters():
            assert torch.isfinite(param).all(), f"NaN/Inf in {name} after training step"
