"""Degradation chain builder and curriculum scheduler for Phase 1 training.

Chains degradations in a realistic order matching real-world signal paths:
source -> processing -> codec -> spatial.
"""

import random

import torch

from data.degradations import (
    BadEQDegradation,
    ClippingDegradation,
    CodecDegradation,
    DynamicCompressionDegradation,
    NoiseDegradation,
    SampleRateDegradation,
    StereoDamageDegradation,
)


# Degradation stages in signal-chain order
_SOURCE_DEGRADATIONS = [SampleRateDegradation, NoiseDegradation]
_PROCESSING_DEGRADATIONS = [BadEQDegradation, DynamicCompressionDegradation, ClippingDegradation]
_CODEC_DEGRADATIONS = [CodecDegradation]
_SPATIAL_DEGRADATIONS = [StereoDamageDegradation]


class DegradationChain:
    """Builds and applies a randomized degradation chain with curriculum learning."""

    def __init__(self, config: dict):
        """
        config: degradation section from phase1.yaml, containing:
            p_codec, p_eq, p_compression, p_clipping, p_sample_rate, p_noise, p_stereo
            max_chain_length
            curriculum: {enabled, warmup_epochs, linear_ramp_epochs}
        """
        self.config = config
        self.max_chain_length = config.get("max_chain_length", 3)
        self.current_severity = 0.0
        self.current_epoch = 0

        # Probability map: degradation class -> config key
        self._prob_map = {
            CodecDegradation: "p_codec",
            BadEQDegradation: "p_eq",
            DynamicCompressionDegradation: "p_compression",
            ClippingDegradation: "p_clipping",
            SampleRateDegradation: "p_sample_rate",
            NoiseDegradation: "p_noise",
            StereoDamageDegradation: "p_stereo",
        }

    def update_epoch(self, epoch: int):
        """Update severity based on curriculum schedule."""
        self.current_epoch = epoch
        curriculum = self.config.get("curriculum", {})

        if not curriculum.get("enabled", False):
            self.current_severity = 1.0
            return

        warmup = curriculum.get("warmup_epochs", 20)
        ramp = curriculum.get("linear_ramp_epochs", 80)

        if epoch < warmup:
            self.current_severity = 0.1  # Mild during warmup
        elif epoch < warmup + ramp:
            progress = (epoch - warmup) / ramp
            self.current_severity = 0.1 + 0.9 * progress
        else:
            self.current_severity = 1.0  # Full random severity

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply a random degradation chain to the waveform."""
        chain = self._build_chain(waveform.shape[0] >= 2)

        for degradation in chain:
            degradation.set_severity(self.current_severity)
            waveform = degradation(waveform, sample_rate)

        return waveform

    def _build_chain(self, is_stereo: bool) -> list:
        """Build a degradation chain respecting signal-path order."""
        chain = []

        # Stage 1: Source degradation
        for cls in _SOURCE_DEGRADATIONS:
            prob_key = self._prob_map[cls]
            if random.random() < self.config.get(prob_key, 0.0):
                chain.append(cls(self.current_severity))

        # Stage 2: Processing degradation
        for cls in _PROCESSING_DEGRADATIONS:
            prob_key = self._prob_map[cls]
            if random.random() < self.config.get(prob_key, 0.0):
                chain.append(cls(self.current_severity))

        # Stage 3: Codec (always last before spatial)
        for cls in _CODEC_DEGRADATIONS:
            prob_key = self._prob_map[cls]
            if random.random() < self.config.get(prob_key, 0.0):
                chain.append(cls(self.current_severity))

        # Stage 4: Spatial (stereo only)
        if is_stereo:
            for cls in _SPATIAL_DEGRADATIONS:
                prob_key = self._prob_map[cls]
                if random.random() < self.config.get(prob_key, 0.0):
                    chain.append(cls(self.current_severity))

        # Enforce: at least one degradation
        if not chain:
            chain.append(CodecDegradation(self.current_severity))

        # Enforce: max chain length
        if len(chain) > self.max_chain_length:
            chain = random.sample(chain, self.max_chain_length)

        return chain
