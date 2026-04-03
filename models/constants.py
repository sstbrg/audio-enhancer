"""Default constants for the audio enhancer model.

All sample rates, bit depths, and architecture defaults live here.
Config YAML values override these at runtime.
"""

# Audio format defaults
INPUT_SAMPLE_RATE = 48000
OUTPUT_SAMPLE_RATE = 96000
OUTPUT_BIT_DEPTH = 24

# Generator defaults
GENERATOR_CHANNELS = 512
GENERATOR_UPSAMPLE_RATES = [2]          # 48kHz → 96kHz (2x)
GENERATOR_UPSAMPLE_KERNELS = [4]
GENERATOR_RESBLOCK_KERNELS = [3, 7, 11]
GENERATOR_RESBLOCK_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

# Discriminator defaults
MPD_PERIODS = [2, 3, 5, 7, 11, 17, 23]  # Extended for 96kHz
MSD_SCALES = 3

# Training defaults
LEAKY_RELU_SLOPE = 0.1
