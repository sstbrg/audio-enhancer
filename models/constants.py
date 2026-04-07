"""Default constants for the audio enhancer model.

All sample rates, bit depths, and architecture defaults live here.
Config YAML values override these at runtime.
"""

# Audio format defaults
INPUT_SAMPLE_RATE = 48000
OUTPUT_SAMPLE_RATE = 96000
OUTPUT_BIT_DEPTH = 24

# CD-quality degradation simulation
CD_SAMPLE_RATE = 44100       # CD standard sample rate
CD_BIT_DEPTH = 16            # CD standard bit depth
CD_DITHER_AMPLITUDE = 0.5    # Triangular dither amplitude in LSBs (TPDF standard)

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
GRAD_CLIP_MAX_NORM = 5.0         # Gradient clipping max norm
LR_SCHEDULER_GAMMA = 0.999       # ExponentialLR decay factor
LOSS_SPIKE_THRESHOLD = 10.0      # Skip G step if g_loss > this * running mean
LOSS_EMA_DECAY = 0.99            # Exponential moving average decay for loss tracking
AMP_SCALER_GROWTH_INTERVAL = 4000  # Steps between AMP scaler growth attempts

# ── Upscale potential analysis ───────────────────────────────────────────────
# FFT parameters
UPSCALE_FFT_N = 4096                 # FFT size for spectral analysis
UPSCALE_SPREAD_SAMPLE_FRAMES = 20    # Evenly-spaced FFT frames sampled from file

# Bit-depth analysis cap (seconds)
UPSCALE_BIT_DEPTH_MAX_SECONDS = 10.0

# Sample-rate ceiling detection
UPSCALE_NOISE_FLOOR_RATIO = 1e-4     # -40 dB below spectrum peak
UPSCALE_ENERGY_SMOOTH_BINS = 20      # Smoothing window (bins) for mean power spectrum
UPSCALE_CEILING_TOLERANCE = 0.05     # Fraction of Nyquist — treat as full if within this

# Codec artifact detection
UPSCALE_CODEC_CUTOFF_DROP_DB = 15.0  # Min dB drop to flag as hard cutoff
UPSCALE_CODEC_CUTOFF_CANDIDATES_HZ = [11000, 15000, 16000, 18000, 19000, 20000, 22000]
UPSCALE_CODEC_CUTOFF_BAND_HZ = 500   # Band width (Hz) for energy measurement around cutoff
UPSCALE_CODEC_PASSBAND_MIN_RATIO = 1e-6  # Min pass-band energy fraction for meaningful test
UPSCALE_PRE_ECHO_WINDOW_MS = 20.0    # Pre-echo analysis frame length (ms)
UPSCALE_PRE_ECHO_RATIO_THRESHOLD = 0.15  # Frame energy vs following transient threshold
UPSCALE_PRE_ECHO_MIN_EVENTS = 2      # Min pre-echo events before flagging
UPSCALE_SBR_CORRELATION_THRESHOLD = 0.80  # Spectral Band Replication detection threshold

# Bit-depth headroom
UPSCALE_CANDIDATE_BIT_DEPTHS = [8, 16, 20, 24, 32]  # Probe depths (ascending)
UPSCALE_BIT_DEPTH_RESIDUAL_THRESHOLD = 1e-10  # Quantisation residual power threshold

# Spectral rolloff vs Nyquist gap
UPSCALE_ROLLOFF_PERCENT = 0.99       # Cumulative energy fraction for rolloff

# Composite score weights (must sum to 1.0)
UPSCALE_WEIGHT_SR_CEILING = 0.35
UPSCALE_WEIGHT_CODEC = 0.25
UPSCALE_WEIGHT_BIT_DEPTH = 0.15
UPSCALE_WEIGHT_GAP = 0.25

# Score verdict thresholds (used in backend _build_summary and frontend renderer)
UPSCALE_SCORE_HIGH = 70     # score >= HIGH  → "high potential"
UPSCALE_SCORE_MEDIUM = 40   # score >= MEDIUM → "medium potential"
UPSCALE_SCORE_LOW = 10      # score >= LOW   → "low potential" (else "already optimal")

# ── Phase 1: Degradation constants ──────────────────────────────────────────

# Codec types and bitrate ranges (kbps)
CODEC_TYPES = ["mp3", "aac", "ogg", "opus", "wma"]
CODEC_BITRATE_RANGES = {
    "mp3": [64, 96, 128, 160, 192, 256, 320],
    "aac": [64, 96, 128, 160, 192, 256],
    "ogg": [64, 96, 128, 160, 192, 256, 320],
    "opus": [32, 48, 64, 96, 128],
    "wma": [96, 128, 192],
}
CODEC_DOUBLE_ENCODE_PROB = 0.05  # Probability of double-encoding

# EQ degradation ranges
EQ_FREQ_RANGE = (60.0, 16000.0)   # Hz, log-distributed
EQ_GAIN_RANGE = (-12.0, 12.0)     # dB
EQ_Q_RANGE = (0.3, 8.0)
EQ_BANDS_RANGE = (2, 5)           # Number of parametric EQ bands
EQ_RESONANT_Q_RANGE = (6.0, 12.0)
EQ_RESONANT_GAIN_RANGE = (6.0, 15.0)  # dB

# Dynamic compression
COMP_THRESHOLD_RANGE = (-20.0, -6.0)   # dBFS
COMP_RATIO_RANGE = (4.0, 20.0)
COMP_ATTACK_RANGE = (0.1, 5.0)         # ms
COMP_RELEASE_RANGE = (50.0, 500.0)     # ms

# Clipping
CLIP_HARD_THRESHOLD_RANGE = (0.3, 0.95)
CLIP_SOFT_GAIN_RANGE = (1.5, 5.0)

# Sample rate / bit depth degradation
DEGRADED_SAMPLE_RATES = [22050, 32000, 44100]
DEGRADED_BIT_DEPTHS = [8, 16]

# Noise floor levels (dBFS)
NOISE_LEVEL_RANGE = (-40.0, -20.0)
HUM_FREQ_OPTIONS = [50.0, 60.0]  # Mains frequencies (EU/US)
HUM_HARMONICS = 5                 # Number of harmonics to include

# Stereo damage
STEREO_WIDTH_RANGE = (0.0, 0.6)       # Width factor for narrowing
STEREO_DELAY_RANGE = (0.0, 2.0)       # ms, per-channel delay
STEREO_CROSSTALK_RANGE = (0.05, 0.3)  # Fraction of channel mixed
