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

# Apollo model sample rate
APOLLO_SAMPLE_RATE = 44100

# Multi-resolution STFT loss configurations: (fft_size, hop_size, win_size)
# Multiple resolutions capture different time-frequency trade-offs.
STFT_LOSS_CONFIGS = [
    (512, 50, 240),
    (1024, 120, 600),
    (2048, 240, 1200),
    (4096, 480, 2400),
]

# GAN inference chunking
GAN_CHUNK_SECONDS = 10          # Chunk length for GAN inference (seconds)
APOLLO_CHUNK_SECONDS = 30       # Chunk length for Apollo inference (seconds)
GAN_CHUNK_OVERLAP_SAMPLES = 4800   # Overlap between GAN chunks (samples at 48kHz)
APOLLO_CHUNK_OVERLAP_SAMPLES = 4410  # Overlap between Apollo chunks (samples at 44.1kHz)

# Training defaults
LEAKY_RELU_SLOPE = 0.1
GRAD_CLIP_MAX_NORM = 5.0         # Gradient clipping max norm
LR_SCHEDULER_GAMMA = 0.999       # ExponentialLR decay factor
LOSS_SPIKE_THRESHOLD = 10.0      # Skip G step if g_loss > this * running mean
LOSS_EMA_DECAY = 0.99            # Exponential moving average decay for loss tracking
AMP_SCALER_GROWTH_INTERVAL = 4000  # Steps between AMP scaler growth attempts
LR_WARMUP_EPOCHS = 0                # Epochs of linear LR warmup (0 = disabled)
LR_WARMUP_START_FACTOR = 0.01       # Initial LR fraction during warmup (1% of base LR)

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

# ── Audio metrics (metrics/evaluate.py) ─────────────────────────────────────
# Sample rates used by external metric models
PAM_SAMPLE_RATE = 44100          # PAM (CLAP-based) expects 44.1kHz
PAM_CHUNK_SECONDS = 7            # PAM processes 7-second chunks
AUDIOBOX_SAMPLE_RATE = 16000     # Audiobox Aesthetics expects 16kHz
MUQ_SAMPLE_RATE = 24000          # MuQ-Eval expects 24kHz
MUQ_MAX_SECONDS = 10             # MuQ-Eval max input length (seconds)
VISQOL_SAMPLE_RATE = 48000       # ViSQOL operates at 48kHz

# Reference-based metric defaults
METRIC_REFERENCE_SAMPLE_RATE = 48000  # Default SR for SI-SNR, SDR comparison
METRIC_EPSILON = 1e-8                 # Numerical stability epsilon for metric computations

# Content preservation (chroma, MFCC, onset)
CONTENT_ANALYSIS_SAMPLE_RATE = 22050  # librosa default SR for content analysis
CHROMA_N_FFT = 4096                   # FFT size for chroma feature extraction
CHROMA_HOP_LENGTH = 512              # Hop length for chroma features
MFCC_N_COEFFICIENTS = 20             # Number of MFCC coefficients to extract
ONSET_TOLERANCE_MS = 50.0            # Onset matching tolerance (milliseconds)

# HF energy ratio metric (SR-specific validation)
HF_ENERGY_CROSSOVER_DIVISOR = 4      # Crossover freq = output_sr / divisor (e.g., 96k/4 = 24kHz)
HF_ENERGY_FFT_SIZE = 8192            # FFT size for HF analysis (higher = more freq resolution)
HF_ENERGY_HOP_SIZE = 2048            # Hop size for HF energy STFT

# ── Music analysis (metrics/music_analysis.py) ──────────────────────────────
# Essentia analysis sample rates
ESSENTIA_DEFAULT_SAMPLE_RATE = 16000  # Essentia default for general analysis
ESSENTIA_KEY_BPM_SAMPLE_RATE = 44100  # Essentia needs 44.1kHz for key/BPM accuracy

# CLAP classification
CLAP_SAMPLE_RATE = 48000              # LAION-CLAP expects 48kHz
CLAP_TEMPERATURE = 10.0               # Softmax temperature for sharpening CLAP logits

# MERT instrument recognition
MERT_MAX_SECONDS = 10                 # MERT max input length (seconds)
INSTRUMENT_MIN_PROBABILITY = 0.05     # Min probability to report instrument as detected

# Genre labels for CLAP zero-shot classification
CLAP_GENRE_LABELS = [
    "rock", "pop", "jazz", "classical", "electronic", "hip hop",
    "R&B", "metal", "folk", "country", "blues", "reggae",
    "ambient", "punk", "soul", "funk", "latin",
]

# Mood labels for CLAP zero-shot classification
CLAP_MOOD_LABELS = [
    "happy", "sad", "energetic", "calm", "aggressive",
    "melancholic", "uplifting", "dark", "romantic", "peaceful",
]

# Instrument labels for CLAP zero-shot instrument detection
CLAP_INSTRUMENT_LABELS = [
    "guitar", "piano", "drums", "bass", "violin", "vocals",
    "synthesizer", "trumpet", "saxophone", "flute", "cello",
    "organ", "harmonica", "accordion",
]

# Prompt templates for CLAP zero-shot classification
CLAP_GENRE_PROMPT_TEMPLATE = "this is {} music"
CLAP_MOOD_PROMPT_TEMPLATE = "music that sounds {}"
CLAP_INSTRUMENT_PROMPT_TEMPLATE = "a recording featuring {}"

# ── analyze.py display thresholds ────────────────────────────────────────────
# Quality indicator thresholds for the CLI analyzer
ANALYZE_CREST_FACTOR_GOOD_DB = 10.0   # Crest factor above this = good dynamics
ANALYZE_DYNAMIC_RANGE_GOOD_DB = 8.0   # Dynamic range above this = good dynamics
ANALYZE_FRAME_SIZE_SECONDS = 0.4      # Frame size for dynamic range analysis
ANALYZE_SILENCE_THRESHOLD = 1e-6      # RMS below this = silence (skip frame)
ANALYZE_CORRELATION_GOOD_LOW = 0.3    # L/R correlation lower bound for "good"
ANALYZE_CORRELATION_GOOD_HIGH = 0.95  # L/R correlation upper bound for "good"

# ── Phase 1: Degradation pipeline ────────────────────────────────────────────

# Codec degradation — supported types and bitrate menus
DEGRADATION_CODEC_TYPES = ["mp3", "aac", "ogg", "opus", "wma"]
DEGRADATION_CODEC_BITRATES: dict[str, list[int]] = {
    "mp3":  [64, 96, 128, 160, 192, 256, 320],
    "aac":  [64, 96, 128, 160, 192, 256],
    "ogg":  [64, 96, 128, 160, 192, 256, 320],
    "opus": [32, 48, 64, 96, 128],
    "wma":  [96, 128, 192],
}
# Probability weight for each codec type (must match DEGRADATION_CODEC_TYPES order)
DEGRADATION_CODEC_WEIGHTS = [0.40, 0.25, 0.15, 0.12, 0.08]  # mp3, aac, ogg, opus, wma
# Probability of double-encoding (simulates transcoding chain, e.g. MP3->AAC)
DEGRADATION_CODEC_DOUBLE_ENCODE_PROB = 0.05

# Bad-EQ degradation parameters
DEGRADATION_EQ_NUM_BANDS_RANGE = (2, 5)            # Number of biquad EQ bands
DEGRADATION_EQ_FREQ_RANGE_HZ = (60.0, 16000.0)    # Center/shelf frequency range
DEGRADATION_EQ_GAIN_RANGE_DB = (-12.0, 12.0)      # Per-band gain range (dB)
DEGRADATION_EQ_Q_RANGE = (0.3, 8.0)               # Biquad Q (bandwidth) range
DEGRADATION_EQ_RESONANCE_Q_RANGE = (6.0, 12.0)    # Narrow Q for resonant peaks
DEGRADATION_EQ_RESONANCE_GAIN_RANGE_DB = (6.0, 15.0)
DEGRADATION_EQ_LOWPASS_RANGE_HZ = (12000.0, 20000.0)  # Bandwidth-limiting lowpass cutoff
DEGRADATION_EQ_HIGHPASS_RANGE_HZ = (40.0, 200.0)      # Highpass cutoff range

# Dynamic compression degradation parameters
DEGRADATION_COMP_THRESHOLD_RANGE_DB = (-20.0, -6.0)        # Compressor threshold
DEGRADATION_COMP_RATIO_RANGE = (4.0, 20.0)                 # Compression ratio
DEGRADATION_COMP_ATTACK_RANGE_MS = (0.1, 100.0)            # Attack time (ms)
DEGRADATION_COMP_RELEASE_RANGE_MS = (10.0, 500.0)          # Release time (ms)
DEGRADATION_COMP_LIMITER_THRESHOLD_RANGE_DB = (-6.0, -1.0) # Brick-wall limiter threshold

# Stereo damage parameters
DEGRADATION_STEREO_WIDTH_RANGE = (0.0, 0.6)          # Width factor (0=mono, 0.6=narrow stereo)
DEGRADATION_STEREO_DELAY_RANGE_MS = (0.0, 2.0)       # Per-channel delay for comb filtering
DEGRADATION_STEREO_MS_GAIN_RANGE_DB = (-12.0, 12.0)  # Mid/side gain imbalance (dB)
DEGRADATION_STEREO_CROSSTALK_RANGE = (0.05, 0.30)    # Channel crosstalk fraction

# Clipping degradation parameters
DEGRADATION_CLIP_THRESHOLD_RANGE = (0.3, 0.95)  # Hard clip threshold (fraction of full scale)
DEGRADATION_CLIP_SOFT_GAIN_RANGE = (1.5, 5.0)   # Soft clip (tanh) gain factor
DEGRADATION_CLIP_INTERSAMPLE_UPSAMPLE = 4        # Upsample factor for intersample clipping

# Sample-rate / bit-depth degradation parameters
DEGRADATION_SR_TARGET_RATES = [22050, 32000, 44100]  # Intermediate reduced sample rates
DEGRADATION_BD_TARGET_DEPTHS = [8, 12, 16]           # Reduced bit depths to simulate
DEGRADATION_BD_DITHER_AMPLITUDE = 0.5                # TPDF dither amplitude in LSBs

# Noise degradation parameters
DEGRADATION_NOISE_WHITE_RANGE_DBFS = (-40.0, -20.0)  # White noise amplitude range
DEGRADATION_NOISE_PINK_RANGE_DBFS = (-40.0, -25.0)   # Pink noise amplitude range
DEGRADATION_NOISE_HUM_FREQS_HZ = [50.0, 60.0]        # Mains hum fundamentals (EU=50, US=60)
DEGRADATION_NOISE_HUM_HARMONICS = 5                   # Number of harmonics to generate
DEGRADATION_NOISE_HUM_RANGE_DBFS = (-40.0, -25.0)    # Hum amplitude range
DEGRADATION_NOISE_IMPULSE_RATE_RANGE = (1.0, 20.0)   # Impulse/click rate (per second)
DEGRADATION_NOISE_IMPULSE_RANGE_DBFS = (-30.0, -10.0)

# DegradationChain stage probability gates
DEGRADATION_CHAIN_P_SOURCE = 0.40      # Probability of applying source degradation (SR/BD/noise)
DEGRADATION_CHAIN_P_PROCESSING = 0.70  # Probability of applying processing (EQ/compression/clip)
DEGRADATION_CHAIN_P_CODEC = 0.60       # Probability of applying codec encoding
DEGRADATION_CHAIN_P_SPATIAL = 0.30     # Probability of applying stereo damage (stereo-only)
DEGRADATION_CHAIN_MAX_LENGTH = 3       # Maximum number of degradations per sample

# Severity curriculum schedule
DEGRADATION_CURRICULUM_WARMUP_EPOCHS = 20  # Epochs of mild severity only
DEGRADATION_CURRICULUM_RAMP_EPOCHS = 80    # Epochs for linear ramp to full severity
DEGRADATION_CURRICULUM_MILD_MAX = 0.3      # Max severity during warmup
DEGRADATION_CURRICULUM_FULL_MIN = 0.7      # Min severity after full ramp

# Phase 1 dataset — Phase 0 SR pair mixing ratio (catastrophic forgetting prevention)
DEGRADATION_PHASE0_MIX_RATIO = 0.20  # Fraction of each batch from clean Phase 0 SR pairs

# Pre-computed codec variants — common bitrates to pre-compute on disk
DEGRADATION_PRECOMPUTE_BITRATES: dict[str, list[int]] = {
    "mp3": [128, 192, 320],
    "aac": [128, 256],
    "ogg": [128],
}

# High-frequency band loss (16-24 kHz) — targets codec artifact restoration
# At 96kHz output, Nyquist=48kHz. The 16-24kHz band is where lossy codecs
# cause the most visible spectral damage (cutoffs, holes, SBR artifacts).
HF_BAND_FFT_SIZE = 4096              # ~23.4 Hz/bin at 96kHz — good resolution
HF_BAND_HOP_SIZE = 1024              # 75% overlap for smooth energy estimates
HF_BAND_WIN_SIZE = 4096              # Match FFT size for rectangular-free analysis
HF_BAND_LOW_FREQ = 16000             # Lower bound of HF band (Hz)
HF_BAND_HIGH_FREQ = 24000            # Upper bound of HF band (Hz) — input Nyquist
LAMBDA_HF_BAND_DEFAULT = 10.0        # Default weight for Phase 1 training

# Phase 1 recommended segment length (longer than Phase 0's 16384 for dynamics restoration)
# Dynamic compression with 500ms release needs segments > 0.5s to be meaningful.
DEGRADATION_SEGMENT_LENGTH = 32768  # ~0.68s at 48kHz input, ~1.37s at 96kHz
