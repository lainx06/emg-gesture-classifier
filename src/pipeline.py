import numpy as np
from scipy import signal

# --- Pipeline Constants ---
FS_CLINICAL = 2048
FS_COMMODITY = 1000

# ADC Quantization Limits (99.9th percentile, Session 1 only, measured at the
# exact point in the chain where each quantizer runs)
V_MAX_2CH      = 0.3329    # 2 channels, post-resample  -> commodity
V_MAX_8CH_2048 = 0.6407    # 8 channels, pre-resample   -> depth_only
V_MAX_8CH_1000 = 0.6395    # 8 channels, post-resample  -> rate_depth

EPSILON = 0.02             # 20 uV = 0.02 mV deadzone for ZC
WINDOW_MS = 200
STEP_MS = 100
TRANSIENT_MS = 200


def quantize_10bit(sig, v_max):
    """
    True 10-bit bipolar map without asymmetric clipping.
    Range: [-v_max, v_max] mapped symmetrically to [0, 1023].
    """
    lsb = (2 * v_max) / 1023.0
    sig_q = np.round((sig + v_max) / lsb)
    sig_q = np.clip(sig_q, 0, 1023)
    return (sig_q * lsb) - v_max


def extract_features(window, fs):
    """
    Extracts RMS, MAV, WL, and ZC.
    Returns flattened array grouped by channel:
    [ch0_rms, ch0_mav, ch0_wl, ch0_zc, ch1_rms, ch1_mav...]
    """
    N = window.shape[0]
    features = []

    rms = np.sqrt(np.mean(window**2, axis=0))
    mav = np.mean(np.abs(window), axis=0)

    # Calculate difference array once to prevent redundant computation
    d = np.diff(window, axis=0)
    wl = np.sum(np.abs(d), axis=0) / N

    signs = np.sign(window)
    signs[signs == 0] = 1

    sc = np.diff(signs, axis=0) != 0
    ac = np.abs(d) >= EPSILON
    zc = np.sum(sc & ac, axis=0) / (N / fs)

    for ch in range(window.shape[1]):
        features.extend([rms[ch], mav[ch], wl[ch], zc[ch]])

    return np.array(features)


def get_windows(duration_ms, fs):
    """
    Calculates window start and end indices dynamically based on fs
    to ensure exact window counts across differing sample rates.
    """
    n_windows = int((duration_ms - WINDOW_MS) // STEP_MS) + 1
    indices = []
    for i in range(n_windows):
        start = round((i * STEP_MS) * fs / 1000)
        end = start + round(WINDOW_MS * fs / 1000)
        indices.append((start, end))
    return indices


def process_trial(sig, condition="clinical"):
    """
    Executes the degradation pipeline dynamically based on isolated constraint
    conditions.

    Conditions:
      clinical      8ch, 2048 Hz, float
      spatial_only  2ch, 2048 Hz, float
      rate_only     8ch, 1000 Hz, float
      depth_only    8ch, 2048 Hz, 10-bit
      rate_depth    8ch, 1000 Hz, 10-bit
      commodity     2ch, 1000 Hz, 10-bit
    """
    # 1. Spatial Constraints (Channels)
    if condition in ["commodity", "spatial_only"]:
        sig = sig[:, [0, 4]]

    # 2. V_MAX Routing for Quantization
    if condition == "commodity":
        v_max = V_MAX_2CH
    elif condition == "depth_only":
        v_max = V_MAX_8CH_2048
    elif condition == "rate_depth":
        v_max = V_MAX_8CH_1000
    else:
        v_max = None  # clinical, spatial_only, rate_only never quantize

    # 3. Rate Constraints (Resampling)
    if condition in ["commodity", "rate_only", "rate_depth"]:
        sig = signal.resample_poly(sig, up=FS_COMMODITY, down=FS_CLINICAL, axis=0)
        fs = FS_COMMODITY
    else:
        fs = FS_CLINICAL

    # 4. Depth Constraints (Quantization)
    if condition in ["commodity", "depth_only", "rate_depth"]:
        if v_max is None:
            raise ValueError(
                f"Quantization attempted on {condition} without a valid V_MAX."
            )
        sig = quantize_10bit(sig, v_max)

    # Causal, single-pass high-pass filter matching edge hardware.
    # NOT sosfiltfilt -- filtfilt is non-causal and no real-time system can do it.
    sos = signal.butter(4, 20, btype='highpass', fs=fs, output='sos')
    sig = signal.sosfilt(sos, sig, axis=0)

    # int() floor, not round(). round(200 * 2048/1000) = 410 samples = 200.195 ms,
    # which trims too much and costs the clinical arm its final window.
    transient_samples = int(TRANSIENT_MS * fs / 1000)
    sig = sig[transient_samples:, :]

    # Time-domain window definition enforces identical window counts across arms
    retained_ms = (len(sig) / fs) * 1000
    window_indices = get_windows(retained_ms, fs)

    features = []
    for start, end in window_indices:
        if end <= len(sig):
            window = sig[start:end, :]
            features.append(extract_features(window, fs))

    return np.array(features)