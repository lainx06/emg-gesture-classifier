import numpy as np
from scipy import signal

# --- Pipeline Constants ---
FS_CLINICAL = 2048
FS_COMMODITY = 1000
V_MAX = 0.3329        # mV limits for the 10-bit ADC
EPSILON = 0.02        # 20 µV = 0.02 mV deadzone for ZC
WINDOW_MS = 200
STEP_MS = 100
TRANSIENT_MS = 200

def quantize_10bit(sig):
    """
    True 10-bit bipolar map without asymmetric clipping.
    Range: [-V_MAX, V_MAX] mapped symmetrically to [0, 1023].
    """
    lsb = (2 * V_MAX) / 1023.0
    sig_q = np.round((sig + V_MAX) / lsb)
    sig_q = np.clip(sig_q, 0, 1023)
    return (sig_q * lsb) - V_MAX

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
    Executes the degradation pipeline: resample -> quantize -> filter -> extract
    """
    if condition == "commodity":
        sig = signal.resample_poly(sig, up=FS_COMMODITY, down=FS_CLINICAL, axis=0)
        fs = FS_COMMODITY
        sig = quantize_10bit(sig)
    else:
        fs = FS_CLINICAL

    # Causal, single-pass high-pass filter matching edge hardware
    sos = signal.butter(4, 20, btype='highpass', fs=fs, output='sos')
    sig = signal.sosfilt(sos, sig, axis=0)
    
    # Use int() to floor the trim, ensuring enough tail data for the final window
    transient_samples = int(TRANSIENT_MS * fs / 1000)
    sig = sig[transient_samples:, :]
    
    # Calculate exact retained duration in ms to enforce window symmetry
    retained_ms = (len(sig) / fs) * 1000
    window_indices = get_windows(retained_ms, fs)
    
    features = []
    for start, end in window_indices:
        if end <= len(sig):
            window = sig[start:end, :]
            features.append(extract_features(window, fs))
        
    return np.array(features)