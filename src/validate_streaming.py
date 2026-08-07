"""Validate the streaming sub-block feature combiner against the direct
windowed implementation in pipeline.py.

The AVR cannot revisit samples, so it accumulates over 100 ms sub-blocks and
combines adjacent pairs into 200 ms windows. RMS and MAV combine additively.
WL and ZC do not: both depend on the difference between adjacent samples, and
the difference spanning a sub-block boundary belongs to neither block alone.

This script computes features both ways and requires exact agreement before
the same logic is moved to C++.
"""

import os
import numpy as np
from scipy import signal

from preprocess import load_trial
from pipeline import (
    quantize_10bit, extract_features, get_windows,
    V_MAX_2CH, EPSILON, FS_CLINICAL, FS_COMMODITY,
    WINDOW_MS, STEP_MS, TRANSIENT_MS,
)

SUB_MS = STEP_MS          # 100 ms sub-block = the window step


# ---------------------------------------------------------------- streaming

class SubBlockAccumulator:
    """Accumulates one 100 ms sub-block for a single channel.

    prev_sample is the last filtered value of the *preceding* sub-block, or
    None for the first. Internal WL and ZC exclude the boundary term; the
    combiner adds it back.
    """

    def __init__(self, prev_sample=None):
        self.n = 0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.wl_internal = 0.0
        self.zc_internal = 0
        self.first = None
        self.last = None
        self._prev = prev_sample

    def push(self, x):
        self.n += 1
        self.sum_sq += x * x
        self.sum_abs += abs(x)
        if self.first is None:
            self.first = x
        if self._prev is not None:
            d = x - self._prev
            self.wl_internal += abs(d)
            s_prev = 1.0 if self._prev >= 0 else -1.0
            s_curr = 1.0 if x >= 0 else -1.0
            if s_prev != s_curr and abs(d) >= EPSILON:
                self.zc_internal += 1
        self._prev = x
        self.last = x


def combine(a, b, fs):
    """Combine two adjacent sub-blocks into one 200 ms window."""
    n = a.n + b.n

    rms = np.sqrt((a.sum_sq + b.sum_sq) / n)
    mav = (a.sum_abs + b.sum_abs) / n

    # b.wl_internal already includes the boundary term, because b was
    # constructed with prev_sample = a.last. Same for b.zc_internal.
    wl = (a.wl_internal + b.wl_internal) / n
    zc = (a.zc_internal + b.zc_internal) / (n / fs)

    return rms, mav, wl, zc


def streaming_features(sig, fs):
    """Feature matrix computed the way the AVR would: one pass, no buffer."""
    n_ch = sig.shape[1]
    sub_n = round(SUB_MS * fs / 1000)
    n_sub = len(sig) // sub_n

    # accumulate every sub-block, carrying the last sample across boundaries
    blocks = [[] for _ in range(n_ch)]
    for ch in range(n_ch):
        prev = None
        for k in range(n_sub):
            acc = SubBlockAccumulator(prev_sample=prev)
            for i in range(k * sub_n, (k + 1) * sub_n):
                acc.push(sig[i, ch])
            blocks[ch].append(acc)
            prev = acc.last

    # combine adjacent pairs -> 200 ms windows
    rows = []
    for k in range(n_sub - 1):
        feats = []
        for ch in range(n_ch):
            feats.extend(combine(blocks[ch][k], blocks[ch][k + 1], fs))
        rows.append(feats)
    return np.array(rows)


# ---------------------------------------------------------------- direct

def direct_features(sig, fs):
    """Feature matrix from pipeline.py's windowing, for comparison."""
    retained_ms = (len(sig) / fs) * 1000
    out = []
    for start, end in get_windows(retained_ms, fs):
        if end <= len(sig):
            out.append(extract_features(sig[start:end, :], fs))
    return np.array(out)


# ---------------------------------------------------------------- driver

def prepare(session, participant, gesture, trial):
    """Commodity condition up to the point where windowing begins."""
    sigs = load_trial(session=session, participant=participant,
                      gesture=gesture, trial=trial)
    sig = sigs["commodity"]
    sig = signal.resample_poly(sig, up=FS_COMMODITY, down=FS_CLINICAL, axis=0)
    sig = quantize_10bit(sig, V_MAX_2CH)

    sos = signal.butter(4, 20, btype="highpass", fs=FS_COMMODITY, output="sos")
    sig = signal.sosfilt(sos, sig, axis=0)

    transient = int(TRANSIENT_MS * FS_COMMODITY / 1000)
    return sig[transient:, :]


def main():
    trials = [(1, 1, 16, 1), (1, 1, 11, 3), (1, 7, 17, 5), (3, 22, 16, 7)]

    print(f"{'trial':<22} {'windows':>8} {'max abs err':>14} {'zc mismatch':>13}")
    print("-" * 60)

    worst = 0.0
    zc_bad = 0

    for s, p, g, tr in trials:
        sig = prepare(s, p, g, tr)
        d = direct_features(sig, FS_COMMODITY)
        st = streaming_features(sig, FS_COMMODITY)

        n = min(len(d), len(st))
        if len(d) != len(st):
            print(f"  WINDOW COUNT MISMATCH: direct {len(d)}, streaming {len(st)}")

        err = np.abs(d[:n] - st[:n])
        # ZC columns are indices 3 and 7 (grouped by channel)
        zc_cols = [3, 7]
        zc_diff = int(np.sum(err[:, zc_cols] > 1e-9))
        other = np.delete(err, zc_cols, axis=1)

        worst = max(worst, other.max())
        zc_bad += zc_diff

        print(f"s{s}p{p}g{g}t{tr:<14} {n:>8} {other.max():>14.3e} {zc_diff:>13}")

    print("-" * 60)
    print(f"worst non-ZC abs error : {worst:.3e}")
    print(f"ZC mismatches          : {zc_bad}")
    print()
    if worst < 1e-9 and zc_bad == 0:
        print("PASS - streaming combiner is exactly equivalent. Port to C++.")
    else:
        print("FAIL - combiner diverges. Do not port until this is resolved.")


if __name__ == "__main__":
    main()