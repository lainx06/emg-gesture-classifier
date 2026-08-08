"""Validate the streaming sub-block feature combiner against the direct
windowed implementation in pipeline.py.

The AVR cannot revisit samples. It accumulates over 100 ms sub-blocks and
combines adjacent pairs into 200 ms windows. Because windows overlap by 50%,
each sub-block is reused in two consecutive windows, so every sub-block must
be SELF-CONTAINED: it may only hold statistics computed from its own samples.
The single difference spanning the A->B boundary belongs to neither block and
is added by combine() at the moment the pair is joined.

An earlier version initialized each block with the previous block's last
sample. That put the A->B boundary term inside B, which was then double
counted when B was reused in the B+C window.

Requires exact agreement (float64, same arithmetic) before porting to C++.
"""

import numpy as np
from scipy import signal

from preprocess import load_trial
from pipeline import (
    quantize_10bit, extract_features, get_windows,
    V_MAX_2CH, EPSILON, FS_CLINICAL, FS_COMMODITY,
    STEP_MS, TRANSIENT_MS,
)

SUB_MS = STEP_MS          # 100 ms sub-block == the window step


def _sign(x):
    """Match numpy sign with the pipeline's sign(0) -> +1 override."""
    return 1.0 if x >= 0 else -1.0


class SubBlock:
    """Statistics for one 100 ms sub-block, from its own samples only.

    wl and zc exclude any term involving a neighbouring block. first and last
    are exposed so combine() can evaluate the boundary difference.
    """

    __slots__ = ("n", "sum_sq", "sum_abs", "wl", "zc", "first", "last", "_prev")

    def __init__(self):
        self.n = 0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.wl = 0.0
        self.zc = 0
        self.first = None
        self.last = None
        self._prev = None

    def push(self, x):
        self.n += 1
        self.sum_sq += x * x
        self.sum_abs += abs(x)
        if self.first is None:
            self.first = x
        if self._prev is not None:
            d = x - self._prev
            self.wl += abs(d)
            if _sign(self._prev) != _sign(x) and abs(d) >= EPSILON:
                self.zc += 1
        self._prev = x
        self.last = x


def combine(a, b, fs):
    """Join two adjacent sub-blocks into one 200 ms window.

    Adds the single boundary term a.last -> b.first that neither block owns.
    """
    n = a.n + b.n
    d = b.first - a.last

    rms = np.sqrt((a.sum_sq + b.sum_sq) / n)
    mav = (a.sum_abs + b.sum_abs) / n
    wl = (a.wl + b.wl + abs(d)) / n

    zc_count = a.zc + b.zc
    if _sign(a.last) != _sign(b.first) and abs(d) >= EPSILON:
        zc_count += 1
    zc = zc_count / (n / fs)

    return rms, mav, wl, zc


def streaming_features(sig, fs):
    """Feature matrix computed the way the AVR will: one pass, no window buffer."""
    n_ch = sig.shape[1]
    sub_n = round(SUB_MS * fs / 1000)
    n_sub = len(sig) // sub_n

    blocks = [[SubBlock() for _ in range(n_sub)] for _ in range(n_ch)]
    for ch in range(n_ch):
        for k in range(n_sub):
            blk = blocks[ch][k]
            for i in range(k * sub_n, (k + 1) * sub_n):
                blk.push(sig[i, ch])

    rows = []
    for k in range(n_sub - 1):
        feats = []
        for ch in range(n_ch):
            feats.extend(combine(blocks[ch][k], blocks[ch][k + 1], fs))
        rows.append(feats)
    return np.array(rows)


def direct_features(sig, fs):
    """Feature matrix from pipeline.py's windowing, for comparison."""
    retained_ms = (len(sig) / fs) * 1000
    out = []
    for start, end in get_windows(retained_ms, fs):
        if end <= len(sig):
            out.append(extract_features(sig[start:end, :], fs))
    return np.array(out)


# ------------------------------------------------------------ synthetic

def synthetic_boundary_tests(fs=FS_COMMODITY, n=3500, seed=0):
    """Stress the boundary logic on cases real data may not produce.

    Covers zeros at the seam, differences exactly at the deadzone, and sign
    changes straddling the seam. Modes 5 and 6 are the important ones: they
    place a genuine sign crossing exactly at and just below the deadzone
    threshold, so both branches of `sc & ac` are exercised together. Mode 3
    alone does not do this, because the pipeline treats sign(0) as positive
    and therefore registers no crossing.
    """
    rng = np.random.default_rng(seed)
    fails = 0

    for t in range(n):
        mode = t % 7
        if mode == 0:
            x = rng.normal(0, 0.05, 200)
        elif mode == 1:
            x = rng.normal(0, 0.005, 200)          # mostly below deadzone
        elif mode == 2:
            x = np.round(rng.normal(0, 0.05, 200) / EPSILON) * EPSILON
        elif mode == 3:
            x = rng.normal(0, 0.05, 200)
            x[99] = 0.0                            # zero exactly at the seam
            x[100] = EPSILON                       # diff at threshold, no crossing
        elif mode == 4:
            x = np.zeros(200)
            x[100:] = 0.1                          # step straight across seam
        elif mode == 5:
            x = rng.normal(0, 0.05, 200)
            x[99] = -EPSILON / 2                   # crossing, diff == EPSILON
            x[100] = EPSILON / 2                   # should count
        else:
            x = rng.normal(0, 0.05, 200)
            x[99] = -EPSILON / 2 + 1e-9            # crossing, diff just under
            x[100] = EPSILON / 2                   # should NOT count

        sig = x.reshape(-1, 1)
        d = direct_features(sig, fs)
        s = streaming_features(sig, fs)
        if d.shape != s.shape or not np.allclose(d, s, rtol=0, atol=1e-12):
            fails += 1
            if fails <= 3:
                print(f"  synthetic mode {mode} failed, max err "
                      f"{np.abs(d - s).max():.3e}")
    return fails


# ------------------------------------------------------------ driver

def prepare(session, participant, gesture, trial):
    """Commodity condition, up to the point where windowing begins."""
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
    print("=== synthetic boundary tests ===")
    sf = synthetic_boundary_tests()
    print(f"failures: {sf} / 3500\n")

    trials = [(1, 1, 16, 1), (1, 1, 11, 3), (1, 7, 17, 5), (3, 22, 16, 7),
              (2, 15, 11, 2), (3, 40, 17, 6)]

    print("=== real trials ===")
    print(f"{'trial':<20} {'windows':>8} {'max abs err':>14} {'zc mismatch':>13}")
    print("-" * 58)

    worst, zc_bad, count_bad = 0.0, 0, 0

    for s, p, g, tr in trials:
        try:
            sig = prepare(s, p, g, tr)
        except Exception as e:
            print(f"s{s}p{p}g{g}t{tr:<11} skipped: {type(e).__name__}")
            continue

        d = direct_features(sig, FS_COMMODITY)
        st = streaming_features(sig, FS_COMMODITY)

        if len(d) != len(st):
            count_bad += 1
            print(f"  WINDOW COUNT MISMATCH: direct {len(d)}, streaming {len(st)}")
        n = min(len(d), len(st))

        err = np.abs(d[:n] - st[:n])
        zc_cols = [3, 7]
        zc_diff = int(np.sum(err[:, zc_cols] > 1e-9))
        other = np.delete(err, zc_cols, axis=1)

        worst = max(worst, other.max())
        zc_bad += zc_diff
        print(f"s{s}p{p}g{g}t{tr:<11} {n:>8} {other.max():>14.3e} {zc_diff:>13}")

    print("-" * 58)
    print(f"worst non-ZC abs error : {worst:.3e}")
    print(f"ZC mismatches          : {zc_bad}")
    print(f"window count mismatches: {count_bad}")
    print(f"synthetic failures     : {sf}")
    print()

    if worst < 1e-12 and zc_bad == 0 and count_bad == 0 and sf == 0:
        print("PASS - streaming combiner is exactly equivalent. Port to C++.")
    else:
        print("FAIL - combiner diverges. Do not port until this is resolved.")


if __name__ == "__main__":
    main()