"""Export everything needed for the Python-to-AVR equivalence check.

Artifacts written to <root>/playback/:

  playback_params.h            PROGMEM constants for playback_verify.ino
  embedded_reference.py        strict float32 mirror of the AVR arithmetic
  playback_vectors.npz         staged reference outputs for every test trial
  codes_<trial>.bin            one file per trial, interleaved int16 ADC codes

Three-level comparison this supports:

  A  pipeline.py           scipy / sklearn, float64      scientific reference
  B  embedded_reference    manual float32, AVR op order  embedded numerical
                                                         implementation effect
  C  playback_verify.ino   physical ATmega328P           C++ transcription error

A->B is not purely a single-precision effect. B also multiplies by a
precomputed reciprocal instead of dividing, and accumulates the LDA dot
product in explicit index order rather than deferring to a vectorized sum.
Those are deliberate deployment choices, and A->B measures their combined
numerical consequence.

The filter is transposed direct form II, matching scipy.signal.sosfilt's
recurrence exactly, so filter structure contributes nothing to A->B.

ONE FILE PER TRIAL. Each ref.run() starts from zero filter state and performs
its own 200-sample warmup. Streaming a concatenated file would carry filter
state across trial boundaries on the AVR but not in the reference, which would
show up as a B->C divergence unrelated to implementation correctness.

Prediction-agreement guarantee: if E is the maximum absolute per-score error
between two levels, the winning score can fall by E while the runner-up rises
by E, so the observed margin shrinks by 2E. A sufficient worst-case condition
for argmax preservation is 2E < min_margin. It is not necessary: predictions
may still agree when it fails.
"""

import os
import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from preprocess import load_trial
from pipeline import (
    quantize_10bit, V_MAX_2CH, EPSILON,
    FS_CLINICAL, FS_COMMODITY, WINDOW_MS, STEP_MS, TRANSIENT_MS,
)

f32 = np.float32

SUBJECT = 1
GESTURES = [11, 16, 17]
GEST_MAP = {11: 0, 16: 1, 17: 2}
TRAIN_TRIALS = [1, 2, 3, 4, 5]

TEST_TRIALS = [
    (s, g, t)
    for s in (1, 2, 3)
    for g in (11, 16, 17)
    for t in (6, 7)
]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "playback")
os.makedirs(OUT, exist_ok=True)

LSB = (2.0 * V_MAX_2CH) / 1023.0
SUB_N = round(STEP_MS * FS_COMMODITY / 1000)
WIN_N = round(WINDOW_MS * FS_COMMODITY / 1000)
TRANSIENT_N = int(TRANSIENT_MS * FS_COMMODITY / 1000)


# ------------------------------------------------------------------ stages

def adc_codes(session, gesture, trial):
    """Everything up to and including quantization, as 0..1023 codes.

    This is the playback boundary: exactly what analogRead would produce.
    """
    sigs = load_trial(session=session, participant=SUBJECT,
                      gesture=gesture, trial=trial)
    sig = sigs["commodity"]
    sig = signal.resample_poly(sig, up=FS_COMMODITY, down=FS_CLINICAL, axis=0)
    return np.clip(np.round((sig + V_MAX_2CH) / LSB), 0, 1023).astype(np.int16)


def codes_to_mv(codes):
    """Reconstruction the AVR performs on receipt."""
    return (codes.astype(np.float64) * LSB) - V_MAX_2CH


def features_float64(codes, sos):
    """Level A from ADC codes onward, using the existing pipeline functions."""
    from pipeline import extract_features, get_windows
    sig = signal.sosfilt(sos, codes_to_mv(codes), axis=0)[TRANSIENT_N:, :]
    retained_ms = (len(sig) / FS_COMMODITY) * 1000
    out = []
    for s, e in get_windows(retained_ms, FS_COMMODITY):
        if e <= len(sig):
            out.append(extract_features(sig[s:e, :], FS_COMMODITY))
    return np.array(out)


# ------------------------------------------------------------------ training

def build_model():
    from pipeline import process_trial
    X, y = [], []
    for g in GESTURES:
        for tr in TRAIN_TRIALS:
            sigs = load_trial(session=1, participant=SUBJECT, gesture=g, trial=tr)
            feats = process_trial(sigs["clinical"], condition="commodity")
            X.append(feats)
            y.extend([GEST_MAP[g]] * feats.shape[0])
    X, y = np.vstack(X), np.array(y)

    scaler = StandardScaler().fit(X)
    lda = LinearDiscriminantAnalysis().fit(scaler.transform(X), y)

    # coef_ rows are ordered by classes_. If a class were absent from training,
    # every downstream index would silently shift.
    assert np.array_equal(lda.classes_, [0, 1, 2]), \
        f"unexpected class ordering: {lda.classes_}"
    assert lda.coef_.shape == (3, 8), f"unexpected coef_ shape: {lda.coef_.shape}"

    return scaler, lda, X.shape[0]


# ------------------------------------------------------------------ emitters

def c_array(name, values, fmt="%.9g"):
    body = ", ".join(fmt % v for v in np.asarray(values).ravel())
    return (f"const float {name}[] PROGMEM = {{ {body} }};\n"
            f"const uint8_t {name}_LEN = {np.asarray(values).size};\n")


def emit_header(sos, scaler, lda):
    sos32 = sos.astype(f32)
    mean32 = scaler.mean_.astype(f32)
    inv_scale32 = (1.0 / scaler.scale_).astype(f32)
    coef32 = lda.coef_.astype(f32)
    icpt32 = lda.intercept_.astype(f32)

    lines = [
        "// Generated by export_playback.py -- do not edit by hand.",
        "// Filter: transposed direct form II, matching scipy.signal.sosfilt.",
        "#pragma once",
        "#include <avr/pgmspace.h>",
        "",
        "#define N_CH        2",
        "#define N_FEAT      8",
        "#define N_CLASS     3",
        f"#define N_SECTIONS  {sos.shape[0]}",
        f"#define SUB_N       {SUB_N}",
        f"#define WIN_N       {WIN_N}",
        f"#define TRANSIENT_N {TRANSIENT_N}",
        "",
        f"const float V_MAX   = {V_MAX_2CH:.9g}f;",
        f"const float ADC_LSB = {LSB:.9g}f;",
        f"const float ZC_EPS  = {EPSILON:.9g}f;",
        f"const float FS      = {float(FS_COMMODITY):.1f}f;",
        "",
        "// b0 b1 b2 a0 a1 a2 per section, row major",
        c_array("SOS", sos32),
        "// scaler: x_scaled = (x - MEAN) * INV_SCALE   (multiply, never divide)",
        c_array("SCALER_MEAN", mean32),
        c_array("SCALER_INV_SCALE", inv_scale32),
        "// LDA: score[c] = INTERCEPT[c] + sum_{f=0..7} COEF[c*N_FEAT+f]*x_scaled[f]",
        c_array("LDA_COEF", coef32),
        c_array("LDA_INTERCEPT", icpt32),
    ]
    path = os.path.join(OUT, "playback_params.h")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path, sos32, mean32, inv_scale32, coef32, icpt32


REFERENCE_SRC = '''"""Level B: strict float32 mirror of the AVR arithmetic.

Generated by export_playback.py. Every intermediate is cast to float32 after
each operation, so this reproduces single-precision accumulation rather than
float64 with a final cast. Operation order matches playback_verify.ino:

  filter      transposed direct form II, sample by sample through both sections
  accumulate  self-contained 100 ms sub-blocks, boundary term added on join
  scale       (x - mean) * inv_scale, multiply not divide
  LDA         dot product accumulated f = 0..7 in index order

run() resets filter state on every call. The AVR must do the same at each
trial boundary, which is why codes are exported one file per trial.
"""

import numpy as np

f32 = np.float32


def _sign(x):
    return f32(1.0) if x >= 0 else f32(-1.0)


class SubBlock:
    __slots__ = ("n", "sum_sq", "sum_abs", "wl", "zc", "first", "last")

    def __init__(self):
        self.n = 0
        self.sum_sq = f32(0.0)
        self.sum_abs = f32(0.0)
        self.wl = f32(0.0)
        self.zc = 0
        self.first = None
        self.last = None

    def push(self, x):
        x = f32(x)
        self.n += 1
        self.sum_sq = f32(self.sum_sq + f32(x * x))
        self.sum_abs = f32(self.sum_abs + f32(abs(x)))
        if self.first is None:
            self.first = x
        else:
            d = f32(x - self.last)
            self.wl = f32(self.wl + f32(abs(d)))
            if _sign(self.last) != _sign(x) and abs(d) >= EPS:
                self.zc += 1
        self.last = x


def combine(a, b):
    n = a.n + b.n
    d = f32(b.first - a.last)
    rms = f32(np.sqrt(f32(f32(a.sum_sq + b.sum_sq) / f32(n))))
    mav = f32(f32(a.sum_abs + b.sum_abs) / f32(n))
    wl = f32(f32(f32(a.wl + b.wl) + f32(abs(d))) / f32(n))
    zc_n = a.zc + b.zc
    if _sign(a.last) != _sign(b.first) and abs(d) >= EPS:
        zc_n += 1
    zc = f32(f32(zc_n) / f32(f32(n) / FS))
    return rms, mav, wl, zc


def run(codes):
    """codes: (N, 2) int16 in 0..1023. Filter state starts at zero."""
    n_samp = codes.shape[0]
    z = [[[f32(0.0), f32(0.0)] for _ in range(N_CH)] for _ in range(N_SECTIONS)]
    filtered = np.empty((n_samp, N_CH), dtype=np.float32)

    for i in range(n_samp):
        for ch in range(N_CH):
            v = f32(f32(f32(codes[i, ch]) * ADC_LSB) - V_MAX)
            for s in range(N_SECTIONS):
                b0, b1, b2, _, a1, a2 = SOS[s]
                y = f32(f32(b0 * v) + z[s][ch][0])
                z[s][ch][0] = f32(f32(f32(b1 * v) - f32(a1 * y)) + z[s][ch][1])
                z[s][ch][1] = f32(f32(b2 * v) - f32(a2 * y))
                v = y
            filtered[i, ch] = v

    sig = filtered[TRANSIENT_N:, :]
    n_sub = sig.shape[0] // SUB_N

    blocks = [[SubBlock() for _ in range(n_sub)] for _ in range(N_CH)]
    for ch in range(N_CH):
        for k in range(n_sub):
            blk = blocks[ch][k]
            for i in range(k * SUB_N, (k + 1) * SUB_N):
                blk.push(sig[i, ch])

    feats, scaled, scores, preds = [], [], [], []
    for k in range(n_sub - 1):
        row = []
        for ch in range(N_CH):
            row.extend(combine(blocks[ch][k], blocks[ch][k + 1]))
        row = np.array(row, dtype=np.float32)
        feats.append(row)

        sc = np.empty(N_FEAT, dtype=np.float32)
        for f in range(N_FEAT):
            sc[f] = f32(f32(row[f] - SCALER_MEAN[f]) * SCALER_INV_SCALE[f])
        scaled.append(sc)

        s = np.empty(N_CLASS, dtype=np.float32)
        for c in range(N_CLASS):
            acc = f32(LDA_INTERCEPT[c])
            for f in range(N_FEAT):
                acc = f32(acc + f32(LDA_COEF[c][f] * sc[f]))
            s[c] = acc
        scores.append(s)
        preds.append(int(np.argmax(s)))

    return {
        "filtered": filtered,
        "features": np.array(feats, dtype=np.float32),
        "scaled": np.array(scaled, dtype=np.float32),
        "scores": np.array(scores, dtype=np.float32),
        "predictions": np.array(preds, dtype=np.int8),
    }
'''


def emit_reference(sos32, mean32, inv32, coef32, icpt32):
    consts = [
        "# --- generated constants ---",
        "N_CH = 2",
        "N_FEAT = 8",
        "N_CLASS = 3",
        f"N_SECTIONS = {sos32.shape[0]}",
        f"SUB_N = {SUB_N}",
        f"WIN_N = {WIN_N}",
        f"TRANSIENT_N = {TRANSIENT_N}",
        f"V_MAX = f32({V_MAX_2CH!r})",
        f"ADC_LSB = f32({LSB!r})",
        f"EPS = f32({EPSILON!r})",
        f"FS = f32({float(FS_COMMODITY)!r})",
        f"SOS = np.array({sos32.tolist()!r}, dtype=np.float32)",
        f"SCALER_MEAN = np.array({mean32.tolist()!r}, dtype=np.float32)",
        f"SCALER_INV_SCALE = np.array({inv32.tolist()!r}, dtype=np.float32)",
        f"LDA_COEF = np.array({coef32.tolist()!r}, dtype=np.float32)",
        f"LDA_INTERCEPT = np.array({icpt32.tolist()!r}, dtype=np.float32)",
        "",
    ]
    src = REFERENCE_SRC.replace("f32 = np.float32\n",
                                "f32 = np.float32\n\n" + "\n".join(consts))
    path = os.path.join(OUT, "embedded_reference.py")
    with open(path, "w") as fh:
        fh.write(src)
    return path


# ------------------------------------------------------------------ main

def main():
    sos = signal.butter(4, 20, btype="highpass", fs=FS_COMMODITY, output="sos")
    # The TDF-II recurrence below drops a0 entirely; it is only valid when
    # the sections are already normalized.
    assert np.allclose(sos[:, 3], 1.0), "TDF-II implementation assumes a0 == 1"
    print(f"filter: {sos.shape[0]} biquad sections, transposed direct form II, "
          f"a0 == 1 verified")

    scaler, lda, n_train = build_model()
    print(f"model:  subject {SUBJECT}, {n_train} training windows, "
          f"classes {lda.classes_}")

    hpath, sos32, mean32, inv32, coef32, icpt32 = emit_header(sos, scaler, lda)
    rpath = emit_reference(sos32, mean32, inv32, coef32, icpt32)
    print(f"wrote {hpath}")
    print(f"wrote {rpath}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("embedded_reference", rpath)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    vectors, margins, score_errs, std_errs = {}, [], [], []
    zc_total = 0

    print()
    print(f"{'trial':<14} {'win':>4} {'feat rel':>10} {'std feat SD':>12} "
          f"{'filt LSB':>10} {'score':>10} {'zc':>4} {'pred':>7} "
          f"{'min margin':>11}")
    print("-" * 88)

    for sess, gest, tr in TEST_TRIALS:
        try:
            codes = adc_codes(sess, gest, tr)
        except Exception as e:
            print(f"s{sess}g{gest}t{tr:<7} skipped: {type(e).__name__}")
            continue

        A_feat = features_float64(codes, sos)
        B = ref.run(codes)
        n = min(len(A_feat), len(B["features"]))

        # ZC is a discrete count; it must match exactly and is reported
        # separately rather than folded into a relative-error statistic.
        ZC_COLS = [3, 7]
        zc_bad = int(np.sum(np.abs(A_feat[:n, ZC_COLS]
                                   - B["features"][:n, ZC_COLS]) > 1e-6))
        zc_total += zc_bad
        cont_A = np.delete(A_feat[:n], ZC_COLS, axis=1)
        cont_B = np.delete(B["features"][:n], ZC_COLS, axis=1)
        rel = np.abs(cont_A - cont_B) / (np.abs(cont_A) + 1e-12)

        A_filt = signal.sosfilt(sos, codes_to_mv(codes), axis=0)
        filt_lsb = np.abs(A_filt - B["filtered"]).max() / LSB

        A_scaled = scaler.transform(A_feat[:n])
        # Standardized features are the input the classifier actually sees.
        # Relative feature error is unstable near zero; this is in units of
        # training standard deviations and is the meaningful bound.
        std_err = float(np.abs(A_scaled - B["scaled"][:n]).max())
        std_errs.append(std_err)
        A_score = lda.decision_function(A_scaled)
        A_pred = A_score.argmax(axis=1)

        score_err = float(np.abs(A_score - B["scores"][:n]).max())
        score_errs.append(score_err)
        agree = int(np.sum(A_pred == B["predictions"][:n]))

        s = np.sort(B["scores"][:n], axis=1)
        margin = float((s[:, -1] - s[:, -2]).min())
        margins.append(margin)

        key = f"s{sess}g{gest}t{tr}"
        vectors.update({
            f"{key}_codes": codes,
            f"{key}_A_features": A_feat,
            f"{key}_A_scores": A_score,
            f"{key}_A_predictions": A_pred,
            f"{key}_B_filtered": B["filtered"],
            f"{key}_B_features": B["features"],
            f"{key}_B_scaled": B["scaled"],
            f"{key}_B_scores": B["scores"],
            f"{key}_B_predictions": B["predictions"],
        })

        binpath = os.path.join(OUT, f"codes_{key}.bin")
        codes.astype("<i2").tofile(binpath)

        print(f"{key:<14} {n:>4} {rel.max():>10.2e} {std_err:>12.2e} "
              f"{filt_lsb:>10.5f} {score_err:>10.2e} {zc_bad:>4} "
              f"{agree}/{n:<4} {margin:>11.4f}")

    print("-" * 88)
    print(f"A-B ZC mismatches (must be 0):      {zc_total}")
    if std_errs:
        print(f"max A-B standardized-feature error: {max(std_errs):.3e} SD")

    if margins:
        E = max(score_errs)
        M = min(margins)
        print(f"max A-B per-score error   E = {E:.3e}")
        print(f"minimum decision margin   M = {M:.4f}")
        print(f"sufficient worst-case guarantee is 2E < M:  2E = {2*E:.3e}  "
              f"{'<' if 2*E < M else '>='} M")
        print("  (2E < M guarantees the winner cannot be overtaken.\n"
              "   Predictions may still agree when it fails, so a failure\n"
              "   here is not evidence of disagreement.)")
        print()
        print(f"budget for the AVR: B->C per-score error must stay below "
              f"M/2 = {M/2:.4f}")

    npz = os.path.join(OUT, "playback_vectors.npz")
    np.savez_compressed(npz, **vectors)
    print(f"\nwrote {npz}")
    print(f"wrote one codes_<trial>.bin per trial "
          f"(interleaved little-endian int16, ch0 ch1 per sample)")
    print("stream each file separately; reset AVR filter state between trials")


if __name__ == "__main__":
    main()