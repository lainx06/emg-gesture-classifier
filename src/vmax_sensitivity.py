"""Sensitivity of results to the quantizer calibration pool.

The quantizer thresholds in pipeline.py were calibrated on all Session 1 trials,
including the held-out Day-1 test trials 6-7 (disclosed in the paper). This
script recalibrates on the training trials (1-5) only, rebuilds the features of
the three quantized conditions with those thresholds, and compares Random
Forest and LDA Day-1 accuracy and Day-1-to-Day-29 drop against the cached
features built with the published thresholds.

Needs data/grabmyo/ and the six caches from build_cache.py. pipeline.py is not
modified: its threshold constants are overridden in memory for the rebuild.

    python src/vmax_sensitivity.py
"""
import contextlib
import io
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import pipeline
from calibrate_vmax import calibrate_vmax
from preprocess import load_trial
from separability import run_evaluation

SUBJECTS = range(1, 44)
SESSIONS = range(1, 4)
GESTURES = [11, 16, 17]
GEST_MAP = {11: 0, 16: 1, 17: 2}
TRIALS = range(1, 8)
CONDITIONS = ("depth_only", "rate_depth", "commodity")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "cache")
OUT = os.path.join(CACHE, "sensitivity")


def models():
    return {"RF": lambda: RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            "LDA": LinearDiscriminantAnalysis}


def rebuild(condition):
    """Same loop order and labelling as build_cache.py."""
    X, Y, G = [], [], []
    for subj in SUBJECTS:
        for sess in SESSIONS:
            for gest in GESTURES:
                for trial in TRIALS:
                    sig = load_trial(session=sess, participant=subj,
                                     gesture=gest, trial=trial)["clinical"]
                    feats = pipeline.process_trial(sig, condition=condition)
                    X.append(feats)
                    Y.extend([GEST_MAP[gest]] * len(feats))
                    G.extend([[subj, sess, trial, w] for w in range(len(feats))])
    return np.vstack(X), np.array(Y, dtype=np.int8), np.array(G, dtype=np.int16)


def evaluate(path):
    out = {}
    for name, make in models().items():
        d1 = run_evaluation(path, make(), "within")
        d29 = run_evaluation(path, make(), "cross_max")
        subs = sorted(set(d1) & set(d29))
        out[name] = (np.array([d1[s] for s in subs]),
                     np.array([d1[s] - d29[s] for s in subs]))
    return out


def main():
    print("Recalibrating on training trials 1-5 only ...")
    with contextlib.redirect_stdout(io.StringIO()):
        new = calibrate_vmax(range(1, 6))
    old = {"V_MAX_2CH": pipeline.V_MAX_2CH,
           "V_MAX_8CH_2048": pipeline.V_MAX_8CH_2048,
           "V_MAX_8CH_1000": pipeline.V_MAX_8CH_1000}
    for k in old:
        print(f"  {k:<15} published {old[k]:.4f} mV   training-only {new[k]:.4f} mV")
        setattr(pipeline, k, new[k])

    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'condition':<11} {'model':<5} {'Day 1 (pub)':>11} {'Day 1 (train)':>13} "
          f"{'drop (pub)':>10} {'drop (train)':>12} {'max |diff| D1/drop':>19}")
    print("-" * 88)
    for cond in CONDITIONS:
        X, Y, G = rebuild(cond)
        with np.load(os.path.join(CACHE, f"{cond}_cache.npz")) as ref:
            if not (np.array_equal(ref["Y"], Y) and np.array_equal(ref["GROUPS"], G)):
                print(f"FAIL: rebuilt {cond} rows do not align with the cache")
                sys.exit(1)
        path = os.path.join(OUT, f"{cond}_trainonly_vmax.npz")
        np.savez_compressed(path, X=X, Y=Y, GROUPS=G)

        pub = evaluate(os.path.join(CACHE, f"{cond}_cache.npz"))
        trn = evaluate(path)
        for m in models():
            (d1p, drp), (d1t, drt) = pub[m], trn[m]
            print(f"{cond:<11} {m:<5} {100 * d1p.mean():>10.2f}% {100 * d1t.mean():>12.2f}% "
                  f"{100 * drp.mean():>9.2f} {100 * drt.mean():>12.2f} "
                  f"{100 * np.abs(d1p - d1t).max():>9.2f} / {100 * np.abs(drp - drt).max():.2f}")
    print("\nAccuracies in percent and drops in percentage points, means over subjects;"
          "\nthe last column is the largest per-subject change.")


if __name__ == "__main__":
    main()
