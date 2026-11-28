"""Quantizer full-scale (V_MAX) calibration.

Computes the 99.9th percentile of |signal| at the exact point in the chain
where each quantizer runs (see pipeline.process_trial):

  V_MAX_8CH_2048  8 channels, before resampling    -> depth_only
  V_MAX_8CH_1000  8 channels, after resampling     -> rate_depth
  V_MAX_2CH       F1F9 and F5F13 (derivations 0 and 4), after resampling
                                                   -> commodity

By default the pool is every Session 1 trial (trials 1-7) of every subject,
which is the calibration the paper reports. --train-only restricts the pool
to the training trials (1-5) as a sensitivity check; it prints values but does
not change pipeline.py.

The run fails unless every expected trial was read, so a partial dataset
cannot silently produce a different calibration.
"""
import argparse
import sys

import numpy as np
from scipy import signal

from preprocess import load_trial

SUBJECTS = range(1, 44)
GESTURES = [11, 16, 17]
COMMODITY_DERIVATIONS = [0, 4]          # F1F9, F5F13 -- matches pipeline.py
CHANNELS = ["F1F9", "F2F10", "F3F11", "F4F12", "F5F13", "F6F14", "F7F15", "F8F16"]

# Values hard-coded in pipeline.py, for comparison.
PIPELINE_VALUES = {"V_MAX_8CH_2048": 0.6407, "V_MAX_8CH_1000": 0.6395,
                   "V_MAX_2CH": 0.3329}


def calibrate_vmax(trials):
    expected = len(SUBJECTS) * len(GESTURES) * len(trials)
    print(f"Scanning Session 1, trials {list(trials)}, all {len(SUBJECTS)} "
          f"subjects ({expected} trials expected)...")

    amp_raw, amp_resampled, failures = [], [], []

    for subj in SUBJECTS:
        for gest in GESTURES:
            for trial in trials:
                try:
                    sigs = load_trial(session=1, participant=subj,
                                      gesture=gest, trial=trial)
                except (FileNotFoundError, OSError, ValueError, AssertionError) as e:
                    failures.append((subj, gest, trial, type(e).__name__, str(e)))
                    continue
                sig_clin = sigs["clinical"]
                amp_raw.append(np.abs(sig_clin))
                sig_rs = signal.resample_poly(sig_clin, up=1000, down=2048, axis=0)
                amp_resampled.append(np.abs(sig_rs))
        if subj % 5 == 0 or subj == SUBJECTS[-1]:
            print(f"Processed Subject {subj}/{SUBJECTS[-1]}")

    print(f"\nRead {len(amp_raw)} of {expected} trials.")
    if failures:
        print(f"FAIL: {len(failures)} trial(s) could not be read. First few:")
        for f in failures[:5]:
            print(f"  subject {f[0]}, gesture {f[1]}, trial {f[2]}: {f[3]}: {f[4]}")
        sys.exit(1)

    global_raw = np.concatenate(amp_raw, axis=0)
    global_resampled = np.concatenate(amp_resampled, axis=0)
    del amp_raw, amp_resampled

    per_channel = np.percentile(global_raw, 99.9, axis=0)
    results = {
        "V_MAX_8CH_2048": float(np.percentile(global_raw, 99.9)),
        "V_MAX_8CH_1000": float(np.percentile(global_resampled, 99.9)),
        # Resampling is applied per column, so these two columns of the
        # resampled 8-channel signal equal the resampled 2-channel signal.
        "V_MAX_2CH": float(np.percentile(
            global_resampled[:, COMMODITY_DERIVATIONS], 99.9)),
    }

    print("\n--- 99.9th percentile per channel (raw 2048 Hz) ---")
    for name, val in zip(CHANNELS, per_channel):
        print(f"  {name}: {val:.4f} mV")

    print("\n--- Quantizer thresholds ---")
    print(f"{'threshold':<16} {'computed':>10} {'pipeline.py':>12}")
    for key, val in results.items():
        print(f"{key:<16} {val:>10.4f} {PIPELINE_VALUES[key]:>12.4f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-only", action="store_true",
                    help="use only training trials 1-5 (sensitivity check)")
    args = ap.parse_args()
    calibrate_vmax(range(1, 6) if args.train_only else range(1, 8))
