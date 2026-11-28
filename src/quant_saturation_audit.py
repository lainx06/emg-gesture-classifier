"""Quantizer saturation audit (paper Section 3.2).

For every trial (43 subjects x 3 sessions x 3 gestures x 7 trials = 2,709) and
each quantized condition, counts samples whose magnitude exceeds V_MAX at the
quantizer input, i.e. samples that saturate to an endpoint code. The signal is
taken at the same point as in pipeline.process_trial:

  commodity   F1F9, F5F13, after resampling to 1000 Hz   V_MAX_2CH
  depth_only  8 channels at 2048 Hz                      V_MAX_8CH_2048
  rate_depth  8 channels, after resampling to 1000 Hz    V_MAX_8CH_1000

Prints clipped and total sample counts per condition and session. The paper
reports about 0.10% of Session 1 samples (the 99.9th-percentile calibration
target) and a maximum of 0.0672% across the cross-session evaluations
(Sessions 2 and 3). Per-trial maxima are printed as well.
"""
import sys

import numpy as np
from scipy import signal

from preprocess import load_trial
from pipeline import FS_CLINICAL, FS_COMMODITY, V_MAX_2CH, V_MAX_8CH_1000, V_MAX_8CH_2048

SUBJECTS = range(1, 44)
SESSIONS = (1, 2, 3)
GESTURES = (11, 16, 17)
TRIALS = range(1, 8)
CONDITIONS = ("commodity", "depth_only", "rate_depth")


def quantizer_inputs(sig_clin):
    rs = signal.resample_poly(sig_clin, up=FS_COMMODITY, down=FS_CLINICAL, axis=0)
    return {
        "commodity":  (rs[:, [0, 4]], V_MAX_2CH),
        "depth_only": (sig_clin, V_MAX_8CH_2048),
        "rate_depth": (rs, V_MAX_8CH_1000),
    }


def main():
    clipped = {(c, s): 0 for c in CONDITIONS for s in SESSIONS}
    total = {(c, s): 0 for c in CONDITIONS for s in SESSIONS}
    trial_max = {(c, s): 0.0 for c in CONDITIONS for s in SESSIONS}
    failures, n_trials = [], 0

    for subj in SUBJECTS:
        for sess in SESSIONS:
            for gest in GESTURES:
                for trial in TRIALS:
                    try:
                        sig = load_trial(session=sess, participant=subj,
                                         gesture=gest, trial=trial)["clinical"]
                    except (FileNotFoundError, OSError, ValueError, AssertionError) as e:
                        failures.append((subj, sess, gest, trial, type(e).__name__))
                        continue
                    n_trials += 1
                    for cond, (x, v_max) in quantizer_inputs(sig).items():
                        k = int(np.count_nonzero(np.abs(x) > v_max))
                        clipped[(cond, sess)] += k
                        total[(cond, sess)] += x.size
                        trial_max[(cond, sess)] = max(trial_max[(cond, sess)],
                                                      100.0 * k / x.size)
        if subj % 5 == 0 or subj == SUBJECTS[-1]:
            print(f"Processed Subject {subj}/{SUBJECTS[-1]}")

    expected = len(SUBJECTS) * len(SESSIONS) * len(GESTURES) * len(TRIALS)
    print(f"\nAudited {n_trials} of {expected} trials.")
    if failures:
        print(f"FAIL: {len(failures)} trial(s) could not be read, e.g. {failures[:3]}")
        sys.exit(1)

    print(f"\n{'condition':<12} {'session':>7} {'clipped':>12} {'samples':>14} "
          f"{'pooled %':>10} {'worst trial %':>14}")
    print("-" * 74)
    for cond in CONDITIONS:
        for sess in SESSIONS:
            c, t = clipped[(cond, sess)], total[(cond, sess)]
            print(f"{cond:<12} {sess:>7} {c:>12,} {t:>14,} "
                  f"{100.0 * c / t:>10.4f} {trial_max[(cond, sess)]:>14.4f}")

    cross = [100.0 * clipped[(c, s)] / total[(c, s)] for c in CONDITIONS for s in (2, 3)]
    s1 = [100.0 * clipped[(c, 1)] / total[(c, 1)] for c in CONDITIONS]
    print(f"\nSession 1, pooled, by condition:       "
          + ", ".join(f"{v:.4f}%" for v in s1) + "   (paper: about 0.10%)")
    print(f"Sessions 2-3, maximum pooled rate:     {max(cross):.4f}%   (paper: 0.0672%)")


if __name__ == "__main__":
    main()
