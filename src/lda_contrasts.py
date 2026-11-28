"""Four LDA hardware-by-session contrasts (paper Section 4.1).

Paired two-sided Wilcoxon signed-rank tests on per-subject LDA accuracies:

  1. clinical vs commodity, Day 1
  2. clinical vs commodity, Day 29
  3. Day 1 vs Day 29, clinical
  4. Day 1 vs Day 29, commodity

Bonferroni threshold 0.05 / 4 = 0.0125. Accuracies come from
separability.run_evaluation, the same evaluator used for the primary analysis
(train on Session 1 trials 1-5, test on trials 6-7 of the target session).

Expected output, as reported in the paper (least significant p = 1.7e-3):
3.6e-8, 1.7e-3, 2.7e-4 and 1.1e-4 respectively.
"""
import os
import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from separability import run_evaluation

N_SUBJECTS = 43
N_SESSIONS, N_GESTURES, N_TRIALS, N_WINDOWS = 3, 3, 7, 37
ALPHA = 0.05 / 4


def check_complete(path):
    """Every subject x session x gesture x trial must have exactly 37 windows.

    The evaluator returns an accuracy even if a recording is missing (for
    example 185 of 222 test windows), so completeness is checked up front.
    """
    with np.load(path) as d:
        Y, G = d["Y"], d["GROUPS"]
    keys = np.column_stack([G[:, 0], G[:, 1], Y, G[:, 2]])
    cells, counts = np.unique(keys, axis=0, return_counts=True)
    n_cells = N_SUBJECTS * N_SESSIONS * N_GESTURES * N_TRIALS
    bad = cells[counts != N_WINDOWS]
    if len(cells) != n_cells or len(bad) or len(Y) != n_cells * N_WINDOWS:
        print(f"FAIL: {os.path.basename(path)} is incomplete: {len(cells)} of "
              f"{n_cells} recordings, {len(Y)} of {n_cells * N_WINDOWS} windows, "
              f"{len(bad)} recording(s) without {N_WINDOWS} windows")
        sys.exit(1)


def main():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "cache")
    clin = os.path.join(base, "clinical_cache.npz")
    comm = os.path.join(base, "commodity_cache.npz")

    check_complete(clin)
    check_complete(comm)

    acc = {
        "clin_d1":  run_evaluation(clin, LinearDiscriminantAnalysis(), "within"),
        "clin_d29": run_evaluation(clin, LinearDiscriminantAnalysis(), "cross_max"),
        "comm_d1":  run_evaluation(comm, LinearDiscriminantAnalysis(), "within"),
        "comm_d29": run_evaluation(comm, LinearDiscriminantAnalysis(), "cross_max"),
    }

    subjects = sorted(set.intersection(*(set(v) for v in acc.values())))
    if len(subjects) != N_SUBJECTS:
        print(f"FAIL: expected {N_SUBJECTS} subjects in every cell, found {len(subjects)}")
        sys.exit(1)
    arr = {k: np.array([v[s] for s in subjects]) for k, v in acc.items()}

    contrasts = [
        ("clinical vs commodity, Day 1",  "clin_d1",  "comm_d1"),
        ("clinical vs commodity, Day 29", "clin_d29", "comm_d29"),
        ("Day 1 vs Day 29, clinical",     "clin_d1",  "clin_d29"),
        ("Day 1 vs Day 29, commodity",    "comm_d1",  "comm_d29"),
    ]

    print(f"LDA cell means (n={len(subjects)}): "
          + ", ".join(f"{k} {100 * arr[k].mean():.1f}%" for k in arr))
    print(f"\n{'contrast':<32} {'mean diff (pts)':>16} {'p':>11}  Bonferroni {ALPHA}")
    print("-" * 76)
    for label, a, b in contrasts:
        p = wilcoxon(arr[a], arr[b]).pvalue
        diff = 100 * (arr[a] - arr[b]).mean()
        print(f"{label:<32} {diff:>+16.2f} {p:>11.2e}  "
              f"{'significant' if p < ALPHA else 'not significant'}")


if __name__ == "__main__":
    main()
