"""Pearson correlations supporting the bipolar derivation (paper Section 3.3).

Reproduces the representative-trial check recorded in notes/channel_selection.md
(session 1, participant 1, gesture 16, trial 1): raw monopolar F1 vs F9, and
bipolar F1F9 vs F5F13.

Both pairs are computed over the same samples, on the full trial and on the
0.5 s edge-trimmed signal the pipeline uses, so the before/after comparison is
like for like. The paper reports the trimmed pair.
"""
import os

import numpy as np
import wfdb

from preprocess import load_trial

SESSION, PARTICIPANT, GESTURE, TRIAL = 1, 1, 16, 1
TRIM = 1024          # 0.5 s at 2048 Hz, as in preprocess.load_trial


def raw_monopolar():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        here, "..", "data", "grabmyo", f"Session{SESSION}",
        f"session{SESSION}_participant{PARTICIPANT}",
        f"session{SESSION}_participant{PARTICIPANT}_gesture{GESTURE}_trial{TRIAL}")
    rec = wfdb.rdrecord(path)
    names = list(rec.sig_name)
    return rec.p_signal[:, names.index("F1")], rec.p_signal[:, names.index("F9")]


def main():
    f1, f9 = raw_monopolar()
    # Bipolar derivations from the same raw record, so both intervals use
    # identical samples. load_trial's commodity output is the trimmed version.
    here = os.path.dirname(os.path.abspath(__file__))
    rec = wfdb.rdrecord(os.path.join(
        here, "..", "data", "grabmyo", f"Session{SESSION}",
        f"session{SESSION}_participant{PARTICIPANT}",
        f"session{SESSION}_participant{PARTICIPANT}_gesture{GESTURE}_trial{TRIAL}"))
    names = list(rec.sig_name)
    col = lambda n: rec.p_signal[:, names.index(n)]
    bip1 = col("F1") - col("F9")
    bip2 = col("F5") - col("F13")

    # Cross-check: the trimmed bipolar pair must equal the pipeline's own.
    pipe = load_trial(session=SESSION, participant=PARTICIPANT,
                      gesture=GESTURE, trial=TRIAL)["commodity"]
    assert np.allclose(pipe[:, 0], bip1[TRIM:-TRIM]) and \
        np.allclose(pipe[:, 1], bip2[TRIM:-TRIM]), "bipolar mismatch with load_trial"

    r = lambda a, b: np.corrcoef(a, b)[0, 1]
    print(f"trial: session {SESSION}, participant {PARTICIPANT}, "
          f"gesture {GESTURE}, trial {TRIAL}\n")
    print(f"{'interval':<22} {'raw F1 vs F9':>13} {'bipolar F1F9 vs F5F13':>23}")
    print(f"{'full trial':<22} {r(f1, f9):>13.3f} {r(bip1, bip2):>23.3f}")
    print(f"{'trimmed (pipeline)':<22} {r(f1[TRIM:-TRIM], f9[TRIM:-TRIM]):>13.3f} "
          f"{r(bip1[TRIM:-TRIM], bip2[TRIM:-TRIM]):>23.3f}")


if __name__ == "__main__":
    main()
