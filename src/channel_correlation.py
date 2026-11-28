"""Pearson correlations supporting the bipolar derivation (paper Section 3.3).

Reproduces the representative-trial check recorded in notes/channel_selection.md
(session 1, participant 1, gesture 16, trial 1):

  raw monopolar F1 vs F9           (paper: r = 0.987)
  bipolar F1F9 vs F5F13            (paper: r = 0.139)

Both are computed on the 0.5 s edge-trimmed signal, as used by the pipeline.
The untrimmed raw correlation is also printed for reference.
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
    r_raw_full = np.corrcoef(f1, f9)[0, 1]
    r_raw = np.corrcoef(f1[TRIM:-TRIM], f9[TRIM:-TRIM])[0, 1]

    bip = load_trial(session=SESSION, participant=PARTICIPANT,
                     gesture=GESTURE, trial=TRIAL)["commodity"]
    r_bip = np.corrcoef(bip[:, 0], bip[:, 1])[0, 1]

    print(f"trial: session {SESSION}, participant {PARTICIPANT}, "
          f"gesture {GESTURE}, trial {TRIAL}")
    print(f"raw F1 vs F9, trimmed       r = {r_raw:.3f}   (paper 0.987)")
    print(f"raw F1 vs F9, untrimmed     r = {r_raw_full:.3f}")
    print(f"bipolar F1F9 vs F5F13       r = {r_bip:.3f}   (paper 0.139)")


if __name__ == "__main__":
    main()
